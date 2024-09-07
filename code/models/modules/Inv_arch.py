import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_util import initialize_weights_xavier
from torch.nn import init
from .common import DWT,IWT
import cv2
from basicsr.archs.arch_util import flow_warp
from models.modules.Subnet_constructor import subnet
import numpy as np

from pdb import set_trace as stx
import numbers

from einops import rearrange
from models.bitnetwork.Encoder_U import DW_Encoder
from models.bitnetwork.Decoder_U import DW_Decoder


## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=4, bias=False, LayerNorm_type="withbias"):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

dwt=DWT()
iwt=IWT()

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

def thops_mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64, model='MIMO-VRN'):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # honestly, there's no significant difference between ReLU and leaky ReLU in terms of performance here
        # but this is how we trained the model in the first place and what we reported in the paper
        if model == 'LSTM-VRN':
            self.relu = nn.ReLU(inplace=True)
        elif model == 'MIMO-VRN':
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, subnet_constructor_v2, channel_num_ho, channel_num_hi, groups, clamp=1.):
        super(InvBlock, self).__init__()
        self.split_len1 = channel_num_ho  # channel_split_num
        self.split_len2 = channel_num_hi  # channel_num - channel_split_num
        self.clamp = clamp

        self.F = subnet_constructor_v2(self.split_len2, self.split_len1, groups=groups)
        self.NF = NAFBlock(self.split_len2)
        if groups == 1: 
            self.G = subnet_constructor(self.split_len1, self.split_len2, groups=groups)
            self.NG = NAFBlock(self.split_len1)
            self.H = subnet_constructor(self.split_len1, self.split_len2, groups=groups)
            self.NH = NAFBlock(self.split_len1)
        else:
            self.G = subnet_constructor(self.split_len1, self.split_len2)
            self.NG = NAFBlock(self.split_len1)
            self.H = subnet_constructor(self.split_len1, self.split_len2)
            self.NH = NAFBlock(self.split_len1)

    def forward(self, x1, x2, rev=False):
        if not rev:
            y1 = x1 + self.NF(self.F(x2))
            self.s = self.clamp * (torch.sigmoid(self.NH(self.H(y1))) * 2 - 1)
            y2 = [x2i.mul(torch.exp(self.s)) + self.NG(self.G(y1)) for x2i in x2]
        else:
            self.s = self.clamp * (torch.sigmoid(self.NH(self.H(x1))) * 2 - 1)
            y2 = [(x2i - self.NG(self.G(x1))).div(torch.exp(self.s)) for x2i in x2]
            y1 = x1 - self.NF(self.F(y2))

        return y1, y2  # torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class InvNN(nn.Module):
    def __init__(self, channel_in_ho=3, channel_in_hi=3, subnet_constructor=None, subnet_constructor_v2=None, block_num=[], down_num=2, groups=None):
        super(InvNN, self).__init__()
        operations = []

        current_channel_ho = channel_in_ho
        current_channel_hi = channel_in_hi
        for i in range(down_num):
            for j in range(block_num[i]):
                b = InvBlock(subnet_constructor, subnet_constructor_v2, current_channel_ho, current_channel_hi, groups=groups)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, x_h, rev=False, cal_jacobian=False):
        # 		out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                x, x_h = op.forward(x, x_h, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(x, rev)
        else:
            for op in reversed(self.operations):
                x, x_h = op.forward(x, x_h, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(x, rev)

        if cal_jacobian:
            return x, x_h, jacobian
        else:
            return x, x_h

class PredictiveModuleMIMO(nn.Module):
    def __init__(self, channel_in, nf, block_num_rbm=8, block_num_trans=4):
        super(PredictiveModuleMIMO, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        res_block = []
        trans_block = []
        for i in range(block_num_rbm):
            res_block.append(ResidualBlockNoBN(nf))
        for j in range(block_num_trans):
            trans_block.append(TransformerBlock(nf))

        self.res_block = nn.Sequential(*res_block)
        self.transformer_block = nn.Sequential(*trans_block)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_block(x)
        res = self.transformer_block(x) + x

        return res

class ConvRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, init_zero=False):
        super(ConvRelu, self).__init__()
        self.init_zero = init_zero
        if self.init_zero:
            self.layers = nn.Conv2d(channels_in, channels_out, 3, stride, padding=1)

        else:
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.layers(x)
    
class PredictiveModuleBit(nn.Module):
    def __init__(self, channel_in, nf, block_num_rbm=4, block_num_trans=2):
        super(PredictiveModuleBit, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        res_block = []
        trans_block = []
        for i in range(block_num_rbm):
            res_block.append(ResidualBlockNoBN(nf))
        for j in range(block_num_trans):
            trans_block.append(TransformerBlock(nf))
        
        blocks = 4
        layers = [ConvRelu(nf, 1, 2)]
        for _ in range(blocks - 1):
            layer = ConvRelu(1, 1, 2)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

        self.res_block = nn.Sequential(*res_block)
        self.transformer_block = nn.Sequential(*trans_block)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_block(x)
        res = self.transformer_block(x) + x
        res = self.layers(res)

        return res


##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=12,prompt_len=3,prompt_size = 36,lin_dim = 12):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class PredictiveModuleMIMO_prompt(nn.Module):
    def __init__(self, channel_in, nf, prompt_len=3, block_num_rbm=8, block_num_trans=4):
        super(PredictiveModuleMIMO_prompt, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        res_block = []
        trans_block = []
        for i in range(block_num_rbm):
            res_block.append(ResidualBlockNoBN(nf))
        for j in range(block_num_trans):
            trans_block.append(TransformerBlock(nf))

        self.res_block = nn.Sequential(*res_block)
        self.transformer_block = nn.Sequential(*trans_block)
        self.prompt = PromptGenBlock(prompt_dim=nf,prompt_len=prompt_len,prompt_size = 36,lin_dim = nf)
        self.fuse = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_block(x)
        res = self.transformer_block(x) + x
        prompt = self.prompt(res)

        result = self.fuse(torch.cat([res, prompt], dim=1))

        return result

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def gauss_noise_mul(shape):
    noise = torch.randn(shape).cuda()

    return noise

class PredictiveModuleBit_prompt(nn.Module):
    def __init__(self, channel_in, nf, prompt_length, block_num_rbm=4, block_num_trans=2):
        super(PredictiveModuleBit_prompt, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        res_block = []
        trans_block = []
        for i in range(block_num_rbm):
            res_block.append(ResidualBlockNoBN(nf))
        for j in range(block_num_trans):
            trans_block.append(TransformerBlock(nf))
        
        blocks = 4
        layers = [ConvRelu(nf, 1, 2)]
        for _ in range(blocks - 1):
            layer = ConvRelu(1, 1, 2)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

        self.res_block = nn.Sequential(*res_block)
        self.transformer_block = nn.Sequential(*trans_block)
        self.prompt = PromptGenBlock(prompt_dim=nf,prompt_len=prompt_length,prompt_size = 36,lin_dim = nf)
        self.fuse = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_block(x)
        res = self.transformer_block(x) + x
        prompt = self.prompt(res)
        res = self.fuse(torch.cat([res, prompt], dim=1))
        res = self.layers(res)

        return res

class VSN(nn.Module):
    def __init__(self, opt, subnet_constructor=None, subnet_constructor_v2=None, down_num=2):
        super(VSN, self).__init__()
        self.model = opt['model']
        self.mode = opt['mode']
        opt_net = opt['network_G']
        self.num_image = opt['num_image']
        self.gop = opt['gop']
        self.channel_in = opt_net['in_nc'] * self.gop
        self.channel_out = opt_net['out_nc'] * self.gop
        self.channel_in_hi = opt_net['in_nc'] * self.gop
        self.channel_in_ho = opt_net['in_nc'] * self.gop
        self.message_len = opt['message_length']

        self.block_num = opt_net['block_num']
        self.block_num_rbm = opt_net['block_num_rbm']
        self.block_num_trans = opt_net['block_num_trans']
        self.nf = self.channel_in_hi 
        
        self.bitencoder = DW_Encoder(self.message_len, attention = "se")
        self.bitdecoder = DW_Decoder(self.message_len, attention = "se")
        self.irn = InvNN(self.channel_in_ho, self.channel_in_hi, subnet_constructor, subnet_constructor_v2, self.block_num, down_num, groups=self.num_image)

        if opt['prompt']:
            self.pm = PredictiveModuleMIMO_prompt(self.channel_in_ho, self.nf* self.num_image, opt['prompt_len'], block_num_rbm=self.block_num_rbm, block_num_trans=self.block_num_trans)
        else:
            self.pm = PredictiveModuleMIMO(self.channel_in_ho, self.nf* self.num_image, opt['prompt_len'], block_num_rbm=self.block_num_rbm, block_num_trans=self.block_num_trans)
            self.BitPM = PredictiveModuleBit(3, 4, block_num_rbm=4, block_num_trans=2)


    def forward(self, x, x_h=None, message=None, rev=False, hs=[], direction='f'):
        if not rev:
            if self.mode == "image":
                out_y, out_y_h = self.irn(x, x_h, rev)
                out_y = iwt(out_y)
                encoded_image = self.bitencoder(out_y, message)          
                return out_y, encoded_image
            
            elif self.mode == "bit":
                out_y = iwt(x)
                encoded_image = self.bitencoder(out_y, message)            
                return out_y, encoded_image

        else:
            if self.mode == "image":
                recmessage = self.bitdecoder(x)

                x = dwt(x)
                out_z = self.pm(x).unsqueeze(1)
                out_z_new = out_z.view(-1, self.num_image, self.channel_in, x.shape[-2], x.shape[-1])
                out_z_new = [out_z_new[:,i] for i in range(self.num_image)]
                out_x, out_x_h = self.irn(x, out_z_new, rev)

                return out_x, out_x_h, out_z, recmessage
            
            elif self.mode == "bit":
                recmessage = self.bitdecoder(x)
                return recmessage

