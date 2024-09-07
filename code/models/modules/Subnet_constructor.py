import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from basicsr.archs.arch_util import flow_warp, ResidualBlockNoBN
from models.modules.module_util import initialize_weights_xavier

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.H = None

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

class DenseBlock_v2(nn.Module):
    def __init__(self, channel_in, channel_out, groups, init='xavier', gc=32, bias=True):
        super(DenseBlock_v2, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.conv_final = nn.Conv2d(channel_out*groups, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        mutil.initialize_weights(self.conv_final, 0)

    def forward(self, x):
        res = []
        for xi in x:
            x1 = self.lrelu(self.conv1(xi))
            x2 = self.lrelu(self.conv2(torch.cat((xi, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((xi, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((xi, x1, x2, x3), 1)))
            x5 = self.lrelu(self.conv5(torch.cat((xi, x1, x2, x3, x4), 1)))
            res.append(x5)
        res = torch.cat(res, dim=1)
        res = self.conv_final(res)

        return res

def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out, groups=None):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            elif init == 'xavier_v2':
                return DenseBlock_v2(channel_in, channel_out, groups, 'xavier')
            else:
                return DenseBlock(channel_in, channel_out)
        else:
            return None

    return constructor
