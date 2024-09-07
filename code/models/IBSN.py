import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, ReconstructionMsgLoss
from models.modules.Quantization import Quantization
from .modules.common import DWT,IWT
from utils.jpegtest import JpegTest
from utils.JPEG import DiffJPEG
import utils.util as util


import numpy as np
import random
import cv2
import time

logger = logging.getLogger('base')
dwt=DWT()
iwt=IWT()

from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from diffusers import RePaintPipeline, RePaintScheduler

class Model_VSN(BaseModel):
    def __init__(self, opt):
        super(Model_VSN, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.gop = opt['gop']
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.opt_net = opt['network_G']
        self.center = self.gop // 2
        self.num_image = opt['num_image']
        self.mode = opt["mode"]
        self.idxx = 0

        self.netG = networks.define_G_v2(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if not self.opt['hide']:
            file_path = "bit_sequence.txt"

            data_list = []

            with open(file_path, "r") as file:
                for line in file:
                    data = [int(bit) for bit in line.strip()]
                    data_list.append(data)
            
            self.msg_list = data_list

        if self.opt['sdinpaint']:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
            ).to("cuda")
        
        if self.opt['controlnetinpaint']:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float32
            ).to("cuda")
            self.pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
            ).to("cuda")
        
        if self.opt['sdxl']:
            self.pipe_sdxl = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to("cuda")
        
        if self.opt['repaint']:
            self.scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
            self.pipe_repaint = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=self.scheduler)
            self.pipe_repaint = self.pipe_repaint.to("cuda")

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.Reconstruction_center = ReconstructionLoss(losstype="center")
            self.Reconstruction_msg = ReconstructionMsgLoss(losstype=self.opt['losstype'])

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []

            if self.mode == "image":
                for k, v in self.netG.named_parameters():
                    if (k.startswith('module.irn') or k.startswith('module.pm')) and v.requires_grad: 
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            elif self.mode == "bit":
                for k, v in self.netG.named_parameters():
                    if (k.startswith('module.bitencoder') or k.startswith('module.bitdecoder')) and v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))


            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  
        self.real_H = data['GT'].to(self.device)
        self.mes = data['MES']

    def init_hidden_state(self, z):
        b, c, h, w = z.shape
        h_t = []
        c_t = []
        for _ in range(self.opt_net['block_num_rbm']):
            h_t.append(torch.zeros([b, c, h, w]).cuda())
            c_t.append(torch.zeros([b, c, h, w]).cuda())
        memory = torch.zeros([b, c, h, w]).cuda()

        return h_t, c_t, memory

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        return l_forw_fit

    def loss_back_rec(self, out, x):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
        return l_back_rec
    
    def loss_back_rec_mul(self, out, x):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
        return l_back_rec

    def optimize_parameters(self, current_step):
        self.optimizer_G.zero_grad()
      
        b, n, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = self.gop // 2

        message = torch.Tensor(np.random.choice([-0.5, 0.5], (self.ref_L.shape[0], self.opt['message_length']))).to(self.device)

        add_noise = self.opt['addnoise']
        add_jpeg = self.opt['addjpeg']
        add_possion = self.opt['addpossion']
        add_sdinpaint = self.opt['sdinpaint']
        degrade_shuffle = self.opt['degrade_shuffle']

        self.host = self.real_H[:, center - intval:center + intval + 1]
        self.secret = self.ref_L[:, :, center - intval:center + intval + 1]
        self.output, container = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=dwt(self.secret[:,0].reshape(b, -1, h, w)), message=message)

        Gt_ref = self.real_H[:, center - intval:center + intval + 1].detach()

        y_forw = container

        l_forw_fit = self.loss_forward(y_forw, self.host[:,0])


        if degrade_shuffle:
            import random
            choice = random.randint(0, 2)
            
            if choice == 0:
                NL = float((np.random.randint(1, 16))/255)
                noise = np.random.normal(0, NL, y_forw.shape)
                torchnoise = torch.from_numpy(noise).cuda().float()
                y_forw = y_forw + torchnoise

            elif choice == 1:
                NL = int(np.random.randint(70,95))
                self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
                y_forw = self.DiffJPEG(y_forw)
            
            elif choice == 2:
                vals = 10**4
                if random.random() < 0.5:
                    noisy_img_tensor = torch.poisson(y_forw * vals) / vals
                else:
                    img_gray_tensor = torch.mean(y_forw, dim=0, keepdim=True)
                    noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                    noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

                y_forw = torch.clamp(noisy_img_tensor, 0, 1)

        else:

            if add_noise:
                NL = float((np.random.randint(1,16))/255)
                noise = np.random.normal(0, NL, y_forw.shape)
                torchnoise = torch.from_numpy(noise).cuda().float()
                y_forw = y_forw + torchnoise

            elif add_jpeg:
                NL = int(np.random.randint(70,95))
                self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
                y_forw = self.DiffJPEG(y_forw)

            elif add_possion:
                vals = 10**4
                if random.random() < 0.5:
                    noisy_img_tensor = torch.poisson(y_forw * vals) / vals
                else:
                    img_gray_tensor = torch.mean(y_forw, dim=0, keepdim=True)
                    noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                    noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

                y_forw = torch.clamp(noisy_img_tensor, 0, 1)

        y = self.Quantization(y_forw)
        all_zero = torch.zeros(message.shape).to(self.device)

        if self.mode == "image":
            out_x, out_x_h, out_z, recmessage = self.netG(x=y, message=all_zero, rev=True)
            out_x = iwt(out_x)
            out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]

            l_back_rec = self.loss_back_rec(out_x, self.host[:,0])
            out_x_h = torch.stack(out_x_h, dim=1)

            l_center_x = self.loss_back_rec(out_x_h[:, 0], self.secret[:,0].reshape(b, -1, h, w))

            recmessage = torch.clamp(recmessage, -0.5, 0.5)

            l_msg = self.Reconstruction_msg(message, recmessage)

            loss = l_forw_fit*2 + l_back_rec + l_center_x*4

            loss.backward()

            if self.train_opt['lambda_center'] != 0:
                self.log_dict['l_center_x'] = l_center_x.item()

            # set log
            self.log_dict['l_back_rec'] = l_back_rec.item()
            self.log_dict['l_forw_fit'] = l_forw_fit.item()
            self.log_dict['l_msg'] = l_msg.item()
            
            self.log_dict['l_h'] = (l_center_x*10).item()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_G.step()

        elif self.mode == "bit":
            recmessage = self.netG(x=y, message=all_zero, rev=True)

            recmessage = torch.clamp(recmessage, -0.5, 0.5)

            l_msg = self.Reconstruction_msg(message, recmessage)
            
            lambda_msg = self.train_opt['lambda_msg']

            loss = l_msg * lambda_msg + l_forw_fit

            loss.backward()

            # set log
            self.log_dict['l_forw_fit'] = l_forw_fit.item()
            self.log_dict['l_msg'] = l_msg.item()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_G.step()

    def test(self, image_id):
        self.netG.eval()
        add_noise = self.opt['addnoise']
        add_jpeg = self.opt['addjpeg']
        add_possion = self.opt['addpossion']
        add_sdinpaint = self.opt['sdinpaint']
        add_controlnet = self.opt['controlnetinpaint']
        add_sdxl = self.opt['sdxl']
        add_repaint = self.opt['repaint']
        degrade_shuffle = self.opt['degrade_shuffle']

        with torch.no_grad():
            forw_L = []
            forw_L_h = []
            fake_H = []
            fake_H_h = []
            pred_z = []
            recmsglist = []
            msglist = []
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2
            b, n, t, c, h, w = self.ref_L.shape
            id=0
            # forward downscaling
            self.host = self.real_H[:, center - intval+id:center + intval + 1+id]
            self.secret = self.ref_L[:, :, center - intval+id:center + intval + 1+id]
            self.secret = [dwt(self.secret[:,i].reshape(b, -1, h, w)) for i in range(n)]

            messagenp = np.random.choice([-0.5, 0.5], (self.ref_L.shape[0], self.opt['message_length']))

            message = torch.Tensor(messagenp).to(self.device)

            if self.opt['bitrecord']:
                mymsg = message.clone()

                mymsg[mymsg>0] = 1
                mymsg[mymsg<0] = 0
                mymsg = mymsg.squeeze(0).to(torch.int)

                bit_list = mymsg.tolist()

                bit_string = ''.join(map(str, bit_list))

                file_name = "bit_sequence.txt"

                with open(file_name, "a") as file:
                    file.write(bit_string + "\n")

            if self.opt['hide']:
                self.output, container = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=self.secret, message=message)
                y_forw = container
            else:
                
                message = torch.tensor(self.msg_list[image_id]).unsqueeze(0).cuda()
                self.output = self.host
                y_forw = self.output.squeeze(1)

            if add_sdinpaint:
                import random
                from PIL import Image
                prompt = ""

                b, _, _, _ = y_forw.shape
                
                image_batch = y_forw.permute(0, 2, 3, 1).detach().cpu().numpy()
                forw_list = []

                for j in range(b):
                    i = image_id + 1
                    masksrc = "../dataset/valAGE-Set-Mask/"
                    mask_image = Image.open(masksrc + str(i).zfill(4) + ".png").convert("L")
                    mask_image = mask_image.resize((512, 512))
                    h, w = mask_image.size
                    
                    image = image_batch[j, :, :, :]
                    image_init = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
                    image_inpaint = self.pipe(prompt=prompt, image=image_init, mask_image=mask_image, height=w, width=h).images[0]
                    image_inpaint = np.array(image_inpaint) / 255.
                    mask_image = np.array(mask_image)
                    mask_image = np.stack([mask_image] * 3, axis=-1) / 255.
                    mask_image = mask_image.astype(np.uint8)
                    image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
                    forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))
                
                y_forw = torch.stack(forw_list, dim=0).float().cuda()

            if add_controlnet:
                from diffusers.utils import load_image
                from PIL import Image

                b, _, _, _ = y_forw.shape
                forw_list = []
                
                image_batch = y_forw.permute(0, 2, 3, 1).detach().cpu().numpy()
                generator = torch.Generator(device="cuda").manual_seed(1)

                for j in range(b):
                    i = image_id + 1
                    mask_path = "../dataset/valAGE-Set-Mask/" + str(i).zfill(4) + ".png"
                    mask_image = load_image(mask_path)
                    mask_image = mask_image.resize((512, 512))
                    image_init = image_batch[j, :, :, :]
                    image_init1 = Image.fromarray((image_init * 255).astype(np.uint8), mode = "RGB")
                    image_mask = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

                    assert image_init.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
                    image_init[image_mask > 0.5] = -1.0  # set as masked pixel
                    image = np.expand_dims(image_init, 0).transpose(0, 3, 1, 2)
                    control_image = torch.from_numpy(image)

                    # generate image
                    image_inpaint = self.pipe_control(
                        "",
                        num_inference_steps=20,
                        generator=generator,
                        eta=1.0,
                        image=image_init1,
                        mask_image=image_mask,
                        control_image=control_image,
                    ).images[0]
                    
                    image_inpaint = np.array(image_inpaint) / 255.
                    image_mask = np.stack([image_mask] * 3, axis=-1)
                    image_mask = image_mask.astype(np.uint8)
                    image_fuse = image_init * (1 - image_mask) + image_inpaint * image_mask
                    forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))

                y_forw = torch.stack(forw_list, dim=0).float().cuda()
            
            if add_sdxl:
                import random
                from PIL import Image
                from diffusers.utils import load_image
                prompt = ""

                b, _, _, _ = y_forw.shape
                
                image_batch = y_forw.permute(0, 2, 3, 1).detach().cpu().numpy()
                forw_list = []

                for j in range(b):
                    i = image_id + 1
                    masksrc = "../dataset/valAGE-Set-Mask/"
                    mask_image = load_image(masksrc + str(i).zfill(4) + ".png").convert("RGB")
                    mask_image = mask_image.resize((512, 512))
                    h, w = mask_image.size
                    
                    image = image_batch[j, :, :, :]
                    image_init = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
                    image_inpaint = self.pipe_sdxl(
                        prompt=prompt, image=image_init, mask_image=mask_image, num_inference_steps=50, strength=0.80, target_size=(512, 512)
                    ).images[0]
                    image_inpaint = image_inpaint.resize((512, 512))
                    image_inpaint = np.array(image_inpaint) / 255.
                    mask_image = np.array(mask_image) / 255.
                    mask_image = mask_image.astype(np.uint8)
                    image_fuse = image * (1 - mask_image) + image_inpaint * mask_image
                    forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))
                
                y_forw = torch.stack(forw_list, dim=0).float().cuda()

            
            if add_repaint:
                from PIL import Image
                
                b, _, _, _ = y_forw.shape
                
                image_batch = y_forw.permute(0, 2, 3, 1).detach().cpu().numpy()
                forw_list = []

                generator = torch.Generator(device="cuda").manual_seed(0)
                for j in range(b):
                    i = image_id + 1
                    masksrc = "../dataset/valAGE-Set-Mask/" + str(i).zfill(4) + ".png"
                    mask_image = Image.open(masksrc).convert("RGB")
                    mask_image = mask_image.resize((256, 256))
                    mask_image = Image.fromarray(255 - np.array(mask_image))
                    image = image_batch[j, :, :, :]
                    original_image = Image.fromarray((image * 255).astype(np.uint8), mode = "RGB")
                    original_image = original_image.resize((256, 256))
                    output = self.pipe_repaint(
                        image=original_image,
                        mask_image=mask_image,
                        num_inference_steps=150,
                        eta=0.0,
                        jump_length=10,
                        jump_n_sample=10,
                        generator=generator,
                    )
                    image_inpaint = output.images[0]
                    image_inpaint = image_inpaint.resize((512, 512))
                    image_inpaint = np.array(image_inpaint) / 255.
                    mask_image = mask_image.resize((512, 512))
                    mask_image = np.array(mask_image) / 255.
                    mask_image = mask_image.astype(np.uint8)
                    image_fuse = image * mask_image + image_inpaint * (1 - mask_image)
                    forw_list.append(torch.from_numpy(image_fuse).permute(2, 0, 1))
                
                y_forw = torch.stack(forw_list, dim=0).float().cuda()

            if degrade_shuffle:
                import random
                choice = random.randint(0, 2)
                
                if choice == 0:
                    NL = float((np.random.randint(1,5))/255)
                    noise = np.random.normal(0, NL, y_forw.shape)
                    torchnoise = torch.from_numpy(noise).cuda().float()
                    y_forw = y_forw + torchnoise

                elif choice == 1:
                    NL = 90
                    self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
                    y_forw = self.DiffJPEG(y_forw)
                
                elif choice == 2:
                    vals = 10**4
                    if random.random() < 0.5:
                        noisy_img_tensor = torch.poisson(y_forw * vals) / vals
                    else:
                        img_gray_tensor = torch.mean(y_forw, dim=0, keepdim=True)
                        noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                        noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

                    y_forw = torch.clamp(noisy_img_tensor, 0, 1)

            else:

                if add_noise:
                    NL = self.opt['noisesigma'] / 255.0
                    noise = np.random.normal(0, NL, y_forw.shape)
                    torchnoise = torch.from_numpy(noise).cuda().float()
                    y_forw = y_forw + torchnoise

                elif add_jpeg:
                    Q = self.opt['jpegfactor']
                    self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(Q)).cuda()
                    y_forw = self.DiffJPEG(y_forw)

                elif add_possion:
                    vals = 10**4
                    if random.random() < 0.5:
                        noisy_img_tensor = torch.poisson(y_forw * vals) / vals
                    else:
                        img_gray_tensor = torch.mean(y_forw, dim=0, keepdim=True)
                        noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                        noisy_img_tensor = y_forw + (noisy_gray_tensor - img_gray_tensor)

                    y_forw = torch.clamp(noisy_img_tensor, 0, 1)

            # backward upscaling
            if self.opt['hide']:
                y = self.Quantization(y_forw)
            else:
                y = y_forw

            if self.mode == "image":
                out_x, out_x_h, out_z, recmessage = self.netG(x=y, rev=True)
                out_x = iwt(out_x)

                out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
                out_x = out_x.reshape(-1, self.gop, 3, h, w)
                out_x_h = torch.stack(out_x_h, dim=1)
                out_x_h = out_x_h.reshape(-1, 1, self.gop, 3, h, w)

                forw_L.append(y_forw)
                fake_H.append(out_x[:, self.gop//2])
                fake_H_h.append(out_x_h[:,:, self.gop//2])
                recmsglist.append(recmessage)
                msglist.append(message)
            
            elif self.mode == "bit":
                recmessage = self.netG(x=y, rev=True)
                forw_L.append(y_forw)
                recmsglist.append(recmessage)
                msglist.append(message)

        if self.mode == "image":
            self.fake_H = torch.clamp(torch.stack(fake_H, dim=1),0,1)
            self.fake_H_h = torch.clamp(torch.stack(fake_H_h, dim=2),0,1)

        self.forw_L = torch.clamp(torch.stack(forw_L, dim=1),0,1)
        remesg = torch.clamp(torch.stack(recmsglist, dim=0),-0.5,0.5)

        if self.opt['hide']:
            mesg = torch.clamp(torch.stack(msglist, dim=0),-0.5,0.5)
        else:
            mesg = torch.stack(msglist, dim=0)

        self.recmessage = remesg.clone()
        self.recmessage[remesg > 0] = 1
        self.recmessage[remesg <= 0] = 0

        self.message = mesg.clone()
        self.message[mesg > 0] = 1
        self.message[mesg <= 0] = 0

        self.netG.train()


    def image_hiding(self, ):
        self.netG.eval()
        with torch.no_grad():
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2
            b, n, t, c, h, w = self.ref_L.shape
            id=0
            # forward downscaling
            self.host = self.real_H[:, center - intval+id:center + intval + 1+id]
            self.secret = self.ref_L[:, :, center - intval+id:center + intval + 1+id]
            self.secret = [dwt(self.secret[:,i].reshape(b, -1, h, w)) for i in range(n)]

            message = torch.Tensor(self.mes).to(self.device)

            self.output, container = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=self.secret, message=message)
            y_forw = container

            result = torch.clamp(y_forw,0,1)

            lr_img = util.tensor2img(result)

            return lr_img

    def image_recovery(self, number):
        self.netG.eval()
        with torch.no_grad():
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2
            b, n, t, c, h, w = self.ref_L.shape
            id=0
            # forward downscaling
            self.host = self.real_H[:, center - intval+id:center + intval + 1+id]
            self.secret = self.ref_L[:, :, center - intval+id:center + intval + 1+id]
            template = self.secret.reshape(b, -1, h, w)
            self.secret = [dwt(self.secret[:,i].reshape(b, -1, h, w)) for i in range(n)]

            self.output = self.host
            y_forw = self.output.squeeze(1)

            y = self.Quantization(y_forw)

            out_x, out_x_h, out_z, recmessage = self.netG(x=y, rev=True)
            out_x = iwt(out_x)

            out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
            out_x = out_x.reshape(-1, self.gop, 3, h, w)
            out_x_h = torch.stack(out_x_h, dim=1)
            out_x_h = out_x_h.reshape(-1, 1, self.gop, 3, h, w)

            rec_loc = out_x_h[:,:, self.gop//2]
            # from PIL import Image
            # tmp = util.tensor2img(rec_loc)
            # save
            residual = torch.abs(template - rec_loc)
            binary_residual = (residual > number).float()
            residual = util.tensor2img(binary_residual)
            mask = np.sum(residual, axis=2)
            # print(mask)

            remesg = torch.clamp(recmessage,-0.5,0.5)
            remesg[remesg > 0] = 1
            remesg[remesg <= 0] = 0

            return mask, remesg
        
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        b, n, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = self.gop // 2
        out_dict = OrderedDict()
        LR_ref = self.ref_L[:, :, center - intval:center + intval + 1].detach()[0].float().cpu()
        LR_ref = torch.chunk(LR_ref, self.num_image, dim=0)
        out_dict['LR_ref'] = [image.squeeze(0) for image in LR_ref]
        
        if self.mode == "image":
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
            SR_h = self.fake_H_h.detach()[0].float().cpu()
            SR_h = torch.chunk(SR_h, self.num_image, dim=0)
            out_dict['SR_h'] = [image.squeeze(0) for image in SR_h]
        
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H[:, center - intval:center + intval + 1].detach()[0].float().cpu()
        out_dict['message'] = self.message
        out_dict['recmessage'] = self.recmessage

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    
    def load_test(self,load_path_G):
        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
