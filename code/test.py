import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def cal_pnsr(sr_img, gt_img):
    # calculate PSNR
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.

    psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)

    return psnr

def get_min_avg_and_indices(nums):
    # Get the indices of the smallest 1000 elements
    indices = sorted(range(len(nums)), key=lambda i: nums[i])[:900]
    
    # Calculate the average of these elements
    avg = sum(nums[i] for i in indices) / 900
    
    # Write the indices to a txt file
    with open("indices.txt", "w") as file:
        for index in indices:
            file.write(str(index) + "\n")
    
    return avg


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--ckpt', type=str, default='/userhome/NewIBSN/EditGuard_open/checkpoints/clean.pth', help='Path to pre-trained model.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        print("phase", phase)
        if phase == 'TD':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    # create model
    model = create_model(opt)
    model.load_test(args.ckpt)
            
    # validation
    avg_psnr = 0.0
    avg_psnr_h = [0.0]*opt['num_image']
    avg_psnr_lr = 0.0
    biterr = []
    idx = 0
    for image_id, val_data in enumerate(val_loader):
        img_dir = os.path.join('results',opt['name'])
        util.mkdir(img_dir)

        model.feed_data(val_data)
        model.test(image_id)

        visuals = model.get_current_visuals()

        t_step = visuals['SR'].shape[0]
        idx += t_step
        n = len(visuals['SR_h'])

        a = visuals['recmessage'][0]
        b = visuals['message'][0]

        bitrecord = util.decoded_message_error_rate_batch(a, b)
        print(bitrecord)
        biterr.append(bitrecord)

        for i in range(t_step):

            sr_img = util.tensor2img(visuals['SR'][i])  # uint8
            sr_img_h = []
            for j in range(n):
                sr_img_h.append(util.tensor2img(visuals['SR_h'][j][i]))  # uint8
            gt_img = util.tensor2img(visuals['GT'][i])  # uint8
            lr_img = util.tensor2img(visuals['LR'][i])
            lrgt_img = []
            for j in range(n):
                lrgt_img.append(util.tensor2img(visuals['LR_ref'][j][i]))

            # Save SR images for reference
            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'SR'))
            util.save_img(sr_img, save_img_path)

            for j in range(n):
                save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:d}_{:s}.png'.format(image_id, i, j, 'SR_h'))
                util.save_img(sr_img_h[j], save_img_path)

            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'GT'))
            util.save_img(gt_img, save_img_path)

            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'LR'))
            util.save_img(lr_img, save_img_path)

            for j in range(n):
                save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:d}_{:s}.png'.format(image_id, i, j, 'LRGT'))
                util.save_img(lrgt_img[j], save_img_path)

            psnr = cal_pnsr(sr_img, gt_img)
            psnr_h = []
            for j in range(n):
                psnr_h.append(cal_pnsr(sr_img_h[j], lrgt_img[j]))
            psnr_lr = cal_pnsr(lr_img, gt_img)

            avg_psnr += psnr
            for j in range(n):
                avg_psnr_h[j] += psnr_h[j]
            avg_psnr_lr += psnr_lr

    avg_psnr = avg_psnr / idx
    avg_biterr = sum(biterr) / len(biterr)
    print(get_min_avg_and_indices(biterr))

    avg_psnr_h = [psnr / idx for psnr in avg_psnr_h]
    avg_psnr_lr = avg_psnr_lr / idx
    res_psnr_h = ''
    for p in avg_psnr_h:
        res_psnr_h+=('_{:.4e}'.format(p))
    print('# Validation # PSNR_Cover: {:.4e}, PSNR_Secret: {:s}, PSNR_Stego: {:.4e},  Bit_Error: {:.4e}'.format(avg_psnr, res_psnr_h, avg_psnr_lr, avg_biterr))


if __name__ == '__main__':
    main()