'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import logging
import os
import os.path as osp
import pickle
import random

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

import data.util as util

try:
    import mc
except ImportError:
    pass
logger = logging.getLogger('base')

class CoCoDataset(data.Dataset):
    def __init__(self, opt):
        super(CoCoDataset, self).__init__()
        self.opt = opt
        # get train indexes
        self.data_path = self.opt['data_path']
        self.txt_path = self.opt['txt_path']
        with open(self.txt_path) as f:
            self.list_image = f.readlines()
        self.list_image = [line.strip('\n') for line in self.list_image]
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))
        self.data_type = self.opt['data_type']
        random.shuffle(self.list_image)
        self.LR_input = True
        self.num_image = self.opt['num_image']

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        image_name = self.list_image[index]
        path_frame = os.path.join(self.data_path, image_name)
        img_GT = util.read_img(None, osp.join(path_frame, path_frame))
        index_h = random.randint(0, len(self.list_image) - 1)

        # random crop
        H, W, C = img_GT.shape
        rnd_h = random.randint(0, max(0, H - GT_size))
        rnd_w = random.randint(0, max(0, W - GT_size))
        img_frames = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_frames = img_frames[:, :, [2, 1, 0]]
        img_frames = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames, (2, 0, 1)))).float().unsqueeze(0)

        # process h_list
        if index_h % 100 == 0:
            path_frame_h = "../dataset/locwatermark/blue.png"
        else:
            image_name_h = self.list_image[index_h]
            path_frame_h = os.path.join(self.data_path, image_name_h)
        
        frame_h = util.read_img(None, osp.join(path_frame_h, path_frame_h))
        H1, W1, C1 = frame_h.shape
        rnd_h = random.randint(0, max(0, H1 - GT_size))
        rnd_w = random.randint(0, max(0, W1 - GT_size))
        img_frames_h = frame_h[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
        img_frames_h = img_frames_h[:, :, [2, 1, 0]]
        img_frames_h = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames_h, (2, 0, 1)))).float().unsqueeze(0)

        img_frames_h = torch.nn.functional.interpolate(img_frames_h, size=(512, 512), mode='nearest', align_corners=None).unsqueeze(0)
        img_frames = torch.nn.functional.interpolate(img_frames, size=(512, 512), mode='nearest', align_corners=None)

        return {'GT': img_frames, 'LQ': img_frames_h}

    def __len__(self):
        return len(self.list_image)