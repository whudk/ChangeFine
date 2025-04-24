##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret, JingyiXie
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import torch
from torch.utils import data


import torch.distributed as dist

from dataset.loader.clipsamDataset import SamClipCD_dataset

#from lib.datasets.loader.gan_loader import gan_dataset


from utils.tools.logger import Logger as Log

import dataset.tools.transforms as trans
from dataset.tools.collate import custom_collate_fn

LOADER_DICT ={
   "sam_clip_cd":SamClipCD_dataset
#    'mim_loader':mim_loader
}

class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer
        # if self.configer.get('phase') == 'test':
        #     return
        from dataset.tools import cv2_aug_transform_chg
        #data transform of changedetection
        if 'cd' in self.configer.get("task"):
            self.aug_train_transform = cv2_aug_transform_chg.CV2AugCompose_CHG(self.configer, split='train')
            self.aug_val_transform = cv2_aug_transform_chg.CV2AugCompose_CHG(self.configer, split='val')
            self.aug_test_transform = cv2_aug_transform_chg.CV2AugCompose_CHG(self.configer, split='test')
            self.img_transform =[]
            assert self.configer.exists('normalize','left') and self.configer.exists('normalize','right'),"check normalize in config.json, normalize of left,right must be in config.json"
            left_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize(div_value=self.configer.get('normalize', 'left')['div_value'],
                                mean=self.configer.get('normalize', 'left')['mean'],
                                std=self.configer.get('normalize', 'left')['std']),
            ])
            self.img_transform.append(left_transform)
            right_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize(div_value=self.configer.get('normalize', 'right')['div_value'],
                                mean=self.configer.get('normalize', 'right')['mean'],
                                std=self.configer.get('normalize', 'right')['std']),
            ])
            self.img_transform.append(right_transform)

        self.label_transform = trans.Compose([
            trans.ToLabel(),
            trans.ReLabel(255, -1), ])


    def build_loader(self, data_path,split = 'train'):
        loader = self.configer.get("train", "loader")
        if loader not in LOADER_DICT.keys():
            raise Exception("loader method of {} is not supported.".format(loader))
        klass = LOADER_DICT[loader]



        kwargs = dict(
            dataset=None,
            aug_transform=(self.aug_train_transform if split == 'train' else self.aug_val_transform),
            img_transform=self.img_transform,
            label_transform=self.label_transform,
            configer=self.configer,
            split = split
        )
        dataset =  klass(data_path, **kwargs)

        #dataset._stastic_ids()
#


        world_size = 1
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            world_size = dist.get_world_size()
        else:
            sampler = None

        data_loader = data.DataLoader(
            dataset,
            batch_size=self.configer.get(split, 'batch_size') //world_size , pin_memory=True,
            num_workers=self.configer.get('data', 'workers') ,
            sampler=sampler,
            shuffle= False if split=='val' else True,
            #collate_fn= custom_collate_fn
            #drop_last=self.configer.get('data', 'drop_last'),
            collate_fn=lambda *args: custom_collate_fn(
            *args, trans_dict = self.configer.get('train', 'data_transformer') if split == 'neg' else self.configer.get(split, 'data_transformer'))
        )
        return data_loader


