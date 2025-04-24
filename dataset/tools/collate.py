#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch, trans_dict):
    oldimg_batch, newimg_batch, label_batch, img_records_batch = zip(*batch)

    oldimg_batch, newimg_batch, label_batch = _pad_same_size(list(oldimg_batch), list(newimg_batch), list(label_batch), trans_dict)
    # Collate the tensor data using the default collate function
    oldimg_batch = default_collate(oldimg_batch)
    newimg_batch = default_collate(newimg_batch)
    label_batch = default_collate(label_batch)

    # img_records_batch is already a tuple of dictionaries, one per item in batch
    # If you need it in a different structure, adjust here

    return oldimg_batch, newimg_batch, label_batch, img_records_batch

def _pad_same_size(oldimg_batch, newimg_batch, label_batch, trans_dict):
    #data_keys = batch[0].keys()
    if trans_dict['size_mode'] == 'diverse_size':
        target_widths = [oldimg_batch.size(2) for i in range(len(oldimg_batch))]
        target_heights = [oldimg_batch.size(1) for i in range(len(oldimg_batch))]

    elif trans_dict['size_mode'] == 'fix_size':
        target_width, target_height = trans_dict['input_size']
        target_widths, target_heights = [target_width] * len(oldimg_batch), [target_height] * len(oldimg_batch)

    elif trans_dict['size_mode'] == 'multi_size':
        ms_input_size = trans_dict['ms_input_size']
        target_width, target_height = ms_input_size[random.randint(0, len(ms_input_size) - 1)]
        target_widths, target_heights = [target_width] * len(oldimg_batch), [target_height] * len(oldimg_batch)

    elif trans_dict['size_mode'] == 'max_size':
        border_width = [sample.size(2) for sample in oldimg_batch]
        border_height = [sample.size(1) for sample in oldimg_batch]
        target_width, target_height = max(border_width), max(border_height)
        target_widths, target_heights = [target_width] * len(oldimg_batch), [target_height] * len(oldimg_batch)

    else:
        raise NotImplementedError('Size Mode {} is invalid!'.format(trans_dict['size_mode']))

    if 'fit_stride' in trans_dict:
        stride = trans_dict['fit_stride']
        for i in range(len(oldimg_batch)):
            target_width, target_height = target_widths[i], target_heights[i]
            pad_w = 0 if (target_width % stride == 0) else stride - (target_width % stride)  # right
            pad_h = 0 if (target_height % stride == 0) else stride - (target_height % stride)  # down
            target_widths[i] = target_width + pad_w
            target_heights[i] = target_height + pad_h

    for i in range(len(oldimg_batch)):
        target_width, target_height = target_widths[i], target_heights[i]



        if len(oldimg_batch[i].size())>3:
            _,channels, height, width = oldimg_batch[i].size()
        else:
            channels, height, width = oldimg_batch[i].size()
        if height == target_height and width == target_width:
            continue

        scaled_size = [width, height]

        if trans_dict['align_method'] in ['only_scale', 'scale_and_pad']:
            w_scale_ratio = target_width / width
            h_scale_ratio = target_height / height
            if trans_dict['align_method'] == 'scale_and_pad':
                w_scale_ratio = min(w_scale_ratio, h_scale_ratio)
                h_scale_ratio = w_scale_ratio

            scaled_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))


            scaled_size_hw = (scaled_size[1], scaled_size[0])
            oldimg_batch[i] = F.interpolate(oldimg_batch[i].unsqueeze(0),
                                        scaled_size_hw, mode='bilinear', align_corners=True).squeeze(0)
            newimg_batch[i] = F.interpolate(newimg_batch[i].unsqueeze(0),
                                            scaled_size_hw, mode='bilinear', align_corners=True).squeeze(0)
            label_batch[i] = F.interpolate(label_batch[i].unsqueeze(0).unsqueeze(0).float(),
                                            scaled_size_hw, mode='bilinear', align_corners=True).long().squeeze(0).squeeze(0)


        pad_width = target_width - scaled_size[0]
        pad_height = target_height - scaled_size[1]
        assert pad_height >= 0 and pad_width >= 0
        if pad_width > 0 or pad_height > 0:
            assert trans_dict['align_method'] in ['only_pad', 'scale_and_pad']
            left_pad = 0
            up_pad = 0
            if 'pad_mode' not in trans_dict or trans_dict['pad_mode'] == 'random':
                left_pad = random.randint(0, pad_width)  # pad_left
                up_pad = random.randint(0, pad_height)  # pad_up

            elif trans_dict['pad_mode'] == 'pad_left_up':
                left_pad = pad_width
                up_pad = pad_height

            elif trans_dict['pad_mode'] == 'pad_right_down':
                left_pad = 0
                up_pad = 0

            elif trans_dict['pad_mode'] == 'pad_center':
                left_pad = pad_width // 2
                up_pad = pad_height // 2

            elif trans_dict['pad_mode'] == 'pad_border':
                if random.randint(0, 1) == 0:
                    left_pad = pad_width
                    up_pad = pad_height
                else:
                    left_pad = 0
                    up_pad = 0
            else:
                raise ValueError ('Invalid pad mode: {}'.format(trans_dict['pad_mode']))


            pad = (left_pad, pad_width-left_pad, up_pad, pad_height-up_pad)

            oldimg_batch[i] = F.pad(oldimg_batch[i], pad=pad, value=0)
            newimg_batch[i] = F.pad(newimg_batch[i], pad=pad, value=0)
            label_batch[i] = F.pad(label_batch[i], pad=pad, value=0)


    return oldimg_batch,newimg_batch, label_batch