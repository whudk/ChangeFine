import os

import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import Counter

from utils.tools.logger import Logger as Log
from .base import _BaseEvaluator
from . import tasks

def _parse_output_spec(spec):
    """
    Parse string like "mask, _, dir, ..., seg" into indices mapping
    {
        "mask": 0,
        "dir": 2,
        "seg": -1
    }
    """
    spec = [x.strip() for x in spec.split(',')]
    existing_task_names = set(tasks.task_mapping)

    # `spec` should not have invalid keys other than in `existing_task_names`
    assert set(spec) - ({'...', '_'} | existing_task_names) == set()
    # `spec` should have at least one key in `existing_task_names`
    assert set(spec) & existing_task_names != set()

    counter = Counter(spec)
    for task in tasks.task_mapping.values():
        task.validate_output_spec(spec, counter)
    assert counter['...'] <= 1

    length = len(spec)
    output_indices = {}
    negative_index = False
    for idx, name in enumerate(spec):
        if name not in ['_', '...']:
            index = idx - length if negative_index else idx
            output_indices[name] = index
        elif name == '...':
            negative_index = True

    return output_indices


class StandardEvaluator(_BaseEvaluator):

    def _output_spec(self):
        if self.configer.conditions.pred_dt_offset:
            default_spec = 'mask, dir'
        elif self.configer.conditions.pred_ml_dt_offset:
            default_spec = 'mask, ml_dir'
        else:
            default_spec = '..., seg'

        return os.environ.get('output_spec', default_spec)

    def _init_running_scores(self):
        self.output_indices = _parse_output_spec(self._output_spec())

        self.running_scores = {}
        for task in tasks.task_mapping.values():
            rss, main_key, metric = task.running_score(self.output_indices, self.configer, self.num_classes)
            if rss is None:
                continue
            self.running_scores.update(rss)
            self.save_net_main_key = main_key
            self.save_net_metric = metric
    def update_score_seg(self,outputs,targets):
        #pred = outputs[0].argmax(dim = 1)
        pred = outputs.cpu().numpy()
        target = targets.cpu().numpy()
        self.running_scores['seg'].update(pred,target)
    def update_score_u2net_bin(self,outputs,targets):
        segout = outputs[0]
        pred = torch.sigmoid(segout)
        pred = pred.cpu().numpy()
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred.astype(np.int)
        target = targets.cpu().numpy()
        self.running_scores['seg'].update(pred,target)
    def update_score(self, outputs, metas):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
            
        for i in range(len(outputs[0])):

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']

            outputs_numpy = {}
            for name, idx in self.output_indices.items():
                item = outputs[idx].permute(0, 2, 3, 1)
                if self.configer.get('dataset') == 'celeba':
                    # the celeba image is of size 1024x1024
                    item = cv2.resize(
                        item[i, :border_size[1], :border_size[0]].cpu().numpy(),
                        tuple(x // 2 for x in ori_img_size), interpolation=cv2.INTER_CUBIC
                    )
                else:
                    item = cv2.resize(
                        item[i, :border_size[1], :border_size[0]].cpu().numpy(),
                        tuple(ori_img_size), interpolation=cv2.INTER_CUBIC
                    )
                outputs_numpy[name] = item

            for name in outputs_numpy:
                tasks.task_mapping[name].eval(
                    outputs_numpy, metas[i], self.running_scores
                )



    def evaluation_object(self, preds, targets):
        #以对象来计算Recall Precision

        from skimage import measure
        targets[targets == -1] = 0
        if isinstance(preds, torch.Tensor):
            pred1 = preds.cpu().numpy().astype(np.uint8)
        if isinstance(targets, torch.Tensor):
            gt = targets.cpu().numpy().astype(np.uint8)

        bs  = preds.shape[0]
        chk_n = 0
        ture_n = 0
        pred_n = 0
        gt_n = 0
        for b in range(bs):

            preds_binary = pred1[b]
            targets_binary  = gt[b]

            preds_labeled, m = measure.label(preds_binary,return_num=True)

            targets_labeled, n  = measure.label(targets_binary,return_num=True)
            for i in range(n):
                mask = np.zeros_like(preds_labeled)
                mask[targets_labeled == i] = 1
                if np.sum(preds_labeled * mask) > 0:
                    chk_n = chk_n + 1
            for j in range(m):
                mask = np.zeros_like(targets_labeled)
                mask[preds_labeled == j] = 1
                if np.sum(targets_labeled * mask) > 0:
                    ture_n = ture_n + 1

            pred_n  = pred_n + m
            gt_n = gt_n + n

        return pred_n,gt_n,chk_n,ture_n







