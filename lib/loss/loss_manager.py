##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: DonnyYou, RainbowSecret, JingyiXie, JianyuanGuo
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


from utils.tools.logger import Logger as Log
from utils.distributed import is_distributed

from lib.loss.loss_helper import hybrid_loss, Contrast_loss

import torch.nn as nn
import sys


SEG_LOSS_DICT = {
    'hybrid_loss':hybrid_loss,
    "contrast_loss":Contrast_loss
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer
    def _parallel(self, loss):
        if is_distributed():
            Log.info('use distributed loss')
            return loss
            
        if self.configer.get('network', 'loss_balance') and len(self.configer.get('gpu')) > 1:
            Log.info('use DataParallelCriterion loss')
            loss = nn.DataParallel(loss)

        return loss

    def get_seg_loss(self, loss_type=None,model = None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            sys.exit(1)
        Log.info('use loss: {}.'.format(key))

        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

    def get_loss_klass(self, loss_type=None, model=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            sys.exit(1)
        Log.info('use loss: {}.'.format(key))

        loss = SEG_LOSS_DICT[key]()
        return self._parallel(loss)
    def build_loss(self, loss_type=None, model=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            sys.exit(1)
        Log.info('use loss: {}.'.format(key))

        loss = SEG_LOSS_DICT[key](self.configer)

        return self._parallel(loss)