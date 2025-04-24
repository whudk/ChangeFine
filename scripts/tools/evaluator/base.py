import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils.tools.logger import Logger as Log
from lib.metrics import running_score as rslib
from lib.metrics import F1_running_score as fscore_rslib


class _BaseEvaluator:

    def __init__(self, configer, trainer, num_classes = None):
        self.configer = configer
        self.trainer = trainer
        self.num_classes = num_classes if num_classes is not None else self.configer.get("data", "num_classes")
        self._init_running_scores()
        self.conditions = configer.conditions


    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
    def use_me(self):
        raise NotImplementedError

    def _init_running_scores(self):
        raise NotImplementedError

    def update_score(self, *args, **kwargs):
        raise NotImplementedError

    def print_scores(self, show_miou=True):
        for key, rs in self.running_scores.items():
            Log.info('Result for {}'.format(key))
            if isinstance(rs, fscore_rslib.F1RunningScore):
                FScore, FScore_cls = rs.get_scores()
                Log.info('Mean FScore: {}'.format(FScore))
                Log.info(
                    'Class-wise FScore: {}'.format(
                        ', '.join(
                            '{:.3f}'.format(x)
                            for x in FScore_cls
                        )
                    )
                )
            elif isinstance(rs, rslib.SimpleCounterRunningScore):
                Log.info('ACC: {}\n'.format(rs.get_mean_acc()))
            else:
                if show_miou and hasattr(rs, 'get_mean_iou'):
                    Log.info('Mean IOU: {}\n'.format(rs.get_mean_iou()))
                Log.info('Pixel ACC: {}\n'.format(rs.get_pixel_acc()))

                if hasattr(rs, 'n_classes'):
                    Log.info('CLS  IOU: {}\n'.format(rs.get_cls_iou()))
                    if rs.n_classes == 2:
                        Log.info(
                            'F1 Score: {} Precision: {} Recall: {}\n'
                            .format(*rs.get_F1_score())
                        )

    def prepare_validaton(self):
        """
        Replicate models if using diverse size validation.
        """
        device_ids = list(range(len(self.configer.get('gpu'))))
        if self.conditions.diverse_size:
            cudnn.benchmark = False
            assert self.configer.get('val', 'batch_size') <= len(device_ids)
            replicas = nn.parallel.replicate(
                self.trainer.seg_net.module, device_ids)
            return replicas

    def update_acc(self):

        try:
            rs = self.running_scores[self.save_net_main_key]
            acc = rs.get_pixel_acc()

            max_acc = self.configer.get('max_accuracy')
            self.configer.update(['accuracy'], acc)
            if acc > max_acc:
                Log.info('acc {} -> {}'.format(max_acc, acc))
                self.configer.update(['max_accuracy'], acc)
        except Exception as e:
            Log.warn(e)
    def update_performance(self):
        import torch.distributed as dist
        try:
            rs = self.running_scores[self.save_net_main_key]
            if self.save_net_metric == 'miou':
                perf = rs.get_mean_iou()
            elif self.save_net_metric == 'acc':
                perf = rs.get_pixel_acc()
            # if dist.is_initialized():
            #     def reduce_tensor(inp):
            #         """
            #         Reduce the loss from all processes so that
            #         process with rank 0 has the averaged results.
            #         """
            #         world_size = dist.get_world_size()
            #         if world_size < 2:
            #             return inp
            #         with torch.no_grad():
            #             reduced_inp = inp
            #             # dist.reduce(reduced_inp, dst=0)
            #             dist.all_reduce(reduced_inp)
            #         return reduced_inp
            #     perf = reduce_tensor(perf) / dist.get_world_size()
            #     tensor_perf[0] = perf
            #     tensor_perf = self.reduce_tensor(tensor_perf)
            #     # print(tensor_perf)
            #     if dist.get_rank() == 0:
            #         perf = tensor_perf / dist.get_world_size()
            #         perf = perf.cpu().item()
            #         print(perf)
            max_perf = self.configer.get('max_performance')
            self.configer.update(['performance'], perf)
            if perf > max_perf:
                Log.info('Performance {} -> {}'.format(max_perf, perf))
                self.configer.update(['max_performance'], perf)
        except Exception as e:
            Log.warn(e)
    def update_performance_instance(self,perf):

        try:
            max_perf = self.configer.get('max_performance')
            self.configer.update(['performance'], perf)
            if perf > max_perf:
                Log.info('Performance {} -> {}'.format(max_perf, perf))
                self.configer.update(['max_performance'], perf)
        except Exception as e:
            Log.warn(e)
    def reset(self):
        for rs in self.running_scores.values():
            rs.reset()



