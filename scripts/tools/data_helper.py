import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from lib.utils.tools.logger import Logger as Log


def _get_list_from_env(name):

    value = os.environ.get(name)
    if value is None:
        return None

    return [x.strip() for x in value.split(',')]


class DataHelper:

    def __init__(self, configer, trainer):
        self.configer = configer
        self.trainer = trainer
        self.conditions = configer.conditions

    def input_keys(self):
        env_value = _get_list_from_env('input_keys')
        if env_value is not None:
            inputs = env_value
        elif self.conditions.use_sw_offset:
            inputs = ['img', 'offsetmap_h', 'offsetmap_w']
        elif self.conditions.use_dt_offset:
            inputs = ['img', 'distance_map', 'angle_map']
        elif self.conditions.use_dt_offset_chg:
            inputs = ['oldimg','newimg']
        elif self.conditions.use_dt_chg:
            inputs = ['oldimg', 'newimg']
        else:
            inputs = ['img']

        return inputs

    def target_keys(self):

        env_value = _get_list_from_env('target_keys')
        if env_value is not None:
            return env_value
        elif self.conditions.pred_sw_offset:
            targets = [
                'labelmap',
                'offsetmap_h',
                'offsetmap_w',
            ]
        elif self.conditions.pred_dt_offset:
            targets = [
                'labelmap',
                'distance_map',
                'angle_map',
            ]
        elif self.conditions.pred_dt_offset_chg:
            targets = [
                'labelmap',
                'distance_map',
                'angle_map',
            ]
        elif self.conditions.pred_ml_dt_offset:
            targets = [
                'labelmap',
                'distance_map',
                'multi_label_direction_map',
            ]
        elif self.configer.get("loss","loss_type") == "instance_loss":
            targets = [
                'labelmap',
                'objlabel',
            ]
        elif self.conditions.pred_dt_obj_chg:
            targets = [
                'target'
            ]
        else:
            targets = ['labelmap']

        return targets

    def obj_targets(self):
        env_value = _get_list_from_env('target_keys')
        if env_value is not None:
            return env_value
        else:
            objlabels = ['objlabel']
        return objlabels
    def _reverse_data_dict(self, data_dict):
        result = {}
        for k, x in data_dict.items():

            if not isinstance(x, torch.Tensor):
                result[k] = x
                continue

            new_x = torch.flip(x, [len(x.shape) - 1])

            # since direction_label_map, direction_multilabel_map will not appear in inputs, we omit the flipping
            if k == 'offsetmap_w':
                new_x = -new_x
            elif k == 'angle_map':
                new_x = x.clone()
                mask = (x > 0) & (x < 180)
                new_x[mask] = 180 - x[mask]
                mask = (x < 0) & (x > -180)
                new_x[mask] = - (180 + x[mask])

            result[k] = new_x

        return result

    def _prepare_sequence(self, seq, force_list=False):

        def split_and_cuda(lst: 'List[List[Tensor, len=N]]', device_ids) -> 'List[List[Tensor], len=N]':
            results = []
            for *items, d in zip(*lst, device_ids):
                if len(items) == 1 and not force_list:
                    results.append(items[0].unsqueeze(0).cuda(d))
                else:
                    results.append([
                        item.unsqueeze(0).cuda(d)
                        for item in items
                    ])
            return results

        if self.conditions.diverse_size and not self.trainer.seg_net.training:
            device_ids = list(range(len(self.configer.get('gpu'))))
            return split_and_cuda(seq, device_ids)
        else:
            return self.trainer.module_runner.to_device(*seq, force_list=force_list)

    def prepare_data(self, data_dict, want_reverse=False,Force_ret = False):



        input_keys, target_keys = self.input_keys(), self.target_keys()

        if self.conditions.use_ground_truth:
            input_keys += target_keys

        Log.info_once('Input keys: {}'.format(input_keys))
        Log.info_once('Target keys: {}'.format(target_keys))

        inputs = [data_dict[k] for k in input_keys]
        batch_size = len(inputs[0])
        targets = [data_dict[k] for k in target_keys]
        if Force_ret:
            return (inputs,*targets),batch_size


        if self.configer.get('network','model_name') == 'hrnetyolo':
            b,(c,h,w) = batch_size,inputs[0][0].shape
            input_seg = torch.zeros((batch_size,c,h,w),dtype=torch.float32)
            targets_seg = torch.zeros((batch_size,h,w),dtype=torch.int64)
            for bs in range(batch_size):
                input_seg[bs,:,:,:]  = inputs[0][bs]
                targets_seg[bs,:,:]  = targets[0][bs]
            inputs[0] = input_seg
            targets[0] = targets_seg
            labels_obj = zip(targets[1])
            targets_obj = []
            for i, l in enumerate(labels_obj):
                no = l[0].shape[0]
                if no > 0:
                    l[0][:, 0] = i
                    targets_obj.append(l[0])
            targets[1] = torch.cat( [ target for target in targets_obj ], 0)
            sequences = [
                self._prepare_sequence(inputs, force_list=True),
                self._prepare_sequence(targets, force_list=False),
            ]
        else:
            sequences = [
                self._prepare_sequence(inputs, force_list=True),
                self._prepare_sequence(targets, force_list=False)
            ]
        if want_reverse:
            rev_data_dict = self._reverse_data_dict(data_dict)
            sequences.extend([
                self._prepare_sequence(
                    [rev_data_dict[k] for k in input_keys],
                    force_list=True
                ),
                self._prepare_sequence(
                    [rev_data_dict[k] for k in target_keys],
                    force_list=False
                )
            ])

        return sequences, batch_size
