#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import gather as torch_gather
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import numpy as np

from utils.tools.logger import Logger as Log
from utils.distributed import get_rank, is_distributed
from  lib.extensions.parallel.data_parallel import DataParallelModel


INITIAL_LOG_LOSS_SCALE = 20.0


class ModuleRunner(object):

    def __init__(self, configer):
        self.configer = configer
        self._init()

    def _init(self):
        self.configer.add(['iters'], 0)
        self.configer.add(['last_iters'], 0)
        self.configer.add(['epoch'], 0)
        self.configer.add(['last_epoch'], 0)
        self.configer.add(['max_performance'], 0.0)
        self.configer.add(['performance'], 0.0)
        self.configer.add(['max_accuracy'], 0.0)
        self.configer.add(['accuracy'], 0.0)
        self.configer.add(['min_val_loss'], 9999.0)
        self.configer.add(['val_loss'], 9999.0)
        self.half = False
        if self.configer.exists('network','train_half'):
            self.half = self.configer.exists('network','train_half')
        if not self.configer.exists('network', 'bn_type'):
            self.configer.add(['network', 'bn_type'], 'torchbn')

        if self.configer.get('phase') == 'train':
            assert len(self.configer.get('gpu')) > 1 or self.configer.get('network', 'bn_type') == 'torchbn'

        Log.info('BN Type is {}.'.format(self.configer.get('network', 'bn_type')))

    def to_device(self, *params, force_list=False):
        if is_distributed():
            device = torch.device('cuda:{}'.format(get_rank()))
        else:
            device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        return_list = list()
        for param in params:
            if isinstance(param,list):
                batch_size = len(param)
                if self.configer.exists("train","batch_per_gpu"):
                    size_per_gpu = self.configer.get("train","batch_per_gpu")
                    assert(size_per_gpu>0)
                    merge_data = params[0]

                    for b in range(1,batch_size):
                        merge_data = torch.cat((merge_data,params[b]),dim = 0)
                    return_list.append(merge_data.to(device))
                else:
                    if self.configer.get('network','model_name') == 'hrnetyolo' or \
                        self.configer.get('network','model_name') == 'hrnetcd':
                        for bs in range(batch_size):
                            param[bs] = param[bs].to(device)
                        return_list.append(param)
                    else:
                        return_list.append(param[0].to(device))
            else:
                return_list.append(param.to(device))

        if force_list:
            return return_list
        else:
            return return_list[0] if len(params) == 1 else return_list

    def _make_parallel(self, net):
       # device_ids = self.configer.get('gpu')
       #  if is_distributed() :
       #      rank = get_rank()
       #      #Log.info("local_rank = {}".format(local_rank))
       #      # return torch.nn.parallel.DistributedDataParallel(
       #      #     net,
       #      #     device_ids=[local_rank],
       #      #     output_device=local_rank,
       #      #     find_unused_parameters=False,
       #      #     broadcast_buffers=False
       #      # )
       #      device = rank % torch.cuda.device_count()
       #      return  torch.nn.parallel.DistributedDataParallel(net.to(device), device_ids=[rank], find_unused_parameters=True)
       #
       #  device_ids = self.configer.get('gpu')
       #  if len(self.configer.get('gpu')) == 1:
       #      self.configer.update(['network', 'gathered'], True)
       #      device_ids = [device_ids]
       #
       #
       #
       #
       #  # if len(device_ids) > 1:
       #  #     return nn.DataParallel(net, device_ids=device_ids)
       #  #return nn.DataParallel(net, device_ids=device_ids)
       #  return DataParallelModel(net, device_ids = [device_ids], gather_=self.configer.get('network', 'gathered'))
        import torch.distributed as dist


        #mprint('start training...')

        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = self.configer.get("seed") * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")




    def restart_from_checkpoint(self, ckp_path, run_variables=None, **kwargs):
        """
        Re-start from checkpoint
        """
        if not os.path.isfile(ckp_path):
            return
        print("Found checkpoint at {}".format(ckp_path))

        # open checkpoint file
        checkpoint = torch.load(ckp_path, map_location="cpu")

        # key is what to look for in the checkpoint file
        # value is the object to load
        # example: {'state_dict': model}
        for key, value in kwargs.items():
            if key in checkpoint and value is not None:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=False)
                    print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
                except TypeError:
                    try:
                        msg = value.load_state_dict(checkpoint[key])
                        print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                    except ValueError:
                        print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
            else:
                print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

        # re load variable important for the run
        if run_variables is not None:
            for var_name in run_variables:
                if var_name in checkpoint:
                    run_variables[var_name] = checkpoint[var_name]


    def load_pretrained_weights(self,model, pretrained_weights, checkpoint_key):
        net_dict = model.state_dict()

        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
            return

    def load_matched_state_dict(self, model, pretrained_weights, checkpoint_key):
        net_dict = model.state_dict()
        net_dict = {k.replace("backbone.", ""): v for k, v in net_dict.items()}
        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        return

    def load_partial_state_dict(self, model, loaded_state_dict):
        model_state = model.state_dict()
        matched_state = {}

        for k, v in loaded_state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    matched_state[k] = v
                else:
                    print(f"Shape mismatch for {k}: {v.shape} vs {model_state[k].shape}, skipped.")
            else:
                print(f"Key {k} not in current model, skipped.")

        model_state.update(matched_state)
        model.load_state_dict(model_state)
    def load_net(self, net,net_path=None):
        #self.to_device(net)
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP



        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        net = DDP(net.to(device), device_ids=[rank], find_unused_parameters=True)

        if not is_distributed():
            net = net.to(torch.device('cpu' if self.configer.get('gpu') is None else 'cuda'))

        net.float()
        net_dict = net.state_dict()

        net_path = self.configer.get("network","resume")
        if net_path is not None:
            if os.path.isfile(net_path):
                Log.info('Loading checkpoint from {}...'.format(net_path))
                #resume_dict = torch.load(net_path,map_location=torch.device('cpu'))

                resume_dict = torch.load(net_path, map_location=torch.device('cpu'))
                state_dict = resume_dict['state_dict']


                if not list(state_dict.keys())[0].startswith('module.'):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict['module.' + k] = v
                    state_dict = new_state_dict


                self.load_partial_state_dict(net,state_dict)
                # 加载模型权重
                #net.load_state_dict(state_dict, strict=False)

                if 'config_dict' in resume_dict:
                    self.configer.update(['performance'], resume_dict['config_dict'].get('max_performance'))
                    if 'max_accuracy' in resume_dict['config_dict'].keys():
                        self.configer.update(['accuracy'], resume_dict['config_dict'].get('max_accuracy'))
                    #if self.configer.get('network', 'resume_continue'):
                    #self.configer.resume(resume_dict['config_dict'])
        return net

    @staticmethod
    def load_state_dict(module, state_dict, strict=False):
        """Load state_dict to a module.
        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.
        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        """
        unexpected_keys = []
        own_state = module.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                Log.warn('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(),
                                           param.size()))
                
        missing_keys = set(own_state.keys()) - set(state_dict.keys())

        err_msg = []
        if unexpected_keys:
            err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
        if missing_keys:
            # we comment this to fine-tune the models with some missing keys.
            err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
        err_msg = '\n'.join(err_msg)
        # if err_msg:
        #     if strict:
        #         raise RuntimeError(err_msg)
        #     else:
        #         Log.warn(err_msg)

    def save_state_dict(self,state,save_mode='iters',ckpt_name = None):
        if is_distributed() and get_rank() != 0:
            return
        if self.configer.get('checkpoints', 'checkpoints_root') is None:
            checkpoints_dir = os.path.join(self.configer.get('project_dir'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))
        else:
            checkpoints_dir = os.path.join(self.configer.get('checkpoints', 'checkpoints_root'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        if ckpt_name is None:
            latest_name = '{}_latest.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
        else:
            latest_name = '{}_latest.pth'.format(ckpt_name)
        torch.save(state, os.path.join(checkpoints_dir, latest_name))

        if save_mode == 'iters':
            if self.configer.get('iters') - self.configer.get('last_iters') >= \
                    self.configer.get('checkpoints', 'save_iters'):
                if ckpt_name is None:
                    latest_name = '{}_iters{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                          self.configer.get('iters'))
                else:
                    latest_name = '{}_iters{}.pth'.format(ckpt_name,
                                                          self.configer.get('iters'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['last_iters'], self.configer.get('iters'))

        elif save_mode == 'epoch':
            if self.configer.get('epoch') - self.configer.get('last_epoch') >= \
                    self.configer.get('checkpoints', 'save_epoch'):
                if ckpt_name is None:
                    latest_name = '{}_epoch{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                          self.configer.get('epoch'))
                else:
                    latest_name = '{}_epoch{}.pth'.format(ckpt_name,
                                                          self.configer.get('epoch'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))


        else:
            Log.error('Metric: {} is invalid.'.format(save_mode))
            exit(1)
    def save_net(self, net,  save_mode='iters', ckpt_name = None, excluded_param_names = None):
        if is_distributed() and get_rank() != 0:
            return


        if excluded_param_names is not None:
        # 创建一个新的state_dict
            state_dict = {name: param for name, param in net.state_dict().items() if name not in excluded_param_names}
        else:
            state_dict = {name: param for name, param in net.state_dict().items()}




        state = {
            'config_dict': self.configer.to_dict(),
            'state_dict': state_dict,
        }
        if self.configer.get('checkpoints', 'checkpoints_root') is None:
            checkpoints_dir = os.path.join(self.configer.get('project_dir'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))
        else:
            checkpoints_dir = os.path.join(self.configer.get('checkpoints', 'checkpoints_root'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if ckpt_name is None:
            latest_name = '{}_latest.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
        else:
            latest_name = '{}_latest.pth'.format(ckpt_name)
        torch.save(state, os.path.join(checkpoints_dir, latest_name))
        if save_mode == 'performance':
            if self.configer.get('performance') >= self.configer.get('max_performance'):
                if ckpt_name is None:
                    latest_name = '{}_max_performance.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                else:
                    latest_name = '{}_max_performance.pth'.format(ckpt_name)
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['max_performance'], self.configer.get('performance'))

        elif save_mode == 'val_loss':
            if self.configer.get('val_loss') < self.configer.get('min_val_loss'):
                if ckpt_name is None:
                    latest_name = '{}_min_loss.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                else:
                    latest_name = '{}_min_loss.pth'.format(ckpt_name)
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['min_val_loss'], self.configer.get('val_loss'))

        elif save_mode == 'iters':
            if self.configer.get('iters') - self.configer.get('last_iters') >= \
                    self.configer.get('checkpoints', 'save_iters'):
                if ckpt_name is None:
                    latest_name = '{}_iters{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('iters'))
                else:
                    latest_name = '{}_iters{}.pth'.format(ckpt_name,
                                                          self.configer.get('iters'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['last_iters'], self.configer.get('iters'))

        elif save_mode == 'epoch':
            if self.configer.get('epoch') - self.configer.get('last_epoch') >= \
                    self.configer.get('checkpoints', 'save_epoch'):
                if ckpt_name is None:
                    latest_name = '{}_epoch{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('epoch'))
                else:
                    latest_name = '{}_epoch{}.pth'.format(ckpt_name,
                                                          self.configer.get('epoch'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))


        else:
            Log.error('Metric: {} is invalid.'.format(save_mode))
            exit(1)

    def freeze_bn(self, net, syncbn=False):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

            if syncbn:
                from lib.extensions import BatchNorm2d, BatchNorm1d
                if isinstance(m, BatchNorm2d) or isinstance(m, BatchNorm1d):
                    m.eval()

    def cancel_gradients_last_layer(self,epoch, model, freeze_last_layer):
        if epoch >= freeze_last_layer:
            return
        for n, p in model.named_parameters():
            if "last_layer" in n:
                p.grad = None

    def clip_grad(self, model, max_grad=10.):
        """Computes a gradient clipping coefficient based on gradient norm."""
        # total_norm = 0
        # for p in model.parameters():
        #     if p.requires_grad:
        #         modulenorm = p.grad.data.norm()
        #         total_norm += modulenorm ** 2
        #
        # total_norm = math.sqrt(total_norm)
        #
        # norm = max_grad / max(total_norm, max_grad)
        # for p in model.parameters():
        #     if p.requires_grad:
        #         p.grad.mul_(norm)
        norms = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                norms.append(param_norm.item())
                clip_coef = max_grad / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
        return norms
    def gather(self, outputs, target_device=None, dim=0):
        r"""
        Gathers tensors from different GPUs on a specified device
          (-1 means the CPU).
        """
        if not self.configer.get('network', 'gathered'):
            if target_device is None:
                target_device = list(range(torch.cuda.device_count()))[0]

            return torch_gather(outputs, target_device, dim=dim)

        else:
            return outputs

    def get_lr_from_scheduler(self,optimizer,scheduler):
        it = self.configer.get("iters")
        lr = scheduler[it]
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
    def get_lr(self, optimizer,scheduler,backbone_list=(0, )):
        it = self.configer.get("iters")
        base_lr_list = scheduler.get_last_lr()
        for backbone_index in backbone_list:
            optimizer.param_groups[backbone_index]['lr'] = base_lr_list[backbone_index]

        return [group['lr'] for group in optimizer.param_groups]
    def adjust_learning_rate(self,optim, cur_iter, max_iters,lr_pow = 0.9):
        # for g in optim.param_groups:
        #     lr = g["lr"]
        lr = self.configer.get("lr","base_lr")
        #cur_iter = min(max_iters,cur_iter * self.configer.get('train','batch_size'))
        scale_running_lr = ((1. - float(cur_iter) / max_iters) ** lr_pow)#.type(torch.float64)
        #scale_running_lr = torch.clamp(scale_running_lr, min=1e-8, max=1.0)

        adjust_lr = max(1e-8,lr  * scale_running_lr)
        for g in optim.param_groups:
            g['lr'] = adjust_lr
    def warm_lr(self, iters, scheduler, optimizer, backbone_list=(0, )):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if not self.configer.exists('lr', 'is_warm') or not self.configer.get('lr', 'is_warm'):
            return

        warm_iters = self.configer.get('lr', 'warm')['warm_iters']
        if iters < warm_iters:
            if self.configer.get('lr', 'warm')['freeze_backbone']:
                for backbone_index in backbone_list:
                    optimizer.param_groups[backbone_index]['lr'] = 0.0

            else:
                lr_ratio = (self.configer.get('iters') + 1) / warm_iters
                base_lr_list = scheduler.get_last_lr()
                for backbone_index in backbone_list:
                    optimizer.param_groups[backbone_index]['lr'] = base_lr_list[backbone_index] * (lr_ratio ** 4)







def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
            master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
        model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
                master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                    param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return torch.zeros_like(param)



class MixedPrecisionTrainer:
    def __init__(
            self,
            *,
            model,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            if hasattr(self.model,'convert_to_fp16'):
                self.model.convert_to_fp16()
            pass

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: torch.Tensor):
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: torch.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: torch.optim.Optimizer):

        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            Log.info(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False



        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: torch.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()

        opt.step()
        return True
    def get_lg_loss_scale(self):
        return self.lg_loss_scale
    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with torch.no_grad():
                param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)
def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
