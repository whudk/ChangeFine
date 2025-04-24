#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Lang Huang, Rainbowsecret
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import torchcontrib
from torch.optim import SGD, Adam,AdamW, lr_scheduler

from utils.tools.logger import Logger as Log
import numpy as np

class OptimScheduler(object):
    def __init__(self, configer):
        self.configer = configer

    def init_optimizer(self, net_params,data_loader = None, warm_up_epoch = 0):
        optimizer = None
        if self.configer.get('optim', 'optim_method') == 'sgd':
            optimizer = SGD(net_params,
                            lr=self.configer.get('lr', 'base_lr'),
                            momentum=self.configer.get('optim', 'sgd')['momentum'],
                            weight_decay=self.configer.get('optim', 'sgd')['weight_decay'],
                            nesterov=self.configer.get('optim', 'sgd')['nesterov'])

        elif self.configer.get('optim', 'optim_method') == 'adam':
            optimizer = Adam(net_params,
                             lr=self.configer.get('lr', 'base_lr'),
                             betas=self.configer.get('optim', 'adam')['betas'],
                             eps=self.configer.get('optim', 'adam')['eps'],
                             weight_decay=self.configer.get('optim', 'adam')['weight_decay'])

        elif self.configer.get('optim', 'optim_method') == 'adamw':
            optimizer = AdamW(net_params,
                             lr=self.configer.get('lr', 'base_lr'),
                             betas=self.configer.get('optim', 'adamw')['betas'],
                             eps=self.configer.get('optim', 'adamw')['eps'],
                             weight_decay=self.configer.get('optim', 'adamw')['weight_decay'])

        else:
            Log.error('Optimizer {} is not valid.'.format(self.configer.get('optim', 'optim_method')))
            exit(1)

        policy = self.configer.get('lr', 'lr_policy')

        scheduler = None
        if policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            self.configer.get('lr', 'step')['step_size'],
                                            gamma=self.configer.get('lr', 'step')['gamma'])

        elif policy == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 self.configer.get('lr', 'multistep')['stepvalue'],
                                                 gamma=self.configer.get('lr', 'multistep')['gamma'])

        elif policy == 'lambda_poly':
            if os.environ.get('lambda_poly_power'):
                _lambda_poly_power = float(os.environ.get('lambda_poly_power'))
                Log.info('Use lambda_poly policy with power {}'.format(_lambda_poly_power))
                lambda_poly = lambda iters: pow((1.0 - iters / self.configer.get('solver', 'max_iters')), _lambda_poly_power)
            elif self.configer.exists('lr', 'lambda_poly'):
                Log.info('Use lambda_poly policy with power {}'.format(self.configer.get('lr', 'lambda_poly')['power']))
                lambda_poly = lambda iters: pow((1.0 - iters / self.configer.get('solver', 'max_iters')), self.configer.get('lr', 'lambda_poly')['power'])
            else:
                Log.info('Use lambda_poly policy with default power 0.9')
                lambda_poly = lambda iters: pow((1.0 - iters / self.configer.get('solver', 'max_iters')), 0.99)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

        elif policy == 'lambda_cosine':

            # lambda_cosine = self.cosine_scheduler(self.configer.get("lr","base_lr"), 1e-6,self.configer.train("epoch"),len(self.))
            #
            #
            # if data_loader is not None:
            #     lr_iters = self.cosine_scheduler(
            #         base_value=self.configer.get('lr', 'base_lr'),
            #         final_value=1e-6,
            #         epochs=self.configer.get("train","epoch"),
            #         niter_per_ep=len(data_loader),
            #         warmup_epochs=warm_up_epoch
            #     )
            #     def lambda_cosine(iters):
            #         return lr_iters[iters]
            #
            # else:
            lambda_cosine = lambda iters: (math.cos(math.pi * iters / self.configer.get('solver', 'max_iters'))+ 1.0) / 2
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)

        elif policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode=self.configer.get('lr', 'plateau')['mode'],
                                                       factor=self.configer.get('lr', 'plateau')['factor'],
                                                       patience=self.configer.get('lr', 'plateau')['patience'],
                                                       threshold=self.configer.get('lr', 'plateau')['threshold'],
                                                       threshold_mode=self.configer.get('lr', 'plateau')['thre_mode'],
                                                       cooldown=self.configer.get('lr', 'plateau')['cooldown'],
                                                       min_lr=self.configer.get('lr', 'plateau')['min_lr'],
                                                       eps=self.configer.get('lr', 'plateau')['eps'])

        elif policy == 'swa_lambda_poly':
            optimizer = torchcontrib.optim.SWA(optimizer)
            normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get('solver', 'max_iters') - normal_max_iters) // 5 + 1     # we use 5 ensembles here
            def swa_lambda_poly(iters):
                if iters < normal_max_iters:
                    return pow(1.0 - iters / normal_max_iters, 0.9)
                else:                                   # set lr to half of initial lr and start swa
                    return 0.5 * pow(1.0 - ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters, 0.9)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_poly)

        elif policy == 'swa_lambda_cosine':
            optimizer = torchcontrib.optim.SWA(optimizer)
            normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get('solver', 'max_iters') - normal_max_iters) // 5 + 1     # we use 5 ensembles here
            def swa_lambda_cosine(iters):
                if iters < normal_max_iters:
                    return (math.cos(math.pi * iters / normal_max_iters) + 1.0) / 2
                else:       # set lr to half of initial lr and start swa
                    return 0.5 * (math.cos(math.pi * ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters) + 1.0) / 2
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_cosine)

        else:
            Log.error('Policy:{} is not valid.'.format(policy))
            exit(1)

        return optimizer, scheduler

    def update_optimizer(self, net, optim_method, lr_policy):
        self.configer.update(('optim', 'optim_method'), optim_method)
        self.configer.update(('lr', 'lr_policy'), lr_policy)
        optimizer, scheduler = self.init_optimizer(net)
        return optimizer, scheduler

    def cosine_scheduler_iters(self, base_value, final_value,  warmup_iters=0, start_warmup_value=0):
        warmup_schedule = np.array([])


        if warmup_iters > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        train_iters = self.configer.get("solver", "max_iters") - warmup_iters
        iters = np.arange(train_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == train_iters + warmup_iters
        return schedule

    def cosine_scheduler(self,base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        train_iters =  min(self.configer.get("solver","max_iters") - warmup_iters,epochs * niter_per_ep - warmup_iters)

        iters =  np.arange(train_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == train_iters + warmup_iters
        return schedule