##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Dengkai
## _kd@whu.edu.cn
## Copyright (c) 2020
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import time
import pdb

import torch
import torch.backends.cudnn as cudnn

from utils.tools.logger import Logger as Log
from utils.tools.configer import Configer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--gpuid', default=[0, 1, 2], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')
    parser.add_argument('--net_prompt',
                        default="clipsam",
                        type=str,
                        dest='network:prompt',
                        choices=['clipsam', 'samclip', 'clip', 'sam'],
                        help='Choose between "clip_sam", "sam_clip", "clip", or "sam" for the prompt type.')

    parser.add_argument('--visualizer', type=str2bool, nargs='?', default=False,
                        dest='eval:visualizer', help='Whether to validate the training set  during resume.')


    # ***********  Params for checkpoint.  **********
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0,1,2], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    # ***********  Params for logging.  **********
    parser.add_argument('--logfile_level', default=None, type=str,
                        dest='logging:logfile_level', help='To set the log level to files.')
    parser.add_argument('--stdout_level', default=None, type=str,
                        dest='logging:stdout_level', help='To set the level to print to screen.')
    parser.add_argument('--log_file', default=None, type=str,
                        dest='logging:log_file', help='The path of log files.')
    parser.add_argument('--rewrite', type=str2bool, nargs='?', default=True,
                        dest='logging:rewrite', help='Whether to rewrite files.')
    parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True,
                        dest='logging:log_to_file', help='Whether to write logging into files.')
    # ***********  Params for env.  **********
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')
    # ***********  Params for distributed training.  **********
    parser.add_argument('--local_rank', type=int, default=0, dest='local_rank', help='local rank of current process')
    parser.add_argument('--use_ground_truth', action='store_true', dest='use_ground_truth', help='Use ground truth for training.')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',help='master ip address for training')
    parser.add_argument('--master_port', type=int, default=12548, help='master port for training')
    parser.add_argument('--nnodes', type=int, default=1, help='num of nodes for training')
    parser.add_argument('--node_rank', type=int, default=1, help='index of node for current process')

    # ***********  Params for trainning.  **********
    parser.add_argument('--train_json', type=str, default="", dest='data:traintxt', help='self supervised trainning txt ')
    parser.add_argument('--val_json', type=str, default="", dest='data:valtxt', help='val path ')
    parser.add_argument('REMAIN', nargs='*')





    args_parser = parser.parse_args()
    Log.info("handle_distributed")
    from utils.distributed import handle_distributed,_setup_process_group
    handle_distributed(args_parser, os.path.expanduser(os.path.abspath(__file__)))
    import platform
    if platform.system() == 'Windows':
        _setup_process_group(args_parser)
    else:
        dist.init_process_group("nccl")
    rank = dist.get_rank()

    device = rank % torch.cuda.device_count()
    seed = args_parser.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    #Log.info('start training...')



    Log.info("handle_distributed success")
    if args_parser.seed is not None:
        random.seed(args_parser.seed)
        torch.manual_seed(args_parser.seed)

    cudnn.enabled = True
    cudnn.benchmark = args_parser.cudnn

    configer = Configer(args_parser=args_parser)

    data_dir = configer.get('data', 'data_dir')
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    abs_data_dir = [os.path.expanduser(x) for x in data_dir]
    configer.update(['data', 'data_dir'], abs_data_dir)

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add(['project_dir'], project_dir)

    if configer.get('logging', 'log_to_file'):
        log_file = configer.get('logging', 'log_file')

        new_log_file = '{}_{}'.format(log_file, time.strftime("%Y-%m-%d_%X", time.localtime()))
        configer.update(['logging', 'log_file'], new_log_file.replace(":",""))
    else:
        configer.update(['logging', 'logfile_level'], None)

    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    model = None
    #if configer.get('method') == 'fcn_segmentor':
    from  scripts.trainer import Trainer
    Log.info("Phase is {}".format(configer.get("phase")))
    if configer.get('phase') == 'train':
        model = Trainer(configer)
        model.train()
    elif configer.get('phase') == 'test':
        model = Trainer(configer)
        model.val()
    else:
        Log.error('Phase: {} is not valid.'.format(configer.get('phase')))
        os._exit(1)
main()
