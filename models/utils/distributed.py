import torch
import torch.nn as nn
import subprocess
import sys
import os
import platform

print(platform.system())
# def is_distributed():
#     if torch.__version__ < '1.0':
#         return torch.distributed.is_initialized()
#     else:
#         return  torch.distributed.is_initialized()
# def get_world_size():
#     if (platform.system() == "Windows"):
#         #if not torch.cuda.is_available():
#         current_env = os.environ.copy()
#         return  len(current_env['CUDA_VISIBLE_DEVICES'].split(','))
#     else:
#         if not torch.distributed.is_initialized():
#             return 1
#     return 1
#
# def get_rank():
#     if (platform.system() == "Windows"):
#         if not torch.cuda.is_available():
#             return 0
#     else:
#         if not torch.distributed.is_initialized():
#             return 0
#     return 0
def is_distributed():
    assert torch.__version__ > '1.7.0',"torch version is {}, please update to latest version than 1.7.0".format(torch.__version__)
    if torch.__version__ <= '1.7.0':
        return torch.distributed.is_initialized()

    return  torch.distributed.is_initialized()
def get_world_size():

    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()
def handle_distributed(args, main_file):
    if len(args.gpu) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
        return

    # if args.dist_method == 'DP':
    #     print("use DP method for trainning")
    #     return
    if args.local_rank >= 0:
        _setup_process_group(args)
        return

    current_env = os.environ.copy()
    if current_env.get('CUDA_VISIBLE_DEVICES') is None:
        current_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
        world_size = len(args.gpu)
    else:
        world_size = len(current_env['CUDA_VISIBLE_DEVICES'].split(','))

    current_env['WORLD_SIZE'] = str(world_size)

    print('World size:', world_size)
    # Logic for spawner
    python_exec = sys.executable
    command_args = sys.argv
    print(python_exec)
    main_index = 0#command_args.index('main.py')
    command_args = command_args[main_index+1:]
    print(command_args)

    if args.nnodes  == 1:
        command_args = [
            python_exec, '-u',
            '-m', 'torch.distributed.launch',
            '--nproc_per_node', str(world_size),
            main_file,
        ] + command_args
    else:
        command_args = [
                           python_exec, '-u',
                           '-m', 'torch.distributed.launch',
                           '--nnodes', str(args.nnodes),
                           '--node_rank', str(args.node_rank),
                           '--master_addr', str(args.master_address),
                           '--master_port', str(args.master_port),
                           '--nproc_per_node', str(world_size),
                           main_file,
                       ] + command_args
    # command_args = [
    #     'torchrun',
    #     '--nnodes={}'.format(str(1)),
    #     '--nproc_per_node={}'.format(str(world_size)),
    #     main_file,
    # ] + command_args
    print(command_args)
    process = subprocess.Popen(command_args, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=command_args)    
    sys.exit(process.returncode)

def _setup_process_group(args):
    from datetime import datetime
    local_rank = args.local_rank
    print('local_rank:{}'.format(local_rank))
    current_dir = os.path.join(os.getcwd(),"ddp")
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")



    if (platform.system() == "Windows"):
        backends = 'gloo'
        init_method = f"file:///{os.path.join(current_dir, 'ddp_{}'.format(current_time))}"
    else:
        backends = 'nccl'
        init_method = f"file://{os.path.join(current_dir, 'ddp_{}'.format(current_time))}"
    torch.cuda.set_device(local_rank)

    print("use backends:{}".format(backends))

    torch.distributed.init_process_group(
        backend = backends,
        init_method=init_method,
        rank=local_rank,
        world_size =  len(args.gpu)
    )
    torch.distributed.barrier()
    print("load distributed success".format(backends))