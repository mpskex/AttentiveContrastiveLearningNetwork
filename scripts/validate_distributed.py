import os
import sys
import argparse
import inspect

import torch
import torch.distributed as dist

import fgvc_cl
from fgvc_cl.utils.Register import REGISTRY, print_loaded
from fgvc_cl.utils.ConfigManager import ConfigManager

"""
By Fangrui Liu (fangrui.liu@ubc.ca) Zihao Liu(lzh19961031@pku.edu.cn)
University of British Columbia Okanagan
Peking University
"""

if __name__ == '__main__':
    p = argparse.ArgumentParser("Training Script")
    # FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
    # automatically by torch.distributed.launch.
    p.add_argument("--local_rank", default=0, type=int)
    p.add_argument('--config', default='../config/Default.yaml', type=str, help='Training Configuration')
    p.add_argument('--batch_size', default=None, type=int, help='Training Batch Size')
    p.add_argument('--exp_num', default=None, type=int, help='Training Experience Number')
    param = p.parse_args(sys.argv[1:])


    # FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
    # the 'WORLD_SIZE' environment variable will also be set automatically.
    param.distributed = False
    if 'WORLD_SIZE' in os.environ:
        param.distributed = int(os.environ['WORLD_SIZE']) > 1

    if param.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(param.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                                             init_method='env://')
    else:
        #   This helps a lot
        #   https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
        torch.distributed.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                                             init_method="tcp://127.0.0.1:51234", rank=param.local_rank, world_size=1)

    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.dirname(inspect.getabsfile(fgvc_cl))),
                                            'models', 'pretrained')

    c = ConfigManager()
    c.load(param.config)

    c.batch_size = param.batch_size if param.batch_size is not None else c.batch_size
    c.experience_num = param.exp_num if param.exp_num is not None else c.experience_num

    if param.local_rank == 0:
        print_loaded(REGISTRY)
    trainer = REGISTRY['TRAINER'][c.trainer](c, local_rank=param.local_rank)
    trainer.load(is_best=True)
    #trainer.validate()
    trainer.validate(with_tta=True)
