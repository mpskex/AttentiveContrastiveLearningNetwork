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
    p.add_argument('--config', default='../config/Default.yaml', type=str, help='Training Configuration')
    param = p.parse_args(sys.argv[1:])

    #   This helps a lot
    #   https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                            init_method="tcp://127.0.0.1:51234", rank=0, world_size=1)

    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.dirname(inspect.getabsfile(fgvc_cl))),
                                            'models', 'pretrained')

    c = ConfigManager()
    c.load(param.config)
    print_loaded(REGISTRY)
    print(str(c))
    trainer = REGISTRY['TRAINER'][c.trainer](c, local_rank=0)
    trainer.train()