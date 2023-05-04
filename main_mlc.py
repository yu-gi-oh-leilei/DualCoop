import math
import os, sys
import random
import time
import json

import _init_paths
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from config import parser_args
from config_opt import get_config

from utils.misc import init_distributed_and_seed
from utils.util import show_args, init_logeger
from main_worker import main_worker


def get_args():
    args = parser_args()
    return args

def main():
    args = get_args()
    args, config = get_config(args)
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(config.DDP.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.DDP.gpus

    # init distributed and seed
    init_distributed_and_seed(config)
    
    # init logeger and show config
    logger = init_logeger(config)
    show_args(config, logger)

    return main_worker(args, config, logger)

if __name__ == '__main__':
    main()


# def prepare_package():
#     import torch
#     import torch.nn as nn
#     import torch.nn.parallel
#     import torch.backends.cudnn as cudnn
#     import torch.distributed as dist
#     import torch.optim
#     import torch.multiprocessing as mp

    # for k,v in model.named_parameters():
    #     if "backbone" in k:
    #         v.requires_grad = False

            # import ipdb; ipdb.set_trace()
            # for k, v in state_dict.items():
            #     print(k)
# count = 1
# for k,v in model.named_parameters():
#     print('layer_{}    name:{}'.format(count, k))
#     count = count+1