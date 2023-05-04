import os, sys
import json
import torch
from typing import List
from copy import deepcopy
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict
import torch.distributed as dist
from utils.logger import setup_logger
##################################################################################

def init_logeger(config):
    os.makedirs(config.INPUT_OUTPUT.output, exist_ok=True)
    logger = setup_logger(output=config.INPUT_OUTPUT.output, distributed_rank=dist.get_rank(), color=True, name="Multi-label Learning Training")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(config.INPUT_OUTPUT.output, "config.yaml")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(config), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # show ddp config
    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(config.DDP.local_rank))
    return logger

# show config
def show_args(config, logger):
    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    if isinstance(config, dict):
        logger.info('\n {}'.format(config))
    else:
        for arg, content in config.__dict__.items():
            logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def _update_bn(self, model):
        for m2, m1 in zip(self.module.named_modules(), model.named_modules()):
            if ('bn' in m2[0]) and ('bn' in m1[0]):
                bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
                bn2['running_mean'].data.copy_(bn1['running_mean'].data)
                bn2['running_var'].data.copy_(bn1['running_var'].data)
                bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def update_bn(self, model):
        with torch.no_grad():
            self._update_bn(model)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
# def voc_from_coco():
    # "module.query_embed.weight": 163840,
    # "module.fc.W": 163840,
    # "module.fc.b": 80
    # [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]


def load_model(cfg, logger, model):
    if cfg.INPUT_OUTPUT.resume:
        if os.path.isfile(cfg.INPUT_OUTPUT.resume):
            logger.info("=> loading checkpoint '{}'".format(cfg.INPUT_OUTPUT.resume))
            checkpoint = torch.load(cfg.INPUT_OUTPUT.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(cfg.INPUT_OUTPUT.resume_omit))

            for omit_name in cfg.INPUT_OUTPUT.resume_omit:
                del state_dict[omit_name]

            # for k, v in state_dict.items():
            #     print(k)
                # if 'pe' in k:
                #     print('================='*5)
                #     print(k)
                #     del state_dict[k]

            model_dict = model.module.state_dict()
            # v是已经保存的权重
            # 
            for k, v in state_dict.items():
                # if k in model_dict and 'backbone' in k and v.shape == model_dict[k].shape:
                if k in model_dict and v.shape == model_dict[k].shape:
                    model_dict[k] = v

                elif cfg.INPUT_OUTPUT.dataname in ('voc2007', 'voc2012') and k in model_dict and v.shape != model_dict[k].shape: # and k in ('query_embed.weight', 'fc.W', 'fc.b'):
                    # print('============================')
                    if 'query_embed.weight' in k or 'fc.W' in k or 'fc.b' in k:
                        voc_from_coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]
                        new_v = torch.zeros_like(model_dict[k])
                        for key, value in enumerate(voc_from_coco):
                            # print(k)
                            if 'query_embed.weight' in k:
                                new_v[key] = v[value]
                            elif 'fc.W' in k:
                                new_v[:, key, :] = v[:, value, :]
                            elif 'fc.b' in k:
                                new_v[:, key] = v[:, value]
                            else:
                                raise ValueError("No query_embed.weight or fc.W or fc.b Found!!!")
                        model_dict[k] = new_v

                else:
                    logger.info('\tMismatched layers: {}'.format(k))


            model.module.load_state_dict(model_dict, strict=False)
            # model.module.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.INPUT_OUTPUT.resume, checkpoint['epoch']))

            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(cfg.INPUT_OUTPUT.resume))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


def kill_process(filename:str, holdpid:int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist


def cal_gpu(module):
    if hasattr(module, 'module') or isinstance(module, torch.nn.DataParallel):
        for submodule in module.module.children():
            if hasattr(submodule, "_parameters"):
                parameters = submodule._parameters
                if "weight" in parameters:
                    return parameters["weight"].device
    else:
        for submodule in module.children():
            if hasattr(submodule, "_parameters"):
                parameters = submodule._parameters
                if "weight" in parameters:
                    return parameters["weight"].device




    