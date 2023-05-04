
import math
import os, sys
import random
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter

from dataset.get_dataset import get_datasets, distributedsampler, without_distributedsampler
from models.builder_baseline import build_baseline
from models.builder_dualclip import build_DualCLIP
from losses.builder_criterion import build_criterion
# from optim.builder_optim import build_optim
from engine import train, validate

from optim.optimizer import build_optimizer
from optim.lr_scheduler import build_lr_scheduler


from utils.util import ModelEma, save_checkpoint, kill_process, load_model, cal_gpu
from utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter

best_mAP = 0
def main_worker(args, cfg, logger):
    global best_mAP

    # build model
    device = torch.device(cfg.TRAIN.device)
    model = build_DualCLIP(cfg)
    model = model.to(device)

    ema_m = ModelEma(model, cfg.TRAIN.ema_decay) # 0.9997^641 = 0.82503551031

    #use_BN
    use_batchnorm = cfg.MODEL.use_BN
    if use_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.DDP.local_rank], broadcast_buffers=False, find_unused_parameters=True) # find_unused_parameters=True  
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))


    
    # criterion
    criterion = build_criterion(cfg, model)

    # args.lr_mult = args.batch_size / 112
    # args.lr_mult = args.batch_size / 128
    cfg.OPTIMIZER.lr_mult = 1.0
    logger.info("lr: {}".format(cfg.OPTIMIZER.lr_mult * cfg.OPTIMIZER.lr))
    

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=cfg.INPUT_OUTPUT.output)
    else:
        summary_writer = None

    # optionally resume from a checkpoint
    if cfg.INPUT_OUTPUT.resume:
        load_model(cfg, logger, model)

    # Data loading and Distributed Sampler
    train_dataset, val_dataset = get_datasets(cfg, logger)
    train_loader, val_loader, train_sampler = distributedsampler(cfg, train_dataset, val_dataset)
    cfg.DATA.len_train_loader = len(train_loader)
    cfg.DATA.classnames = train_loader.dataset.classnames

    if cfg.TRAIN.evaluate:
        _, mAP = validate(val_loader, model, criterion, cfg, logger)

        logger.info(' * mAP {mAP:.5f}'.format(mAP=mAP))
        return
    
    # lr_scheduler and optimizer
    # warmup_scheduler, lr_scheduler, optimizer = build_optim(cfg, model)
    optimizer = build_optimizer(model, cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg)
    warmup_scheduler = None

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        cfg.TRAIN.epochs,
        [eta, epoch_time, losses, mAPs, losses_ema, mAPs_ema],
        prefix='=> Test Epoch: ')


    # global value
    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    
    torch.cuda.empty_cache()
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.epochs):
        train_sampler.set_epoch(epoch)
        if cfg.TRAIN.ema_epoch == epoch:
            ema_m = ModelEma(model.module, cfg.TRAIN.ema_decay)
            torch.cuda.empty_cache()        
        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(train_loader, model, ema_m, criterion, optimizer, lr_scheduler, epoch, cfg, logger, warmup_scheduler)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % cfg.EVAL.val_interval == 0:

            # evaluate on validation set
            loss, mAP = validate(val_loader, model, criterion, cfg, logger)
            if cfg.TRAIN.ema_epoch > epoch:
                loss_ema, mAP_ema = 0, 0
            else:
                loss_ema, mAP_ema = validate(val_loader, ema_m.module, criterion, cfg, logger)
            losses.update(loss)
            mAPs.update(mAP)
            losses_ema.update(loss_ema)
            mAPs_ema.update(mAP_ema)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (cfg.TRAIN.epochs - epoch - 1))

            regular_mAP_list.append(mAP)
            ema_mAP_list.append(mAP_ema)

            progress.display(epoch, logger)


            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('val_loss', loss, epoch)
                summary_writer.add_scalar('val_mAP', mAP, epoch)
                summary_writer.add_scalar('val_loss_ema', loss_ema, epoch)
                summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

            # remember best (regular) mAP and corresponding epochs
            if mAP > best_regular_mAP:
                best_regular_mAP = max(best_regular_mAP, mAP)
                best_regular_epoch = epoch
            if mAP_ema > best_ema_mAP:
                best_ema_mAP = max(mAP_ema, best_ema_mAP)
            
            if mAP_ema > mAP:
                mAP = mAP_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = mAP > best_mAP
            if is_best:
                best_epoch = epoch
            best_mAP = max(mAP, best_mAP)

            logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
            logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))

            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.arch,
                    'state_dict': state_dict,
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(cfg.INPUT_OUTPUT.output, 'checkpoint.pth.tar'))

            if math.isnan(loss) or math.isnan(loss_ema):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.arch,
                    'state_dict': model.state_dict(),
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(cfg.INPUT_OUTPUT.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)


            # early stop
            if cfg.TRAIN.early_stop:
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
                    if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        if dist.get_rank() == 0 and cfg.TRAIN.kill_stop:
                            filename = sys.argv[0].split(' ')[0].strip()
                            killedlist = kill_process(filename, os.getpid())
                            logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist)) 
                        break

    logger.info("Best mAP {}:".format(best_mAP))

    if summary_writer:
        summary_writer.close()
    
    return 0