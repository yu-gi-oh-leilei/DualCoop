import torch
import time
import torch
import time
import os
import math
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import voc_mAP
from engine.validate import validate
from models.builder_dualclip import do_forward_and_criterion


def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, cfg, logger, warmup_scheduler=None):
    if cfg.TRAIN.amp:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.amp)
    
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, cfg.TRAIN.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return [param_group][-1]['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.eval()
    model.module.prompt_learner.train()
    criterion[cfg.LOSS.loss_mode].train()


    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        if cfg.TRAIN.amp:
            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.amp):
                output, loss = do_forward_and_criterion(cfg, images, target, model, criterion)
        else:
            output, loss = do_forward_and_criterion(cfg, images, target, model, criterion)

        # record loss and memory
        losses.update(loss.item(), images.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)


        if cfg.TRAIN.amp:
            # amp backward function
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if cfg.OPTIMIZER.max_clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.max_clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            loss.backward()
            if cfg.OPTIMIZER.max_clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.max_clip_grad_norm)
            optimizer.step()

        # record learning_rate
        lr.update(get_learning_rate(optimizer))
        if epoch >= cfg.TRAIN.ema_epoch:
            ema_m.update(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        speed_gpu.update(images.size(0) / batch_time.val, batch_time.val)
        speed_all.update(images.size(0) * dist.get_world_size() / batch_time.val, batch_time.val)

        if i % cfg.INPUT_OUTPUT.print_freq == 0:
            progress.display(i, logger)

    # adjust learning rate
    if cfg.OPTIMIZER.lr_scheduler != 'OneCycleLR':
        scheduler.step()

    return losses.avg
