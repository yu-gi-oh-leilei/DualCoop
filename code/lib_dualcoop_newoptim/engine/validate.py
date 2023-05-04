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
from utils.metric import voc_mAP, asl_mAP
from models.builder_dualclip import do_forward_and_criterion

@torch.no_grad()
def validate(val_loader, model, criterion, cfg, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    model.eval()
    criterion[cfg.LOSS.loss_mode].eval()
    saved_data = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            lable = target.clone()

            # compute output
            if cfg.TRAIN.amp:
                with torch.cuda.amp.autocast(enabled=cfg.TRAIN.amp):
                    output, loss = do_forward_and_criterion(cfg, images, target, model, criterion)
            else:
                output, loss = do_forward_and_criterion(cfg, images, target, model, criterion)

            if torch.isnan(loss):
                saveflag = True
            
            output_sm = output

            # record loss
            losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            # save some data            
            lable[lable < 0] = 0
            _item = torch.cat((output_sm.detach().cpu().data, lable.detach().cpu().data), 1)
            saved_data.append(_item)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.INPUT_OUTPUT.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )

        # import ipdb; ipdb.set_trace()
        # calculate mAP
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(cfg.INPUT_OUTPUT.output, saved_name), saved_data)
        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            logger.info("Calculating mAP:")
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = voc_mAP
            mAP, aps = metric_func([os.path.join(cfg.INPUT_OUTPUT.output, _filename) for _filename in filenamelist], cfg.DATA.num_class, return_each=True, logger=logger)
            logger.info("  mAP: {}".format(mAP))
            if cfg.INPUT_OUTPUT.out_aps:
                logger.info("  aps: {}".format(np.array2string(aps, precision=5)))
        else:
            mAP = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return loss_avg, mAP

def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()

# from utils.misc import concat_all_gather
            # if dist.get_rank() == 0:
            #     output_gather = concat_all_gather(output_sm)
            #     lable__gather     = concat_all_gather(lable)
            #     _y_pred.append(output_gather.detach().cpu().data)
            #     _y_true.append(lable__gather.detach().cpu().data)
            # if dist.get_world_size() > 1:
            #     dist.barrier()
            
            # metric_func = asl_mAP  
            # mAP, aps = metric_func(torch.cat(_y_true, dim=0).numpy(), preds = torch.cat(_y_pred, dim=0).numpy())
            # logger.info("  mAP: {}".format(mAP))
    
        # _y_pred = []
        # _y_true = []