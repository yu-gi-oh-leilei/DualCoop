import torch
from .aslloss import AsymmetricLoss, AsymmetricLossOptimized
from .bceloss import BinaryCrossEntropyLossOptimized, BCELoss
from .dualcoop_loss import AsymmetricLoss_partial


def build_criterion(cfg, model):
    # criterion
    criterion = {}
    if cfg.LOSS.loss_mode == 'asl':
        criterion['asl'] = AsymmetricLossOptimized(
            gamma_neg=cfg.LOSS.ASL.gamma_neg, 
            gamma_pos=cfg.LOSS.ASL.gamma_pos,
            clip=cfg.LOSS.ASL.loss_clip,
            disable_torch_grad_focal_loss=cfg.LOSS.ASL.dtgfl,
            eps=cfg.LOSS.ASL.eps,
        )
    
    elif cfg.LOSS.loss_mode == 'dualcoop':
        criterion['dualcoop'] = AsymmetricLoss_partial(
            gamma_neg=cfg.LOSS.DUALCOOP.gamma_neg, 
            gamma_pos=cfg.LOSS.DUALCOOP.gamma_pos, 
            clip=cfg.LOSS.DUALCOOP.loss_clip, 
            eps=cfg.LOSS.DUALCOOP.eps, 
            disable_torch_grad_focal_loss=cfg.LOSS.DUALCOOP.dtgfl,
            thresh_pos = cfg.LOSS.DUALCOOP.thresh_pos,
            thresh_neg = cfg.LOSS.DUALCOOP.thresh_neg
            )

    elif cfg.LOSS.loss_mode == 'bce':
        criterion['bce'] = BCELoss(reduce=True, size_average=True)

    else:
        raise NotImplementedError("Unknown loss mode %s" % cfg.LOSS.loss_mode)

    device = cal_gpu(model)
    if isinstance(criterion, dict):
        for k, v in criterion.items():
            criterion[k] = v.to(device)

    return criterion

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