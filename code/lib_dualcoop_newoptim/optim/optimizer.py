"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import warnings
import torch
import torch.nn as nn

from .radam import RAdam

AVAI_OPTIMS = ["Adam", "amsgrad", "SGD", "RMSprop", "RAdam", "AdamW"]


def build_optimizer(model, cfg):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """
    optim = cfg.OPTIMIZER.optim
    lr = cfg.OPTIMIZER.lr
    weight_decay = cfg.OPTIMIZER.weight_decay
    momentum = cfg.OPTIMIZER.momentum
    sgd_dampening = cfg.OPTIMIZER.sgd_dampening
    sgd_nesterov = cfg.OPTIMIZER.sgd_nesterov
    rmsprop_alpha = cfg.OPTIMIZER.rmsprop_alpha
    adam_beta1 = cfg.OPTIMIZER.adam_beta1
    adam_beta2 = cfg.OPTIMIZER.adam_beta2
    staged_lr = cfg.OPTIMIZER.staged_lr
    new_layers = cfg.OPTIMIZER.new_layers
    base_lr_mult = cfg.OPTIMIZER.base_lr_mult

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            "Unsupported optim: {}. Must be one of {}".format(
                optim, AVAI_OPTIMS
            )
        )

    if staged_lr:
        if not isinstance(model, nn.Module):
            raise TypeError(
                "When staged_lr is True, model given to "
                "build_optimizer() must be an instance of nn.Module"
            )

        if hasattr(model, 'module'):
            model = model.module

        if isinstance(new_layers, str):
            if new_layers is None:
                warnings.warn(
                    "new_layers is empty, therefore, staged_lr is useless"
                )
            new_layers = [new_layers]

        base_params = []
        base_layers = []
        new_params = []

        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)

        param_groups = [
            {
                "params": base_params,
                "lr": lr * base_lr_mult
            },
            {
                "params": new_params
            },
        ]

    else:
        if isinstance(model, nn.Module):
            # param_groups = model.parameters()
            param_groups = filter(lambda p: p.requires_grad, model.parameters())
        else:
            param_groups = model

    if optim == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "SGD":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "RMSprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "RAdam":
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    return optimizer
