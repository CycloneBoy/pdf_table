#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：model_utils
# @Author  ：cycloneboy
# @Date    ：20xx/11/7 14:20
from functools import partial
from typing import List
import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind.long())
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _get_polynomial_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int,
                                                            num_training_steps: int,
                                                            total_epoch:int,
                                                            step_epoch:List[int],
                                                            drop_radio:float=0.1,
                                                            ):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    epoch_size = math.ceil(num_training_steps/total_epoch)
    step1 = step_epoch[0] * epoch_size
    step2 = step_epoch[1] * epoch_size
    if current_step < step1:
        return 1.0
    elif step1 <= current_step < step2:
        return drop_radio
    else:
        return drop_radio*drop_radio

def get_polynomial_constant_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                                 total_epoch,
                                                 step_epoch,
                                                 drop_radio=0.1,
                                                 last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_polynomial_constant_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        total_epoch=total_epoch,
        step_epoch=step_epoch,
        drop_radio=drop_radio
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
