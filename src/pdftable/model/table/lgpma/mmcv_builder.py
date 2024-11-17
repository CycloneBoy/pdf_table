#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：mmcv_builder
# @Author  ：cycloneboy
# @Date    ：20xx/9/21 17:42
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmcv_utils import build
from .resnet import ResNet
from .fpn import FPN

from .rpn_head import RPNHead

BACKBONES = {
    "ResNet": ResNet
}

NECKS = {
    "FPN": FPN
}

SHARED_HEADS ={
    # "Shared2FCBBoxHead": Shared2FCBBoxHead,
}

HEADS = {
    "RPNHead": RPNHead,
    # "LGPMARoIHead":LGPMARoIHead,
}


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_shared_head(cfg):
    """Build shared head."""
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)
