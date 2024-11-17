#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：ppmodel 
# @File    ：modeling_picodet
# @Author  ：cycloneboy
# @Date    ：20xx/4/18 17:44

import torch
import torch.nn as nn

from .csp_pan import CSPPAN
from .lcnet import LCNet
from .pico_head import PicoHead

__all__ = [
    "PicoDet"
]


class PicoDet(nn.Module):
    """
    Generalized Focal Loss network, see https://arxiv.org/abs/2006.04388

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        head (object): 'PicoHead' instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone_config, neck_config, head_config):
        super(PicoDet, self).__init__()
        self.backbone = LCNet(**backbone_config)
        self.neck = CSPPAN(**neck_config)
        self.head = PicoHead(**head_config)
        self.export_post_process = True
        # self.export_post_process = False
        self.export_nms = True
        self.inputs = {}

    def _forward(self, x=None, pixel_values=None, scale_factor=None):
        if x is None:
            x = self.inputs if pixel_values is None else pixel_values
        body_feats = self.backbone(image=x) if pixel_values is not None else self.backbone(x)
        fpn_feats = self.neck(body_feats)
        head_outs = self.head(fpn_feats, self.export_post_process)
        if self.training or not self.export_post_process:
            return head_outs, None
        else:
            scale_factor = x['scale_factor'] if scale_factor is None else scale_factor
            bboxes, bbox_num = self.head.post_process(
                head_outs, scale_factor, export_nms=self.export_nms)
            return bboxes, bbox_num

    def get_loss(self, x):
        loss = {}

        head_outs, _ = self._forward(x)
        loss_gfl = self.head.get_loss(head_outs, self.inputs)
        loss.update(loss_gfl)
        total_loss = torch.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self, x, pixel_values=None, scale_factor=None):
        if not self.export_post_process:
            return {'picodet': self._forward(x, pixel_values=pixel_values, scale_factor=scale_factor)[0]}
        elif self.export_nms:
            bbox_pred, bbox_num = self._forward(x, pixel_values=pixel_values, scale_factor=scale_factor)
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
            return output
        else:
            bboxes, mlvl_scores = self._forward(x, pixel_values=pixel_values, scale_factor=scale_factor)
            output = {'bbox': bboxes, 'scores': mlvl_scores}
            return output

    def forward(self, x, pixel_values=None, scale_factor=None):
        self.inputs = x
        res = self.get_pred(x, pixel_values=pixel_values, scale_factor=scale_factor)
        return res
