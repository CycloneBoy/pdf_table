#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：modeling_db_net
# @Author  ：cycloneboy
# @Date    ：20xx/7/13 17:10
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from .configuration_dbnet import DbNetConfig
from .dbnet import DBModel, DBNasModel, VLPTModel
from .ocr_detection_utils import boxes_from_bitmap, polygons_from_bitmap
from ...utils import logger

"""
ocr detection
"""

__all__ = [
    "OCRDetectionDbNet"
]


class OCRDetectionDbNet(nn.Module):

    def __init__(self, config: DbNetConfig, **kwargs):
        """initialize the ocr recognition model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__()

        self.config = config
        self.thresh = config.thresh
        self.return_polygon = config.return_polygon
        self.backbone = config.backbone
        self.detector = None
        if self.backbone == 'resnet50':
            self.detector = VLPTModel()
        elif self.backbone == 'resnet18':
            self.detector = DBModel()
        elif self.backbone == 'proxylessnas':
            self.detector = DBNasModel()
        else:
            raise TypeError(
                f'detector backbone should be either resnet18, resnet50, but got {config.backbone}'
            )
        if config.model_path != '':
            model_path = os.path.join(config.model_path, 'pytorch_model.pt')
            self.detector.load_state_dict(torch.load(model_path, map_location='cpu',weights_only=True), strict=True)
            logger.info(f"加载模型：{model_path}")

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            img (`torch.Tensor`): image tensor,
                shape of each tensor is [3, H, W].

        Return:
            results (`torch.Tensor`): bitmap tensor,
                shape of each tensor is [1, H, W].
            org_shape (`List`): image original shape,
                value is [height, width].
        """
        if isinstance(input, dict):
            pred = self.detector(input['image'])
            result = {
                'results': pred,
                'org_shape': input.get('org_shape', None)
            }
        else:
            result = self.detector(input)
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pred = inputs['results'][0]
        height, width = inputs['org_shape']
        segmentation = pred > self.thresh
        if self.return_polygon:
            boxes, scores = polygons_from_bitmap(pred, segmentation, width,
                                                 height)
        else:
            boxes, scores = boxes_from_bitmap(pred, segmentation, width,
                                              height)
        result = {'det_polygons': np.array(boxes)}
        return result
