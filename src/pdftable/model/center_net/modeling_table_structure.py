#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：modeling_table_structure
# @Author  ：cycloneboy
# @Date    ：20xx/7/17 17:03
import os

import torch
from torch import nn

from .configuration_centernet import TableCenterNetConfig
from .modeling_centernet import DLASeg
from ...utils import logger

__all__ = [
    "TableStructureRec"
]


class TableStructureRec(nn.Module):

    def __init__(self, config: TableCenterNetConfig = None):
        super().__init__()

        self.config = config
        self.recognizer = DLASeg(base_name='dla34',
                                 pretrained=False,
                                 down_ratio=4,
                                 head_conv=256)

        if config.model_path is not None and config.model_path != '':
            raw_model_path = os.path.join(config.model_path, 'pytorch_model.bin')
            if not os.path.exists(raw_model_path):
                raw_model_path = os.path.join(config.model_path, 'pytorch_model.pt')

            checkpoint = torch.load(raw_model_path, map_location='cpu')
            params_pretrained = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            params_pretrained = {str(k).replace("recognizer.",""): v for k, v in params_pretrained.items()}
            self.recognizer.load_state_dict(params_pretrained)
            logger.info(f"加载模型：{raw_model_path}")

    def forward(self, image):
        result = self.recognizer(image)
        result = result[0]
        return result
