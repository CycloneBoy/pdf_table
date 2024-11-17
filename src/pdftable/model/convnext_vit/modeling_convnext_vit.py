#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：modeling_convnext_vit.py
# @Author  ：cycloneboy
# @Date    ：20xx/4/4 14:50

import torch
from torch import nn
from transformers import ViTModel, ViTForImageClassification, ViTConfig, ConvNextConfig

from .modeling_convnext import ConvNextModel
from .modeling_vit import ViTForSTR

__all__ = [
    "ConvNextViT"
]


class ConvNextViT(nn.Module):

    def __init__(self):
        super(ConvNextViT, self).__init__()
        config_connext = ConvNextConfig(num_channels=1,
                                        depths=[3, 3, 8, 3],
                                        hidden_sizes=[96, 192, 256, 512])
        self.cnn_model = ConvNextModel(config_connext)
        config_vit = ViTConfig(patch_size=1,
                               num_channels=512,
                               hidden_size=192,
                               num_attention_heads=3,
                               intermediate_size=int(4 * 192),
                               image_size=[1, 75],
                               num_labels=7644)
        self.vitstr = ViTForSTR(config_vit)

    def forward(self, input):
        """ Transformation stage """
        # RGB2GRAY
        input = input[:, 0:1, :, :] * 0.2989 + input[:, 1:2, :, :] * 0.5870 + input[:, 2:3, :, :] * 0.1140
        output_cnn = self.cnn_model(input)
        features = output_cnn.last_hidden_state
        # print(f"features: {features.shape}")
        output = self.vitstr(features, interpolate_pos_encoding=False)
        return output
