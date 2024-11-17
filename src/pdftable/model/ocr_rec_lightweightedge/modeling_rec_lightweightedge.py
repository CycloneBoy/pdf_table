#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：modeling_rec_lightweightedge
# @Author  ：cycloneboy
# @Date    ：20xx/7/14 11:21


import torch
import torch.nn as nn

from .nas_block import plnas_linear_mix_se

__all__ = [
    'OcrRecLightweightEdge'
]


class OcrRecLightweightEdge(nn.Module):
    """
        基于混合rep block的nas模型
        Args:
            input (tensor): batch of input images
    """

    def __init__(self):
        super().__init__()
        self.our_nas_model = plnas_linear_mix_se(1, 128)
        self.embed_dim = 128
        self.head = nn.Linear(self.embed_dim, 7644)

    def forward(self, input):
        # RGB2GRAY
        input = input[:, 0:1, :, :] * 0.2989 \
                + input[:, 1:2, :, :] * 0.5870 \
                + input[:, 2:3, :, :] * 0.1140
        x = self.our_nas_model(input)
        x = torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)
        b, s, e = x.size()
        x = x.reshape(b * s, e)
        prediction = self.head(x).view(b, s, -1)
        return prediction
