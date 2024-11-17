#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：model_docx_layout
# @Author  ：cycloneboy
# @Date    ：20xx/9/27 14:22

import collections.abc
import os

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_docxlayout import DocXLayoutConfig
from .model_dla import get_pose_net
from ..table.lgpma.checkpoint import load_checkpoint

"""
版面分析
"""
__all__ = [
    "DocXLayoutModel"
]


class DocXLayoutPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DocXLayoutConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DocXLayoutModel(DocXLayoutPreTrainedModel):
    def __init__(self, config: DocXLayoutConfig, ):
        super().__init__(config)
        self.config = config

        self.model = get_pose_net(num_layers=config.num_layers,
                                  heads=config.heads,
                                  head_conv=config.head_conv,
                                  down_ratio=config.down_ratio,
                                  convert_onnx=config.convert_onnx)

        if self.config.model_path != "" and os.path.exists(self.config.model_path):
            self.load_model()

    def load_model(self, ):
        load_checkpoint(self.model, self.config.model_path, map_location="cpu")

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        outputs = self.model(pixel_values)
        sequence_output = outputs[-1]
        if not return_dict:
            head_outputs = sequence_output
            return head_outputs

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
        )


class DocXLayoutForObjectDetection(DocXLayoutPreTrainedModel):
    def __init__(self, config: DocXLayoutConfig):
        super().__init__(config)

        self.backbone = DocXLayoutModel(config, )
