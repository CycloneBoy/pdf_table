#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_centernet
# @Author  ：cycloneboy
# @Date    ：20xx/7/17 17:28

from collections import OrderedDict
from typing import Mapping, Dict

from transformers import PretrainedConfig
from transformers.onnx import OnnxConfig

__all__ = [
    "TableCenterNetConfig",
    "TableCenterNetOnnxConfig",
]


class TableCenterNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieModel`] or a [`TFErnieModel`]. It is used to
    instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ERNIE

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the ERNIE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ErnieModel`] or [`TFErnieModel`].


    Examples:

    """
    model_type = "table_structure"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            model_name: str = "CenterNet",
            backbone: str = "dla34",
            task_type: str = "wtw",
            down_ratio: int = 4,
            head_conv: int = 256,
            img_width: int = 1024,
            model_path: str = "",
            debug: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.backbone = backbone
        self.task_type = task_type
        self.down_ratio = down_ratio
        self.head_conv = head_conv
        self.img_width = img_width
        self.model_path = model_path
        self.model_provider = "model_scope"
        self.predictor_type = "pytorch"
        self.debug = debug


class TableCenterNetOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def default_onnx_opset(self) -> int:
        return 14

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # ("logits", {0: "batch", 2: "height", 3: "width"}),
                ("hm", {0: "batch", }),
                ("v2c", {0: "batch", }),
                ("c2v", {0: "batch", }),
                ("reg", {0: "batch", }),
            ]
        )
