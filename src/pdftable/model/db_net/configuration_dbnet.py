#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_dbnet
# @Author  ：cycloneboy
# @Date    ：20xx/7/13 17:56
from collections import OrderedDict
from typing import Mapping, Dict

from transformers import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

__all__ = [
    'DbNetConfig',
    'DbNetOnnxConfig',
]


class DbNetConfig(PretrainedConfig):
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
    model_type = "dbnet"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            backbone: str = "resnet18",
            img_width: int = 736,
            thresh: float = 0.2,
            return_polygon: bool = False,
            model_path: str = "",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.img_width = img_width
        self.thresh = thresh
        self.return_polygon = return_polygon
        self.model_path = model_path


class DbNetOnnxConfig(OnnxConfig):

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
                ("logits", {0: "batch", }),
            ]
        )
