#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : configuration_cls_pulc.py
# @Author   : cycloneboy
# @Date     : 20xx/10/15 - 13:29
from collections import OrderedDict
from typing import Mapping, Dict, List

from transformers import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

__all__ = [
    'ClsPulcConfig',
]

CLS_PULC_TASK_CONFIG = {
    "table_attribute": {
        "use_ssld": True,
        "class_num": 6,
    },
    "text_image_orientation": {
        "use_ssld": True,
        "class_num": 4,
    },
    "textline_orientation": {
        "use_ssld": True,
        "class_num": 2,
        "stride_list": [2, [2, 1], [2, 1], [2, 1], [2, 1]]
    },
    "language_classification": {
        "use_ssld": True,
        "class_num": 10,
        "stride_list": [2, [2, 1], [2, 1], [2, 1], [2, 1]]
    }
}


class ClsPulcConfig(PretrainedConfig):
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
    model_type = "lcnet"

    def __init__(
            self,
            backbone: str = "PPLCNet",
            task_type: str = "text_image_orientation",
            use_ssld: bool = True,
            class_num: int = 4,
            stride_list: List = None,
            model_path: str = "",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.task_type = task_type
        self.use_ssld = use_ssld
        self.class_num = class_num
        self.stride_list = stride_list
        self.model_path = model_path

        if self.task_type is not None and self.task_type in CLS_PULC_TASK_CONFIG:
            config = CLS_PULC_TASK_CONFIG[self.task_type]
            self.use_ssld = config["use_ssld"]
            self.class_num = config["class_num"]
            self.stride_list = config.get("stride_list", None)

    def get_model_params(self,):
        return CLS_PULC_TASK_CONFIG[self.task_type]