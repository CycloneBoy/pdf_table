#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_slanet
# @Author  ：cycloneboy
# @Date    ：20xx/9/4 15:54


from collections import OrderedDict
from typing import Mapping, Dict

from transformers import PretrainedConfig

__all__ = [
    "SLANetConfig",
]


class SLANetConfig(PretrainedConfig):
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
            model_name: str = "SLANet",
            backbone: str = "SLANet",
            task_type: str = "general",
            table_max_len: int = 488,
            merge_no_span_structure: bool = True,
            model_path: str = "",
            lang: str = "ch",
            vocab_file: str = "",
            debug: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.backbone = backbone
        self.task_type = task_type
        self.table_max_len = table_max_len
        self.merge_no_span_structure = merge_no_span_structure
        self.model_path = model_path
        self.lang = lang
        self.vocab_file = vocab_file
        self.model_provider = "PaddleOCR"
        self.predictor_type = "onnx"
        self.debug = debug

    def get_vocab_file(self):
        lang_prefix = "_ch" if self.lang == "ch" else ""
        if self.vocab_file is None or len(self.vocab_file) == 0:
            return f"{self.model_path}/table_structure_dict{lang_prefix}.txt"
        else:
            return self.vocab_file
