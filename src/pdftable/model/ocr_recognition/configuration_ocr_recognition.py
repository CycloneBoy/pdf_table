#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_ocr_recognition.py
# @Author  ：cycloneboy
# @Date    ：20xx/7/14 11:27


from collections import OrderedDict
from typing import Mapping, Dict

from transformers import PretrainedConfig
from transformers.onnx import OnnxConfig

__all__ = [
    "OCRRecognitionConfig",
    "OCRRecognitionOnnxConfig",
]


class OCRRecognitionConfig(PretrainedConfig):
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
    model_type = "ocr_recognition"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            recognizer: str = "ConvNextViT",
            do_chunking: bool = True,
            img_height: int = 32,
            img_width: int = 804,
            model_path: str = "",
            task_type: str = "document",
            vocab_file: str = "vocab.txt",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.recognizer = recognizer
        self.do_chunking = do_chunking
        self.img_height = img_height
        self.img_width = img_width
        self.model_path = model_path
        self.task_type = task_type
        self.vocab_file = vocab_file

        if self.recognizer not in ["CRNN", "LightweightEdge", "ConvNextViT"]:
            self.recognizer = "ConvNextViT"

        if self.recognizer in ["CRNN", "LightweightEdge"]:
            self.task_type = "general"
        elif self.recognizer == "ConvNextViT" and self.task_type not in ["general", "handwritten",
                                                                         "document", "licenseplate",
                                                                         "scene"]:
            self.task_type = "document"


class OCRRecognitionOnnxConfig(OnnxConfig):

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
                # ("logits", {0: "batch1"}),
                ("logits", {0: "batch", }),
            ]
        )
