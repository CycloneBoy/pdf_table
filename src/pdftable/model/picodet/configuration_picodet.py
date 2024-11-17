#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：ppmodel 
# @File    ：configuration_picodet
# @Author  ：cycloneboy
# @Date    ：20xx/5/13 11:16

from collections import OrderedDict
from typing import Mapping, Dict

from transformers import PretrainedConfig
from transformers.onnx import OnnxConfig

"""
picodet 配置文件
"""

__all__ = [
    "PicodetConfig",
    "PicodetOnnxConfig"
]


class PicodetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieModel`] or a [`TFErnieModel`]. It is used to
    instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ERNIE

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        task_type (`str`, *optional*, defaults 'ch' ):
            db_net - ['ch', 'en', 'table]

    Examples:

    """
    model_type = "picodet"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            backbone: str = "LCNet",
            task_type: str = "ch",
            img_height: int = 800,
            img_width: int = 608,
            norm_mean: list = [0.485, 0.456, 0.406],
            norm_std: list = [0.229, 0.224, 0.225],
            order: str = "hwc",
            scale: float = 1. / 255.,
            score_threshold: float = 0.5,
            nms_threshold: float = 0.5,
            label_file: str = None,

            strides: list = [8, 16, 32, 64],
            nms_top_k: int = 1000,
            keep_top_k: int = 100,
            model_path: str = "",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.task_type = task_type
        self.img_height = img_height
        self.img_width = img_width
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.order = order
        self.scale = scale

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.label_file = label_file if label_file is not None else self.get_label_dir(lang=self.task_type)
        self.strides = strides
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.model_path = model_path

        self.label_config = {
            # CDLA
            "ch": {
                "text": 0,
                "title": 1,
                "figure": 2,
                "figure_caption": 3,
                "table": 4,
                "table_caption": 5,
                "header": 6,
                "footer": 7,
                "reference": 8,
                "equation": 9,
            },
            # publaynet
            "en": {
                "text": 0,
                "title": 1,
                "list": 2,
                "table": 3,
                "figure": 4,
            },
            "table": {
                "table": 0
            }
        }

        self.label2id = self.label_config.get(self.task_type, self.label_config["ch"])
        self.id2label = {idx: cate for cate, idx in self.label2id.items()}


    def get_label_dir(self, lang="ch"):
        if lang == "zh":
            lang = "ch"
        label_file_dict = {
            "ch": "layout_cdla_dict.txt",
            "en": "layout_publaynet_dict.txt",
            "table": "layout_table_dict.txt",
        }

        label_file = label_file_dict.get(lang.lower(), label_file_dict["ch"])
        return label_file


class PicodetOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                # ("pixel_values", {0: "batch"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # ("bbox", {0: "batch"}),
                # ("bbox_num", {0: "batch"}),
                ("logits", {0: "batch"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    @property
    def default_onnx_opset(self) -> int:
        return 14
