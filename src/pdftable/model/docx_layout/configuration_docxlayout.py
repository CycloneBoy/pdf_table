#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_docx.py
# @Author  ：cycloneboy
# @Date    ：20xx/9/27 14:50
from collections import OrderedDict
from typing import Mapping, Dict

from packaging import version
from transformers import PretrainedConfig
from transformers.onnx import OnnxConfig

__all__ = [
    "DocXLayoutConfig",
    "DocXLayoutOnnxConfig",
]


class DocXLayoutConfig(PretrainedConfig):
    model_type = "docxlayout"

    def __init__(
            self,
            model_name: str = "DocXLayout",
            backbone: str = "dla34",
            task_type: str = "general",
            num_layers: int = 34,
            heads: Dict = None,
            head_conv: int = 256,
            down_ratio: int = 4,
            convert_onnx: bool = False,
            top_k: int = 100,
            scores_thresh: float = 0.3,
            num_classes: int = 13,
            use_nms: bool = True,
            model_path: str = "",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.backbone = backbone
        self.task_type = task_type
        self.num_layers = num_layers
        self.heads = heads if heads is not None else {'cls': 4, 'ftype': 3, 'hm': 11, 'hm_sub': 2, 'reg': 2,
                                                      'reg_sub': 2, 'wh': 8, 'wh_sub': 8}
        self.head_conv = head_conv
        self.down_ratio = down_ratio
        self.convert_onnx = convert_onnx
        self.model_path = model_path

        self.top_k = top_k
        self.scores_thresh = scores_thresh
        self.num_classes = num_classes
        self.use_nms = use_nms

        self.label2id = {
            "title": 0,
            "figure": 1,
            "text": 2,
            "header": 3,
            "page_number": 4,
            "footnote": 5,
            "footer": 6,
            "table": 7,
            "table_caption": 8,
            "figure_caption": 9,
            "equation": 10,
            "full_column": 11,
            "sub_column": 12
        }
        self.id2label = {idx: cate for cate, idx in self.label2id.items()}


class DocXLayoutOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    @property
    def default_onnx_opset(self) -> int:
        return 12
