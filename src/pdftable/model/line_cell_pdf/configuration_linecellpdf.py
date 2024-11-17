#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_linecellpdf
# @Author  ：cycloneboy
# @Date    ：2024/1/2 15:40


from typing import Mapping, Dict

from transformers import PretrainedConfig

"""
LineCellPdf configuration
"""

__all__ = [
    "LineCellPdfConfig",
]


class LineCellPdfConfig(PretrainedConfig):
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
    model_type = "linecellpdf"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            model_name: str = "LineCellPdf",
            backbone: str = "digital_pdf",
            task_type: str = "PubTabNet",
            line_tol: int = 2,

            output_dir: str = None,
            page: int = None,
            debug: bool = True,
            model_path: str = "",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.backbone = backbone
        self.task_type = task_type
        self.line_tol = line_tol

        self.output_dir = output_dir
        self.page = page
        self.debug = debug
        self.model_path = model_path
        self.model_provider = "Other"
        self.predictor_type = "other"
