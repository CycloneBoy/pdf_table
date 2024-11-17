#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : configuration_linecell.py
# @Author   : cycloneboy
# @Date     : 20xx/12/16 - 16:11

from typing import Mapping, Dict

from transformers import PretrainedConfig

"""
LineCell configuration
"""

__all__ = [
    "LineCellConfig",
]


class LineCellConfig(PretrainedConfig):
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
    model_type = "linecell"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            model_name: str = "LineCell",
            backbone: str = "opencv",
            task_type: str = "PubTabNet",
            process_background: bool = False,
            line_scale: int = 40,
            line_scale_vertical: int = 50,
            line_tol: int = 2,
            line_mark_tol: int = 6,
            threshold_block_size: int = 15,
            threshold_constant: int = -2,
            iterations: int = 0,
            diff_angle: int = 400,

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
        self.process_background = process_background
        self.line_scale = line_scale
        self.line_scale_vertical = line_scale_vertical
        self.line_tol = line_tol
        self.line_mark_tol = line_mark_tol
        self.threshold_block_size = threshold_block_size
        self.threshold_constant = threshold_constant
        self.iterations = iterations
        self.diff_angle = diff_angle

        self.output_dir = output_dir
        self.page = page
        self.debug = debug
        self.model_path = model_path
        self.model_provider = "Other"
        self.predictor_type = "other"
