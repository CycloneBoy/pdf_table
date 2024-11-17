#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_db_pp
# @Author  ：cycloneboy
# @Date    ：20xx/8/31 10:06

from typing import Dict

from transformers import PretrainedConfig

from pdftable.utils.ocr import OcrCommonUtils

"""
paddle DB config
"""

__all__ = [
    'DbPPConfig',
]


class DbPPConfig(PretrainedConfig):
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
            backbone: str = "PP-OCRv4",
            det_limit_side_len: int = 960,
            det_limit_type: str = 'max',
            det_box_type: str = 'quad',
            thresh: float = 0.3,
            box_thresh: float = 0.6,
            unclip_ratio: float = 1.5,
            use_dilation: bool = False,
            score_mode: str = "fast",
            max_candidates: int = 1000,
            model_path: str = "",
            lang: str = "ch",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type
        self.det_box_type = det_box_type
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.use_dilation = use_dilation
        self.score_mode = score_mode
        self.max_candidates = max_candidates
        self.model_path = model_path

        self.src_lang = lang
        rec_lang, det_lang = OcrCommonUtils.parse_lang_ppocr(lang)
        self.lang = det_lang
