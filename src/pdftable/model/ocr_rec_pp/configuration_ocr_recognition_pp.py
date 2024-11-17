#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_ocr_recognition_pp
# @Author  ：cycloneboy
# @Date    ：20xx/8/31 16:58


from typing import Dict

from transformers import PretrainedConfig

__all__ = [
    "PPOcrRecognitionConfig",
]

from pdftable.utils.ocr import OcrCommonUtils


class PPOcrRecognitionConfig(PretrainedConfig):
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
            backbone: str = "PP-OCRv4",
            rec_image_shape: str = "3, 48, 320",
            limited_max_width: int = 1280,
            limited_min_width: int = 16,
            rec_batch_num: int = 6,
            model_path: str = "",
            lang: str = "ch",
            vocab_file: str = "",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.rec_image_shape = [int(v.strip()) for v in rec_image_shape.split(",")]
        self.limited_max_width = limited_max_width
        self.limited_min_width = limited_min_width
        self.rec_batch_num = rec_batch_num
        self.model_path = model_path
        self.src_lang = lang
        rec_lang, det_lang = OcrCommonUtils.parse_lang_ppocr(lang)
        self.lang = rec_lang
        self.vocab_file = vocab_file

        if self.backbone.lower() in ["ppocrv4","pp-ocrv4"]:
            self.backbone = "PP-OCRv4"
        elif self.backbone.lower() in ["ppocrv3","pp-ocrv3"]:
            self.backbone = "PP-OCRv3"
        else:
            self.backbone = "PP-OCRv4"

    def get_vocab_file(self):
        if self.vocab_file is None or len(self.vocab_file) == 0:
            return f"{self.model_path}/{self.lang}_dict.txt"
        else:
            return self.vocab_file
