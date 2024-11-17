#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：configuration_ocr_document
# @Author  ：cycloneboy
# @Date    ：20xx/7/14 15:09


from typing import Dict

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

__all__ = [
    "OCRDocumentConfig",
    "OCRToHtmlConfig",
]


class OCRDocumentConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OCRModel`] or a [`TFOCRModel`]. It is used to
    instantiate a OCR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OCR

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        detector (`str`, *optional*, defaults 'resnet18' ):
            db_net - ['resnet50', 'resnet18', 'proxylessnas]
        thresh (`float`, *optional*, defaults to 0.2):
            db_net thresh
        return_polygon (`bool`, *optional*, defaults to False):
            return_polygon
        model_path_detector (`str`, *optional*, defaults to ''):
            model_path_detector
        recognizer (`str`, *optional*, defaults 'ConvNextViT' ):
            recognizer model type - ['ConvNextViT', 'CRNN', 'LightweightEdge]
        do_chunking (`bool`, *optional*, defaults to False):
            do_chunking : when recognizer = 'ConvNextViT' is True else False
        img_height (`int`, *optional*, defaults to 32):
            img_height
        img_width (`int`, *optional*, defaults to 32):
            img_width : when recognizer = 'ConvNextViT' is 804 else 640
        task_type (`str`, *optional*, defaults 'general' ):
            run task type  - ['general', 'handwritten', 'document', 'licenseplate', 'scene']
        model_path_recognizer (`str`, *optional*, defaults to ''):
            model_path_detector
        model_path (`str`, *optional*, defaults to ''):
            model_path
    Examples:

    """
    model_type = "ocr_document"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            detect_model: str = "db",
            detector: str = "resnet18",
            thresh: float = 0.2,
            return_polygon: bool = False,
            model_path_detector: str = "",

            recognizer: str = "ConvNextViT",
            do_chunking: bool = True,
            img_height: int = 32,
            img_width: int = 804,
            task_type: str = "document",
            model_path_recognizer: str = "",

            model_path: str = "",
            lang: str = "ch",
            debug: bool = False,

            table_structure_model: str = "Lore",
            model_path_table_structure: str = "",
            table_structure_task_type: str = "wtw",
            table_structure_merge: bool = False,
            update_version: str = "v1.1",

            layout_model: str = "picodet",
            layout_model_task_type: str = "ch",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.detect_model = detect_model
        self.detector = detector
        self.thresh = thresh
        self.return_polygon = return_polygon
        self.model_path_detector = model_path_detector

        self.recognizer = recognizer
        self.do_chunking = do_chunking
        self.img_height = img_height
        self.img_width = img_width
        self.task_type = task_type
        self.model_path_recognizer = model_path_recognizer
        self.model_path = model_path
        self.lang = lang
        self.debug = debug

        self.table_structure_model = table_structure_model
        self.model_path_table_structure = model_path_table_structure
        self.table_structure_task_type = table_structure_task_type
        # lore and linecell merge
        self.table_structure_merge = table_structure_merge
        self.update_version = update_version

        self.layout_model = layout_model
        self.layout_model_task_type = layout_model_task_type

        self.fix_model_names()

    def fix_model_names(self):
        if self.detector.lower() in ["ppocrv4", "pp-ocrv4"]:
            self.detector = "PP-OCRv4"
        elif self.detector.lower() in ["ppocrv3", "pp-ocrv3"]:
            self.detector = "PP-OCRv3"
        elif self.detector.lower() in ["resnet18", "resnet"]:
            self.detector = "resnet18"

        # detector
        if self.detector in ["PP-OCRv4", "PP-OCRv3", "PP-Table"] and self.detect_model != "db_pp":
            self.detect_model = "db_pp"
        elif self.detector not in ["PP-OCRv4", "PP-OCRv3", "PP-Table"] and self.detect_model == "db_pp":
            self.detector = "PP-OCRv4"
        elif self.detector in ["resnet18", "proxylessnas"] and self.detect_model != "db":
            self.detect_model = "db"
        elif self.detector not in ["resnet18", "proxylessnas"] and self.detect_model == "db":
            self.detect_model = "resnet18"

        # recognizer
        if self.recognizer in ["ConvNextViT", "CRNN", "LightweightEdge",
                               "PP-OCRv4", "PP-OCRv3", "PP-Table"]:
            self.recognizer = "PP-OCRv4"

        if self.table_structure_model == "LoreAndLineCell":
            self.table_structure_model = "Lore"
            self.table_structure_merge = True

        if self.lang == 'ch' and self.layout_model_task_type == "ch":
            self.layout_model_task_type = "ch"
        elif self.layout_model == "picodet" and self.layout_model_task_type not in ["ch", "en", "table"]:
            self.layout_model_task_type = "en"

    def get_tsr_model_name(self):
        add_merge = ""
        if self.table_structure_merge:
            add_merge = "AndLineCell"
        table_structure_model = f"{self.table_structure_model}{add_merge}"
        run_name = [self.lang, table_structure_model, self.table_structure_task_type]
        return "_".join(run_name)

    def tsr_match_need_use_master(self):
        need_use_master = False
        if self.table_structure_model in ["Lgpma", "MtlTabNet", "TableMaster"]:
            need_use_master = True
        return need_use_master


class OCRToHtmlConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OCRModel`] or a [`TFOCRModel`]. It is used to
    instantiate a OCR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OCR

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        detector (`str`, *optional*, defaults 'resnet18' ):
            db_net - ['resnet50', 'resnet18', 'proxylessnas]

    Examples:

    """
    model_type = "ocr_html"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            detector: str = "resnet18",
            thresh: float = 0.2,
            return_polygon: bool = False,
            model_path_detector: str = "",

            recognizer: str = "ConvNextViT",
            do_chunking: bool = True,
            img_height: int = 32,
            img_width: int = 804,
            task_type: str = "general",
            model_path_recognizer: str = "",

            model_path: str = "",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.detector = detector
        self.thresh = thresh
        self.return_polygon = return_polygon
        self.model_path_detector = model_path_detector

        self.recognizer = recognizer
        self.do_chunking = do_chunking
        self.img_height = img_height
        self.img_width = img_width
        self.task_type = task_type
        self.model_path_recognizer = model_path_recognizer
        self.model_path = model_path
