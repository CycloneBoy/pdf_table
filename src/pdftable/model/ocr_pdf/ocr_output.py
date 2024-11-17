#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_output
# @Author  ：cycloneboy
# @Date    ：20xx/7/24 17:34
from dataclasses import dataclass
from typing import List, Union, Dict

import numpy as np
from transformers.utils import ModelOutput

from pdftable.entity.table_entity import OcrCell

"""
模型输出结果
"""

__all__ = [
    "OcrSystemModelOutput"
]


@dataclass
class OcrSystemModelOutput(ModelOutput):
    """
    ocr System 输出结果
    """
    file_name: str = None
    src_id: int = None
    page: int = None
    run_time: str = None
    metric: Dict = None
    image_full: np.ndarray = None
    table_cell_result: List = None
    det_result: Union[List, np.ndarray] = None
    layout_result: List = None
    ocr_result: List = None
    raw_filename: str = None
    save_html_file: str = None
    ocr_cell_content: List[OcrCell] = None
    merge_ocr_cells: List[OcrCell] = None
    pdf_html: List = None
    image_shape: List = None
    image_scalers: List = None
    pdf_scalers: List = None
    is_pdf: bool = None
    table_structure_result: List = None
    image_rotate: bool = None
    all_table_valid_check: bool = None
    image_name: str = None
    use_master:bool = None

    def get_table_structure_bboxs(self):
        bboxs = self.table_structure_result
        if isinstance(bboxs, dict):
            bboxs = bboxs["polygons"]
            if isinstance(bboxs, list):
                bboxs = np.concatenate(bboxs, axis=0)

        return bboxs
