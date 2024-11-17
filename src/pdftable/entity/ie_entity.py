#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable
# @File    ：ie_entity
# @Author  ：cycloneboy
# @Date    ：20xx/6/19 14:17
from typing import Any

from pdfminer.layout import LTTextLineHorizontal
from pydantic import BaseModel

from pdftable.entity.table_entity import OcrCell

__all__ = [
    'PdfTextCell',
    'PdfTextCellV2',
    'PdfCheckResult',
    'PdfPageInfo',
    "PdfTaskParams",
]


class PdfTextCell(object):

    def __init__(self, text_cell, y_index=None):
        self.text_cell: LTTextLineHorizontal = text_cell
        self.y_index = y_index

    def __str__(self):
        return self.get_text()

    @property
    def x1(self):
        return self.text_cell.x1

    def get_text(self):
        return self.text_cell.get_text()


class PdfTextCellV2(object):

    def __init__(self, text_cell, y_index=None):
        self.text_cell: OcrCell = text_cell
        self.y_index = y_index

    def __str__(self):
        return self.get_text()

    @property
    def x1(self):
        return self.text_cell.left_top.x

    def get_text(self):
        return self.text_cell.get_text_to_show()


class PdfCheckResult(BaseModel):
    """
    pdf_check_result

    """
    model_name: str = None
    document_type: str = None
    begin_time: str = None
    end_time: str = None
    use_time: float = None
    use_time_hour: float = None
    acc: float = None
    tp: int = None
    fp: int = None
    total: int = None
    total_label: int = None
    src_total: int = None

    precision: float = None
    recall: float = None
    f1: float = None

    def calc_f1(self):
        if self.total_label is None or self.total is None:
            return

        self.acc = self.tp / self.total if self.total > 0 else 0
        self.precision = self.tp / self.total if self.total > 0 else 0
        self.recall = self.tp / self.total_label if self.total_label > 0 else 0
        if self.total > 0 and self.total_label > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.f1 = 0


class PdfPageInfo(BaseModel):
    """
   PdfPageInfo

    """
    id: int = None
    src_id: int = None
    table_id: int = None
    current_page: int = None
    total_table: int = None
    total_predict_table: int = None


class PdfTaskParams(BaseModel):
    """
   PdfTaskParams

    """
    model_name: str = "v1.1"
    check_total: int = 2000
    document_type: str = "benchmark"
    delete_check_success: bool = True
    mysql_json_config_filename: Any = None
    thread_num: int = 4
    is_image_pdf: bool = True
    table_metric_id: Any = None
    task_id: int = 0
    run_batch: Any = None
    run_begin_time: Any = None
    need_check_imaged_pdf: bool = True

    tsr_model_name: Any = None
    table_task_type: Any = None
    lang: Any = None
    ocr_task_type: Any = None
