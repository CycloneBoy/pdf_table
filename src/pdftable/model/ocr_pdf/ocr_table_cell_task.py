#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_table_cell_task
# @Author  ：cycloneboy
# @Date    ：20xx/7/20 14:39


import time

from .base_infer_task import BaseInferTask
from ..pdf_table.table_cell_extract import TableCellExtract
from ...utils import logger

"""
OcrTableCellTask
"""

__all__ = [
    'OcrTableCellTask'
]


class OcrTableCellTask(BaseInferTask):

    def __init__(self, task="ocr_table_cell", model="line",
                 **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self.predictor = TableCellExtract(output_dir=self.output_dir)

    def set_output_dir(self, output_dir):
        self.predictor.output_dir = output_dir

    def _construct_model(self, model):
        pass

    def _build_processor(self):
        pass

    def _preprocess(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        results = {"inputs": inputs}
        return results

    def _run_model(self, inputs, **kwargs):
        infer_data_loader = inputs["inputs"]
        is_pdf = kwargs.get("is_pdf", False)
        page = kwargs.get("page", None)
        ocr_system_output = kwargs.get("ocr_system_output", None)

        results = []
        begin = time.time()
        for index, batch in enumerate(infer_data_loader):
            tables, metric = self.predictor.extract_cell(batch, is_pdf=is_pdf,
                                                         page=page,
                                                         ocr_system_output=ocr_system_output)

            results.append({
                "results": tables,
                "metric": metric,
            })

        use_time = time.time() - begin
        inputs["results"] = results
        inputs["use_time"] = use_time
        return inputs

    def _postprocess(self, inputs, **kwargs):
        results = []

        use_time = inputs["use_time"]

        predict_result = inputs["results"]
        for index, batch in enumerate(predict_result):
            predict = batch["results"]

            results.append(predict)

        logger.info(f"推理结束：{len(predict_result)} - 耗时：{use_time:.3f} s / {use_time / 60:.3f} min.")
        return results
