#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : processor_linecell.py
# @Author   : cycloneboy
# @Date     : 20xx/12/16 - 16:50


from typing import Dict, Any

import numpy as np

from pdftable.model.line_cell import LineCellConfig

__all__ = [
    "LineCellPreProcessor",
    "LineCellPostProcessor"
]

from pdftable.utils.ocr import OcrCommonUtils


class LineCellPreProcessor(object):

    def __init__(self, config: LineCellConfig):
        super().__init__()

        self.config = config

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        data_batch = []
        for item in inputs:
            data = {
                "inputs": item
            }

            # data["inputs"] = item
            data_batch.append(data)
        return data_batch


class LineCellPostProcessor(object):

    def __init__(self, config: LineCellConfig, output_dir=None, show_info=True):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.show_info = show_info

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        preds = inputs['results']

        all_bbox_list = []
        all_image_bbox_list = []
        structure_list = []
        logi_list = []
        sep_logi_list = []
        for pred in preds:
            bbox_list = pred['table_bbox']
            structure_str_list = pred['html']
            table_image_bbox = pred.get('table_image_bbox', None)
            if table_image_bbox is not None:
                table_image_bbox = OcrCommonUtils.box_transform(table_image_bbox)
                all_image_bbox_list.extend(table_image_bbox)

            table_image_axis = pred.get('table_image_axis', None)
            if table_image_axis is not None:
                logi_list.extend(table_image_axis)
                sep_logi_list.append(table_image_axis)

            bbox_list = OcrCommonUtils.box_transform(bbox_list)

            all_bbox_list.append(bbox_list)
            structure_list.append(structure_str_list)

        result = {
            'polygons': np.array(all_image_bbox_list),
            'structure_str_list': structure_list,
            'logi': np.array(logi_list),
            'polygons_sep': all_bbox_list,
            'logi_sep': sep_logi_list,
            "inputs": inputs["inputs"],
            "table_cells": inputs['results'],
            "table_cell_metric": {},
        }

        return result
