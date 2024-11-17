#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_to_html_task
# @Author  ：cycloneboy
# @Date    ：20xx/7/24 17:40
import os
import time
import traceback
from copy import deepcopy
from typing import List, Dict

import cv2
import numpy as np

from pdftable.entity import HtmlContentType
from pdftable.entity.table_entity import Point, OcrCell
from pdftable.model import TableProcessUtils
from pdftable.model.ocr_pdf import OCRToHtmlConfig
from pdftable.model.ocr_pdf.ocr_output import OcrSystemModelOutput
from pdftable.utils import CommonUtils, FileUtils, logger, PdfUtils, MathUtils

"""
OCR 结果转html 
"""

__all__ = [
    'OcrToHtmlTask'
]


class OcrToHtmlTask(object):

    def __init__(self, config: OCRToHtmlConfig = None, debug=True,
                 output_dir=None, **kwargs):
        super().__init__()
        self.config = config
        self.debug = debug
        self.output_dir = output_dir
        self.src_id = kwargs.get('src_id', None)

        self.result_http_server = CommonUtils.get_result_http_server(output_dir=self.output_dir)

        self.diff = 2

    def convert_to_html(self, ocr_system_output: OcrSystemModelOutput):
        """
        ocr 转html

        :param ocr_system_output:
        :return:
        """
        ocr_results = ocr_system_output.ocr_result
        metric = {}
        remain_cells, table_text_bboxs = self.get_table_text_cell(ocr_system_output)

        # 获取table 之外的ocr
        table_text_bbox_set = set([item.to_key() for item in table_text_bboxs])
        for cell in ocr_results:
            if cell.to_key() not in table_text_bbox_set:
                remain_cells.append(cell)

        ocr_system_output.ocr_cell_content = remain_cells

        html = self.ocr_result_to_html(ocr_system_output)

        table_html_str = "\n".join(html) + "\n"
        save_html_file = f"{self.output_dir}/{ocr_system_output.raw_filename}.html"
        FileUtils.save_to_text(save_html_file, table_html_str)
        logger.info(f"保存识别的html: {save_html_file}")
        ocr_system_output.save_html_file = save_html_file

        return metric

    def ocr_result_to_html(self, ocr_system_output: OcrSystemModelOutput) -> List:
        """
        转html

        :param ocr_system_output:
        :return:
        """
        all_ocr_cells = ocr_system_output.ocr_cell_content

        all_ocr_cells.sort(key=lambda cell: (cell.y1, cell.x1))

        self.parse_text_line_align(ocr_system_output)

        results = []
        for cell in ocr_system_output.merge_ocr_cells:
            results.append(cell.get_show_html())

        ocr_system_output.pdf_html = results
        return results

    def parse_text_line_align(self, ocr_system_output: OcrSystemModelOutput):
        """
        文本数据对齐

        :param ocr_system_output:
        :return:
        """
        ocr_cell_content = PdfUtils.modify_ocr_block_line_type(ocr_system_output.ocr_cell_content)
        ocr_system_output.ocr_cell_content = ocr_cell_content

        merge_ocr_cells = PdfUtils.merge_ocr_text_paragraph(ocr_cell_content)
        ocr_system_output.merge_ocr_cells = merge_ocr_cells

        return ocr_system_output

    def get_table_text_cell(self, ocr_system_output: OcrSystemModelOutput):
        """
        转换 table bbox 到 ocr bbox

        :param ocr_system_output:
        :return:
        """
        table_text_bboxs = []
        remain_cells = []
        for table_idx, table in enumerate(ocr_system_output.table_cell_result):
            table_bbox = table["bbox"]
            is_image = table["is_image"]

            if is_image:
                logger.info(f"当前表格是图片误识别：{table_idx} - {table['bbox']}")
                continue

            is_layout_figure = table["is_layout_figure"]
            if is_layout_figure:
                match_figure = table["match_figure"]
                table_cell, text_bboxs = self.build_layout_image(match_figure, ocr_system_output=ocr_system_output)

            else:
                text_bboxs = table["text_bboxs"]
                table_html = table["table_html"]
                db_table_html = table["db_table_html"]

                left_top = Point(x=table_bbox[0], y=table_bbox[1])
                right_bottom = Point(x=table_bbox[2], y=table_bbox[3])
                table_cell = OcrCell(left_top=left_top, right_bottom=right_bottom,
                                     text="\n".join(db_table_html),
                                     db_text="\n".join(table_html),
                                     cell_type=HtmlContentType.TABLE)

            table_text_bboxs.extend(text_bboxs)

            remain_cells.append(table_cell)
        return remain_cells, table_text_bboxs

    def __call__(self, ocr_system_output: OcrSystemModelOutput):
        start = time.time()
        logger.info(f"开始OCR转html")
        metric = self.convert_to_html(ocr_system_output=ocr_system_output)

        use_time = time.time() - start
        result_dir_url = CommonUtils.get_result_http_server(output_dir=ocr_system_output.save_html_file)

        logger.info(f"结束OCR转html, 耗时：{use_time:.3f} s, "
                    f"提取表格数量：{len(ocr_system_output.table_cell_result)} 个。 "
                    f"结果url: {result_dir_url}")
        return ocr_system_output

    def build_layout_image(self, match_figure, ocr_system_output: OcrSystemModelOutput):
        """
        构造 ocr cell

        :param match_figure:
        :param ocr_system_output:
        :return:
        """
        bbox = [round(item) for item in match_figure["bbox"]]

        # 图片中的text cell
        text_bboxs, remain_cells = TableProcessUtils.get_text_in_table_bbox(bbox=bbox,
                                                                            ocr_results=ocr_system_output.ocr_result,
                                                                            diff=self.diff)

        if ocr_system_output.image_full is None:
            image_name = FileUtils.get_pdf_to_image_file_name(ocr_system_output.file_name)
            raw_image = cv2.imread(image_name)
        else:
            raw_image = ocr_system_output.image_full

        image = deepcopy(raw_image)[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        bbox_str = '_'.join([str(item) for item in bbox])
        save_name = f"image_{bbox_str}.png"

        relative_dir = f"image/{ocr_system_output.page}/{save_name}"

        save_image_file = os.path.join(self.output_dir, relative_dir)
        cv2.imwrite(save_image_file, image)
        logger.info(f"保存layout图片：{save_image_file}")

        height = abs(bbox[3] - bbox[1])
        width = bbox[2] - bbox[0]
        if ocr_system_output.pdf_scalers is not None:
            x1, y1, x2, y2 = MathUtils.scale_pdf(bbox, ocr_system_output.pdf_scalers)
            height = abs(y2 - y1)
            width = abs(x2 - x1)

        image_info = {
            "key": f"image_{bbox_str}",
            "name": FileUtils.get_file_name(save_image_file),
            "raw_name": save_name,
            "save_name": save_image_file,
            "relative_dir": f"./{relative_dir}",
            "bbox": bbox,
            "height": height,
            "width": width,
            "image_size": image.shape[:2],
        }

        image_info_file_name = f"{self.output_dir}/image/{ocr_system_output.page}/image.json"
        if FileUtils.check_file_exists(image_info_file_name):
            image_infos = FileUtils.load_json(image_info_file_name)
            image_infos.append(image_info)
            FileUtils.dump_json(image_info_file_name, image_infos)

        raw_data = {
            "is_image": True,
            "text": save_image_file,
            "bbox": np.array(bbox).reshape([2, 2]),
            "image_info": image_info,
        }
        table_cell = OcrCell(raw_data=raw_data, cell_type=HtmlContentType.IMAGE)

        return table_cell, text_bboxs
