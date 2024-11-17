#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_pdf_text_task
# @Author  ：cycloneboy
# @Date    ：20xx/7/31 15:54
import time
from copy import deepcopy
from typing import List, cast

import cv2
import numpy as np
from pdfminer.layout import LTTextBoxVertical, LTText, LTImage

from .ocr_output import OcrSystemModelOutput
from ..pdf_table.table_common import TableProcessUtils
from ...entity import OcrCell, HtmlContentType
from ...utils import CommonUtils, logger, PdfUtils, FileUtils, MathUtils

"""
OcrPdfTextTask
"""

__all__ = [
    'OcrPdfTextTask'
]


class OcrPdfTextTask(object):

    def __init__(self, config=None, debug=True, layout_kwargs=None,
                 output_dir=None, **kwargs):
        super().__init__()
        self.config = config
        self.debug = debug
        self.layout_kwargs = layout_kwargs if layout_kwargs is not None else {}
        self.output_dir = output_dir
        self.src_id = kwargs.get('src_id', None)

        self.diff = 2

        self.pdf_width = None
        self.pdf_height = None
        self.pdf_images = []
        self.pdf_image_mapping = {}

        self.image_width = None
        self.image_height = None

        self.image_scalers = None
        self.pdf_scalers = None
        self.color_list = CommonUtils.get_color_list()

    def __call__(self, ocr_system_output: OcrSystemModelOutput, layout_kwargs=None) -> List[OcrCell]:
        start = time.time()
        logger.info(f"开始PDF提取text。")
        result_text_cells = self.extract_text(ocr_system_output=ocr_system_output)
        text_cells = self.convert_text_cell_to_ocr_cell(result_text_cells)
        self.show_text_cells(ocr_system_output=ocr_system_output, ocr_cells=text_cells)

        use_time = time.time() - start
        logger.info(f"结束PDF提取text, 耗时：{use_time} s, "
                    f"提取text数量：{len(text_cells)} 个。")
        return text_cells

    def extract_text(self, ocr_system_output: OcrSystemModelOutput, layout_kwargs=None):
        """
        提取文字
        :param ocr_system_output:
        :param layout_kwargs:
        :return:
        """
        file_name = ocr_system_output.file_name
        image_height = ocr_system_output.image_shape[0]
        image_width = ocr_system_output.image_shape[1]

        layout_kwargs = layout_kwargs if layout_kwargs is not None else self.layout_kwargs
        layout, dimensions, horizontal_text, vertical_text, images, filtered_images = PdfUtils.get_pdf_object(file_name,
                                                                                                              layout_kwargs)

        self.pdf_images, self.pdf_image_mapping = PdfUtils.save_pdf_image(images=images,
                                                                          output_dir=self.output_dir,
                                                                          image_dir=f"image/{ocr_system_output.page}")
        self.pdf_width, self.pdf_height = dimensions

        self.image_scalers, self.pdf_scalers = TableProcessUtils.get_pdf_scaler(image_shape=[image_height, image_width],
                                                                                pdf_shape=[self.pdf_height,
                                                                                           self.pdf_width])
        ocr_system_output.image_scalers = self.image_scalers
        ocr_system_output.pdf_scalers = self.pdf_scalers

        remain_text_bbox = {
            "horizontal": horizontal_text,
            "vertical": vertical_text,
        }

        result_text_cells = {
            "horizontal": [],
            "vertical": [],
        }

        for index, table in enumerate(ocr_system_output.table_cell_result):
            table_bbox = table["bbox"]
            # bbox 是 lt,rb 而 pdf 上的box 是lb,rt
            table_bbox_new = [table_bbox[0], table_bbox[3], table_bbox[2], table_bbox[1]]
            bbox = MathUtils.scale_image_bbox(bbox=table_bbox_new, factors=self.pdf_scalers)

            match, match_image, remain_images = TableProcessUtils.check_table_match_images(bbox,
                                                                                           images=images)
            if match:
                # if match_image.name not in [item.name for item in add_images]:
                #     add_images.append(match_image)
                table["is_image"] = True
                logger.info(f"当前table是误识别：{index} - {table} - 匹配到图片：{match_image.name}")
                continue

            image_table_cells = deepcopy(table["table_cells"])
            table_cells = self.convert_table_cell_to_pdf(image_table_cells)

            text_cells, remain_text_bbox = PdfUtils.get_pdf_text_in_bbox(bbox=bbox,
                                                                         horizontal_text=remain_text_bbox["horizontal"],
                                                                         vertical_text=remain_text_bbox["vertical"])
            # text box 拆分: 将一行文本box 跨多个cell的拆分
            if "not_spit_text" in table:
                new_text_cells = text_cells
            else:
                new_text_cells = TableProcessUtils.text_box_split_to_cell(table_cells=table_cells,
                                                                          text_cells=text_cells)
            result_text_cells["horizontal"].extend(new_text_cells["horizontal"])
            result_text_cells["vertical"].extend(new_text_cells["vertical"])

        # remain_image
        result_text_cells["horizontal"].extend(images)

        # remain_text_bbox
        result_text_cells["horizontal"].extend(remain_text_bbox["horizontal"])
        result_text_cells["vertical"].extend(remain_text_bbox["vertical"])

        return result_text_cells

    def convert_text_cell_to_ocr_cell(self, result_text_cells) -> List[OcrCell]:

        all_text_cells = []
        all_text_cells.extend(result_text_cells["horizontal"])
        all_text_cells.extend(result_text_cells["vertical"])

        text_cells = []
        for index, cell in enumerate(all_text_cells):
            # 而 pdf 上的box 是lb,rt
            x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x0, cell.y1, cell.x1, cell.y0), self.image_scalers)

            bbox = [x1, y1, x2, y1, x2, y2, x1, y2]
            # new_bbox = OcrCommonUtils.order_point(bbox)

            is_image = False
            image_info = None

            # 垂直文字
            if isinstance(cell, LTTextBoxVertical):
                raw_text = "\n".join(
                    cast(LTText, obj).get_text() for obj in cell if isinstance(obj, LTText)
                )
            elif isinstance(cell, LTImage):
                is_image = True
                image_info = self.pdf_image_mapping.get(PdfUtils.get_pdf_image_key(cell), None)
                if image_info is not None:
                    raw_text = image_info["save_name"]
                else:
                    raw_text = ""
                    logger.info(f"没有找到对应的PDF IMAGE: {cell}")
            else:
                raw_text = cell.get_text().strip("\n")

            ocr_text = {
                "index": index + 1,
                "is_image": is_image,
                "text": raw_text,
                "bbox": np.array(bbox).reshape([4, 2]),
            }

            if is_image and image_info is not None:
                ocr_text["image_info"] = image_info

            text_cells.append(ocr_text)

        # 转换ocr text cell
        ocr_cells = []
        for one_item in text_cells:
            cell = OcrCell(raw_data=one_item, cell_type=HtmlContentType.TXT)
            ocr_cells.append(cell)

        return ocr_cells

    def convert_table_cell_to_pdf(self, image_table_cells):
        table_cells = []
        for k, v in image_table_cells.items():
            for cell in v:
                x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x1, cell.y1, cell.x2, cell.y2), self.pdf_scalers)
                cell.x1 = x1
                cell.y1 = y1
                cell.x2 = x2
                cell.y2 = y2

                table_cells.append(cell)

        return table_cells

    def show_text_cells(self, ocr_system_output: OcrSystemModelOutput, ocr_cells: List[OcrCell]):
        if self.output_dir is not None and self.debug:
            file_name = ocr_system_output.file_name
            save_image_file = f"{self.output_dir}/{FileUtils.get_file_name(file_name)}_text_cell.jpg"
            image_file = ocr_system_output.image_name
            image = cv2.imread(image_file)
            cell_text_image_file = TableProcessUtils.show_text_cell(image=image,
                                                                    save_image_file=save_image_file,
                                                                    text_cells=[ocr_cells],
                                                                    image_scalers=self.image_scalers)
