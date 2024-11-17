#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_cell_extract_from_pdf
# @Author  ：cycloneboy
# @Date    ：20xx/12/28 13:48

import copy
import time
import traceback
from pprint import pprint
from typing import List

import cv2
import numpy as np
from PIL import Image
from pdfminer.layout import LTRect, LTPage

from pdftable.entity import LineDirectionType
from pdftable.entity.table_entity import Line, Point
from pdftable.model.ocr_pdf.ocr_output import OcrSystemModelOutput
from pdftable.model.pdf_table.table_common import (TableProcessUtils
                                                   )
from pdftable.model.pdf_table.table_core import Cell
from pdftable.model.table.line_cell.table_cell_extract_algo import TableCellExtractAlgo
from pdftable.utils import (
    logger,
    FileUtils,
    TimeUtils,
    PdfUtils,
    CommonUtils, Constants, MathUtils,
)
from pdftable.utils.ocr import OcrCommonUtils
from pdftable.utils.table.image_processing import PdfImageProcessor

__all__ = [
    "TableCellExtractFromPdf"
]


class TableCellExtractFromPdf(object):

    def __init__(
            self,
            line_tol=2,

            output_dir=None,
            page=None,
            debug=False,
            **kwargs,
    ):
        self.line_tol = line_tol

        self.image = None
        self.image_scalers = None
        self.pdf_scalers = None

        self.file_name = None
        self.is_pdf = False
        self.image_name = None
        self.raw_filename = None
        self.output_dir = output_dir
        self.is_pdf_image = False

        self.result_http_server = CommonUtils.get_result_http_server(output_dir=self.output_dir)

        self.debug = debug
        self.color_list = CommonUtils.get_color_list()

        self.vertical_segments_v2: List[Line] = None
        self.horizontal_segments_v2: List[Line] = None
        self.image_width = None
        self.image_height = None

        self.pdf_width = None
        self.pdf_height = None
        self.images = []
        self.filtered_images = []
        self.pdf_images = []
        self.pdf_image_mapping = {}
        self.page = page
        self.regions = None
        self.table_cells: List[Cell] = []
        self.pdf_layout: LTPage = None
        self.pdf_dimensions = None

        # self.init()

    def pdf2image(self, file_name, image_name=None):
        image_name, do_convert = PdfImageProcessor.convert_pdf_to_image(file_name=file_name, image_name=image_name)
        return image_name, do_convert

    def save_image_with_name(self, image, name="line_mask.png"):
        save_image_file = f"{self.output_dir}/{self.raw_filename}_{name}"
        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, image)
        logger.info(f"save {name} image：{save_image_file}")

    def extract_table_cell(self, line_list: List[LTRect], joint_list: List[LTRect]):
        horizontal_lines, vertical_lines = self.extract_table_line(line_list)
        joint_points = self.extract_table_joint_points(joint_list)

        merge_vertical_lines = Line.merge_lines(vertical_lines, direction=LineDirectionType.VERTICAL)
        logger.info(f"合并垂直线段：{len(vertical_lines)} -> {len(merge_vertical_lines)}")

        tables = {}
        for line in merge_vertical_lines:
            min_x, max_x = self.find_horizontal_range(y_min=line.min_y,
                                                      y_max=line.max_y,
                                                      line_list=horizontal_lines,
                                                      diff=self.line_tol)
            if self.is_pdf:
                bbox = (min_x, line.max_y, max_x, line.min_y)
            else:
                bbox = (min_x, line.min_y, max_x, line.max_y)
            find_joint_points = self.find_joint_points_by_y(y_min=line.min_y,
                                                            y_max=line.max_y,
                                                            points=joint_points,
                                                            diff=self.line_tol)
            tables[bbox] = [point.to_cv() for point in find_joint_points]

        logger.info(f"发现表格数量：{len(tables)}")

        return tables, horizontal_lines, vertical_lines

    def find_horizontal_lines_by_y(self, y_min, y_max, line_list: List[Line], diff=2) -> List[Line]:
        results = []
        for line in line_list:
            if y_min - diff < line.min_y <= line.max_y < y_max + diff:
                results.append(line)
        return results

    def find_joint_points_by_y(self, y_min, y_max, points: List[Point], diff=2) -> List[Point]:
        results = []
        for point in points:
            if y_min - diff < point.y < y_max + diff:
                results.append(point)
        return results

    def find_horizontal_range(self, y_min, y_max, line_list: List[Line], diff=2):
        min_x = 1e10
        max_x = 0
        for line in line_list:
            if y_min - diff < line.min_y <= line.max_y < y_max + diff:
                if line.min_x <= min_x:
                    min_x = line.min_x
                if line.max_x >= max_x:
                    max_x = line.max_x
        return min_x, max_x

    def extract_table_line(self, line_list: List[LTRect]):
        vertical_lines = []
        horizontal_lines = []
        for rect in line_list:
            rect_h = rect.height
            rect_w = rect.width
            if rect_h < self.line_tol < rect_w:
                left = Point(x=rect.x0, y=(rect.y0 + rect.y1) / 2)
                right = Point(x=rect.x1, y=left.y)
                one_line = Line(left=left, right=right,
                                direction=LineDirectionType.HORIZONTAL,
                                width=rect_w, height=rect_h)
                horizontal_lines.append(one_line)

            elif rect_h > self.line_tol > rect_w:
                left = Point(x=(rect.x0 + rect.x1) / 2, y=rect.y0)
                right = Point(x=left.x, y=rect.y1)
                one_line = Line(left=left, right=right,
                                direction=LineDirectionType.VERTICAL,
                                width=rect_w, height=rect_h)
                vertical_lines.append(one_line)

        return horizontal_lines, vertical_lines

    def extract_table_joint_points(self, joint_list: List[LTRect]) -> List[Point]:
        joint_points = []
        key_set = set()
        for rect in joint_list:
            rect_h = rect.height
            rect_w = rect.width
            if rect_h < self.line_tol and rect_w < self.line_tol:
                point = Point(x=(rect.x0 + rect.x1) / 2, y=(rect.y0 + rect.y1) / 2)
                if point.key() not in key_set:
                    joint_points.append(point)
                    key_set.add(point.key())
        return joint_points

    def extract_table_by_line(self, table_bboxs: np.ndarray = None):
        """
        提取表格
        
        :return:
        """
        if table_bboxs is not None:
            line_list, skip_list = TableCellExtractAlgo.transform_bbox_and_logits_to_rect(table_bboxs=table_bboxs)
        elif self.pdf_layout is not None:
            line_list, other_list, skip_list = PdfUtils.get_pdf_rect_from_layout(layout=self.pdf_layout,
                                                                                 line_max=self.line_tol)
        else:
            dimensions, line_list, other_list, skip_list = PdfUtils.extract_pdf_rect(file_name=self.file_name,
                                                                                     line_max=self.line_tol)

        tables, horizontal_lines, vertical_lines = self.extract_table_cell(line_list=line_list,
                                                                           joint_list=skip_list)

        return tables, horizontal_lines, vertical_lines

    def show_merge_cell(self, table_idx, all_cell_results: List[Cell],
                        image_scalers=None,
                        sep_file_name="_table_cell_", thickness=2):
        """
        显示表格的网格图片

        :param table_idx:
        :param all_cell_results:
        :param image_scalers:
        :param sep_file_name:
        :return:
        """
        src_im = copy.deepcopy(self.image)

        for index, cell in enumerate(all_cell_results):
            if not cell.image_bbox:
                if image_scalers:
                    cell.image_bbox = MathUtils.scale_pdf(cell.bbox, image_scalers)
                else:
                    cell.image_bbox = [round(item) for item in cell.bbox]
            color = self.color_list[cell.row_index % len(self.color_list)]
            cv2.rectangle(src_im, cell.image_bbox[:2], cell.image_bbox[2:], color, thickness)

        save_image_file = f"{self.output_dir}/{self.raw_filename}{sep_file_name}{table_idx + 1}.jpg"
        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, src_im)

        logger.info(f"保存表格cell图像：{len(all_cell_results)} - {save_image_file}")
        return save_image_file

    def extract_cell(self, file_name, page=None, table_bboxs=None,
                     ocr_system_output: OcrSystemModelOutput = None,
                     force_convert_to_image=False):
        if page is not None:
            self.page = page
        # is_image = False
        # if file_name.endswith(".png"):
        #     is_image = True
        #     file_name = file_name.replace(".png", ".pdf")
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()

        self.file_name = file_name
        self.is_pdf = FileUtils.is_pdf_file(file_name)
        self.raw_filename = FileUtils.get_file_name(file_name)
        if self.page is None and self.raw_filename.startswith("page-"):
            self.page = int(self.raw_filename[5:])

        image_name = FileUtils.get_pdf_to_image_file_name(file_name)
        if self.debug and (force_convert_to_image or not FileUtils.check_file_exists(image_name)):
            image_name, do_convert = self.pdf2image(file_name=self.file_name, )
        self.is_pdf_image = file_name != image_name
        self.image_name = image_name

        if self.is_pdf_image:
            layout, dimensions, horizontal_text, vertical_text, images, filtered_images = PdfUtils.get_pdf_object(
                file_name)
            self.pdf_width, self.pdf_height = dimensions
            self.pdf_layout = layout
            self.pdf_dimensions = dimensions

            self.images = images
            self.filtered_images = filtered_images
            self.pdf_images, self.pdf_image_mapping = PdfUtils.save_pdf_image(images=images,
                                                                              output_dir=self.output_dir,
                                                                              image_dir=f"image/{self.page}")
        # if self.debug:
        if self.image is None:
            self.image = cv2.imread(self.image_name)
            self.image_height, self.image_width = self.image.shape[:2]
        if self.pdf_height is not None:
            self.image_scalers, self.pdf_scalers = TableProcessUtils.get_pdf_scaler(image_shape=[self.image_height,
                                                                                                 self.image_width],
                                                                                    pdf_shape=[self.pdf_height,
                                                                                               self.pdf_width])

        # 获取表格区域
        self.regions = TableProcessUtils.get_table_bbox_regions(ocr_system_output, diff=20)

        # 添加表格cell识别的边线添加到threshold
        if ocr_system_output is not None and ocr_system_output.table_structure_result is not None:
            save_html_file = f"{self.output_dir}/{ocr_system_output.raw_filename}"
            self.table_cells = self.get_table_cell_from_ocr_system_output(ocr_system_output,
                                                                          save_html_file=save_html_file)

        # 通过灰度图片提取表格
        table_bbox, horizontal_lines, vertical_lines = self.extract_table_by_line(table_bboxs=table_bboxs)

        table_bbox = TableProcessUtils.table_bbox_merge(table_bbox, diff=10)
        self.table_bbox_unscaled = copy.deepcopy(table_bbox)
        self.table_bbox = table_bbox

        if self.debug:
            save_image_file = f"{self.output_dir}/{self.raw_filename}_table_joint_point.jpg"
            TableProcessUtils.save_table_join_point(image=self.image,
                                                    table_bbox_unscaled=self.table_bbox_unscaled,
                                                    save_image_file=save_image_file,
                                                    pdf_images=self.pdf_images,
                                                    image_scalers=self.image_scalers)

        self.vertical_segments_v2 = vertical_lines
        self.horizontal_segments_v2 = horizontal_lines

        all_table_cell = []
        all_table_html = []
        all_db_table_html = []

        run_time = FileUtils.get_file_name(f"{self.output_dir}.txt")

        metric = {
            "run_time": run_time,
            "file_name": file_name,
            "image_name": self.image_name,
            "image_scalers": self.image_scalers,
            "pdf_scalers": self.pdf_scalers,
            "result_dir_url": f"{self.result_http_server}/{run_time}",
            "show_url": f"{self.result_http_server}/{run_time}/{self.raw_filename}_show.html",
            "line_mask_image": f"{self.output_dir}/{self.raw_filename}_line_mask.png",
            "table_joint_point_image": f"{self.output_dir}/{self.raw_filename}_table_joint_point.jpg",
            "table_metric": []
        }

        pdf_table_algo = TableCellExtractAlgo(
            line_tol=self.line_tol,
            debug=self.debug,
            output_dir=self.output_dir
        )

        tables = []
        for table_idx, table_bbox in enumerate(
                sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=self.is_pdf)
        ):
            try:

                one_metric = {}

                text_bbox, all_cell_results, table_cell_metric = pdf_table_algo.generate_table_cell(table_idx,
                                                                                                    table_bbox,
                                                                                                    table_bbox=self.table_bbox,
                                                                                                    v_segments=vertical_lines,
                                                                                                    h_segments=horizontal_lines,
                                                                                                    pdf_images=self.pdf_images,
                                                                                                    line_tol=self.line_tol * 2,
                                                                                                    is_pdf=self.is_pdf)

                # 过滤单一的表格
                if table_idx == 0 and len(all_cell_results) == 1:
                    logger.info(f"过滤只有一行和一列的表格：{table_idx} - {text_bbox}")
                    continue

                save_html_file = f"{self.output_dir}/{self.raw_filename}_table_{table_idx + 1}.html"
                table_cells, table_html, match_metric, db_table_html = pdf_table_algo.match_table_cell_and_text_cell(
                    table_idx=table_idx,
                    table_cells=all_cell_results,
                    text_cells=text_bbox,
                    save_html_file=save_html_file)

                bbox_scale = table_bbox
                table_image_bbox = None
                table_image_axis = None
                if self.image_scalers is not None:
                    table_cells = TableProcessUtils.convert_table_cell_to_image_scale(table_cells,
                                                                                      image_scalers=self.image_scalers)
                    bbox_scale = MathUtils.scale_pdf(table_bbox, self.image_scalers)
                image_bboxes = []
                image_axis = []
                for cell in table_cells:
                    image_bboxes.append(cell.image_bbox)
                    image_axis.append(cell.get_pred_logit())
                table_image_bbox = np.array(image_bboxes)
                table_image_axis = np.array(image_axis)
                if self.debug:
                    table_cell_image = self.show_merge_cell(table_idx=table_idx,
                                                            all_cell_results=all_cell_results,
                                                            image_scalers=self.image_scalers)

                table_row_dict_sorted = TableProcessUtils.convert_table_cell_to_dict(table_cells)
                # 过滤提取的边界为空的误识别
                table_bboxes = np.array([cell.bbox for cell in table_cells])

                one_table = {
                    "index": table_idx,
                    "is_image": False,
                    # # table point ： lt,rb
                    "bbox": bbox_scale,
                    "bbox_image": bbox_scale,
                    "table_cells": table_row_dict_sorted,
                    "html": "".join(table_html),
                    "table_bbox": table_bboxes,
                    "table_image_bbox": table_image_bbox,
                    "table_image_axis": table_image_axis,
                }

                tables.append(one_table)

                all_table_cell.append(table_cells)
                all_table_html.append(table_html)
                all_db_table_html.append(db_table_html)

                one_metric.update(table_cell_metric)
                one_metric.update(match_metric)
                metric["table_metric"].append(one_metric)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"提取表格异常：{self.file_name} - {table_idx}")

        end_time = time.time()
        use_time = end_time - begin_time

        end_time_str = TimeUtils.now_str()
        use_time_str = TimeUtils.calc_diff_time(begin_time_str, end_time_str)
        metric["begin_time"] = begin_time_str
        metric["end_time"] = end_time_str
        metric["use_time"] = use_time_str
        metric["use_time2"] = use_time

        show_url = metric["result_dir_url"]
        logger.info(f"table cell extract finished, use time：{use_time:.3f} s,"
                    f" extract total table：{len(tables)} - {show_url}")

        if self.debug:
            # 绘制表格CELL结构
            image_draw = OcrCommonUtils.draw_table_cell_boxes(image_full=self.image,
                                                              cell_result=tables)

            image_draw = Image.fromarray(image_draw)
            save_file = f"{self.output_dir}/ocr_{self.raw_filename}_{run_time}.jpg"
            image_draw.save(save_file)
            logger.info(f"save_file: {save_file}")

        return tables, metric

    def __call__(self, inputs, **kwargs):
        tables, metric = self.extract_cell(file_name=inputs, **kwargs)
        return tables

    def get_table_cell_from_ocr_system_output(self, ocr_system_output: OcrSystemModelOutput, save_html_file) -> List[
        Cell]:
        """
        通过 logi 获取table cell

        :param ocr_system_output:
        :param save_html_file:
        :return:
        """
        table_cells = []
        table_bboxs = ocr_system_output.get_table_structure_bboxs()
        if "logi" in ocr_system_output.table_structure_result:
            logits = ocr_system_output.table_structure_result["logi"]

            table_cells = TableProcessUtils.get_table_cell_from_table_logit(table_bboxs=table_bboxs,
                                                                            logits=logits,
                                                                            save_html_file=save_html_file)

        return table_cells


def main():
    output_dir = f"{Constants.HTML_BASE_DIR}/pdf_debug/{TimeUtils.get_time()}"
    # output_dir = f"{Constants.HTML_BASE_DIR}/pdf_debug/2023-12-27"

    file_dir = f"{Constants.PDF_CACHE_BASE}/ocr_file/temp_dir"

    page = 79
    # file_name = f"{file_dir}/page-27.pdf"
    file_name = f"{output_dir}/pdf/page-{page}.pdf"

    debug = False
    # debug = True
    parser = TableCellExtractFromPdf(output_dir=output_dir, page=page, debug=debug)

    tables = parser.extract_cell(file_name)
    pprint(tables)


if __name__ == '__main__':
    main()
