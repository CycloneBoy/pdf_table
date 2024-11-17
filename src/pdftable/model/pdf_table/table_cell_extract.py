#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_cell_extract
# @Author  ：cycloneboy
# @Date    ：20xx/7/19 16:51

import copy
import time
import traceback
from typing import List, Dict

import cv2
import numpy as np

from pdftable.entity.table_entity import Line
from pdftable.model.ocr_pdf.ocr_output import OcrSystemModelOutput
from pdftable.model.pdf_table.table_common import (equal_point, equal_in,
                                                   cell_equal_in_list, equal,
                                                   cell_equal_in, calc_cell_width,
                                                   cell_horizontal_line_exists_v2, cell_vertical_line_exists_v2,
                                                   TableProcessUtils, point_in_bbox_list
                                                   )
from pdftable.model.pdf_table.table_core import Cell
from pdftable.utils import (
    logger,
    Constants,
    MathUtils,
    FileUtils,
    TimeUtils,
    PdfUtils,
    CommonUtils,
)
from pdftable.utils.table.image_processing import PdfImageProcessor

"""
表格结构识别
    - 传统基于线段的方法
    - PDF table
"""


class TableCellExtract(object):

    def __init__(
            self,
            process_background=False,
            line_scale=40,
            line_scale_vertical=50,
            line_tol=2,
            joint_tol=2,
            line_mark_tol=6,
            threshold_block_size=15,
            threshold_constant=-2,
            iterations=0,
            diff_angle=400,

            output_dir=None,
            page=None,
            debug=False,
            **kwargs,
    ):
        self.process_background = process_background
        self.line_scale = line_scale
        self.line_scale_vertical = line_scale_vertical
        self.line_tol = line_tol
        self.line_mark_tol = line_mark_tol
        self.threshold_block_size = threshold_block_size
        self.threshold_constant = threshold_constant
        self.iterations = iterations
        self.diff_angle = diff_angle

        self.image = None
        self.threshold = None
        self.image_scalers = None
        self.pdf_scalers = None

        self.file_name = None
        self.is_pdf = True
        self.image_name = None
        self.raw_filename = None
        self.output_dir = output_dir
        self.is_pdf_image = False

        self.result_http_server = CommonUtils.get_result_http_server(output_dir=self.output_dir)

        self.debug = debug
        self.color_list = CommonUtils.get_color_list()

        self.avg_vertical_width = None
        self.avg_horizontal_height = None

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

        # self.init()

    def init(self):
        self.pdf2image(file_name=self.file_name, image_name=self.image_name)

    def pdf2image(self, file_name, image_name=None):
        image_name, do_convert = PdfImageProcessor.convert_pdf_to_image(file_name=file_name, image_name=image_name)
        self.is_pdf_image = file_name != image_name

        return image_name, do_convert

    def extract_cell(self, file_name, is_pdf=False, page=None, ocr_system_output: OcrSystemModelOutput = None):
        if page is not None:
            self.page = page
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()

        self.file_name = file_name
        self.is_pdf = is_pdf
        self.raw_filename = FileUtils.get_file_name(file_name)

        image_name, do_convert = self.pdf2image(file_name=self.file_name, )
        self.image_name = image_name

        # 是否是已经旋转过
        if image_name.find("_rotate") > -1:
            rotated_image = image_name
            logger.info(f"图片已经进行过旋转修复: {image_name}")

        # rotated_image = PdfImageProcessor.rotate_image_v2(image_name, save_image_file=None, )
        if self.is_pdf_image:
            layout, dimensions, horizontal_text, vertical_text, images, filtered_images = PdfUtils.get_pdf_object(
                file_name)
            self.pdf_width, self.pdf_height = dimensions

            self.images = images
            self.filtered_images = filtered_images
            self.pdf_images, self.pdf_image_mapping = PdfUtils.save_pdf_image(images=images,
                                                                              output_dir=self.output_dir,
                                                                              image_dir=f"image/{self.page}")
            rotated_image = image_name
            logger.info(f"pdf image not need rotated: {file_name}")
        elif ocr_system_output is None or (ocr_system_output is not None and not ocr_system_output.image_rotate):
            logger.info(f"开始图片旋转修复: {image_name}")
            rotated_image, angle = PdfImageProcessor.rotate_image(image_name, save_image_file=None,
                                                                  threshold_block_size=self.threshold_block_size,
                                                                  threshold_constant=self.threshold_constant,
                                                                  line_scale_horizontal=self.line_scale,
                                                                  line_scale_vertical=self.line_scale,
                                                                  iterations=self.iterations,
                                                                  diff_angle=self.diff_angle)
            logger.info(f"完成图片旋转修复: {image_name}")
        else:
            rotated_image = image_name

        self.image, self.threshold = PdfImageProcessor.adaptive_threshold(imagename=rotated_image,
                                                                          process_background=self.process_background,
                                                                          blocksize=self.threshold_block_size,
                                                                          c=self.threshold_constant,
                                                                          )
        image_width = self.image.shape[1]
        image_height = self.image.shape[0]
        self.image_width = image_width
        self.image_height = image_height

        if self.is_pdf_image and len(self.filtered_images) > 0:
            image_scalers, pdf_scalers = TableProcessUtils.get_pdf_scaler(image_shape=[self.image_height,
                                                                                       self.image_width],
                                                                          pdf_shape=[self.pdf_height,
                                                                                     self.pdf_width])
            self.threshold = PdfImageProcessor.add_pdf_point_to_image(self.threshold,
                                                                      images=self.filtered_images,
                                                                      image_scalers=image_scalers)
            logger.info(f"PDF图片中的虚线点绘制到threshold上，{len(self.filtered_images)} 个点。")

        # 获取表格区域
        self.regions = TableProcessUtils.get_table_bbox_regions(ocr_system_output, diff=20)

        # 添加表格cell识别的边线添加到threshold
        if ocr_system_output is not None and ocr_system_output.table_structure_result is not None:
            if self.debug:
                self.save_image_with_name(image=self.threshold, name="threshold.jpg")

            self.threshold = TableProcessUtils.add_box_to_image(self.threshold,
                                                                bboxs=ocr_system_output.get_table_structure_bboxs(),
                                                                layout_tables=self.regions,
                                                                )
            save_html_file = f"{self.output_dir}/{ocr_system_output.raw_filename}"
            self.table_cells = self.get_table_cell_from_ocr_system_output(ocr_system_output,
                                                                          save_html_file=save_html_file)

            if self.debug:
                self.save_image_with_name(image=self.threshold, name="threshold_add_box.jpg")

        # 通过灰度图片提取表格
        (table_bbox, horizontal_segments, vertical_segments,
         horizontal_segments_v2, vertical_segments_v2) = self.extract_table_by_line()

        table_bbox = TableProcessUtils.table_bbox_merge(table_bbox, diff=10)
        self.table_bbox_unscaled = copy.deepcopy(table_bbox)

        pdf_scalers = (1, 1, image_height)
        image_scalers = (1, 1, image_height)

        self.image_scalers = image_scalers
        self.pdf_scalers = pdf_scalers

        # 保存表格交点图像
        if len(self.pdf_images) > 0:
            self.scale_image_bbox_to_image()

        if self.debug:
            save_image_file = f"{self.output_dir}/{self.raw_filename}_table_joint_point.jpg"
            TableProcessUtils.save_table_join_point(image=self.image,
                                                    table_bbox_unscaled=self.table_bbox_unscaled,
                                                    save_image_file=save_image_file,
                                                    pdf_images=self.pdf_images, )

        # 转换坐标到pdf : 这里还是image
        self.table_bbox, self.vertical_segments, self.horizontal_segments = MathUtils.scale_image(
            table_bbox, vertical_segments, horizontal_segments, pdf_scalers
        )

        self.vertical_segments_v2 = vertical_segments_v2
        self.horizontal_segments_v2 = horizontal_segments_v2

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

        tables = []
        # sort tables based on y-coord
        for table_idx, table_bbox in enumerate(
                sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=True)
        ):
            try:
                bbox_scale = MathUtils.scale_pdf(table_bbox, self.image_scalers)

                if self.is_pdf_image:
                    table_bbox_pdf = self.scale_table_bbox_to_pdf(table_bbox=bbox_scale)
                    tabel_match, match_image, remain_images = TableProcessUtils.check_table_match_images(table_bbox_pdf,
                                                                                                         images=self.images)
                    if tabel_match:
                        logger.info(f"当前table是误识别：{table_idx} - {table_bbox} - 匹配到图片：{match_image.name}")
                        continue

                one_metric = {}
                # cols, rows, v_s, h_s = self._generate_columns_and_rows(table_idx, table_bbox)

                text_bbox, all_cell_results, table_cell_metric = self.generate_table_cell(table_idx, table_bbox,
                                                                                          line_tol=self.avg_horizontal_height * 2)

                # 过滤单一的表格
                if table_idx == 0 and len(all_cell_results) == 1:
                    logger.info(f"过滤只有一行和一列的表格：{table_idx} - {text_bbox}")
                    # table_cell_metric.update(metric)
                    # self.delete_not_found_table(table_cell_metric)
                    continue

                table_cells, table_html, match_metric, db_table_html = self.match_table_cell_and_text_cell(
                    table_idx=table_idx,
                    table_cells=all_cell_results,
                    text_cells=text_bbox)

                # cell 坐标系从pdf 转换成image
                for cell in table_cells:
                    x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x1, cell.y1, cell.x2, cell.y2), self.image_scalers)
                    cell.x1 = x1
                    cell.y1 = y1
                    cell.x2 = x2
                    cell.y2 = y2
                    cell.clean_text()

                image_bboxes = []
                image_axis = []
                for cell in table_cells:
                    image_bboxes.append(cell.bbox)
                    image_axis.append(cell.get_pred_logit())
                table_image_bbox = np.array(image_bboxes)
                table_image_axis = np.array(image_axis)

                table_row_dict_sorted = TableProcessUtils.convert_table_cell_to_dict(table_cells)
                # 过滤提取的边界为空的误识别
                table_bbox = np.array([cell.bbox for cell in table_cells])

                one_table = {
                    "index": table_idx,
                    "is_image": False,
                    # # table point ： lb,rt
                    "bbox": [bbox_scale[0], bbox_scale[3], bbox_scale[2], bbox_scale[1]],
                    "table_cells": table_row_dict_sorted,
                    "html": "".join(table_html),
                    "table_bbox": table_bbox,
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

        return tables, metric

    def __call__(self, inputs, **kwargs):
        tables, metric = self.extract_cell(file_name=inputs, **kwargs)
        return tables

    def filter_cell_by_rule(self, table_cells: Dict[str, List[Cell]]):
        pass

    def extract_table_by_line(self):
        """
        通过灰度图片提取表格

        :return:
        """
        vertical_mask, vertical_segments, \
            avg_vertical_angle, self.avg_vertical_width, vertical_segments_v2 = PdfImageProcessor.find_lines_angle(
            self.threshold,
            regions=self.regions,
            direction="vertical",
            line_scale=self.line_scale_vertical,
            iterations=self.iterations,
        )

        horizontal_mask, horizontal_segments, \
            avg_horizontal_angle, self.avg_horizontal_height, horizontal_segments_v2 = PdfImageProcessor.find_lines_angle(
            self.threshold,
            regions=self.regions,
            direction="horizontal",
            line_scale=self.line_scale,
            iterations=self.iterations,
        )

        logger.info(f"current image：horizontal_angle - {avg_horizontal_angle} - {self.avg_horizontal_height}")
        logger.info(f"current image：vertical_angle - {avg_vertical_angle} - {self.avg_vertical_width}")

        line_mask = vertical_mask + horizontal_mask
        if self.debug:
            self.save_image_with_name(line_mask)

        contours = PdfImageProcessor.find_contours(vertical_mask, horizontal_mask)
        table_bbox = PdfImageProcessor.find_joints(contours, vertical_mask, horizontal_mask)

        return table_bbox, horizontal_segments, vertical_segments, horizontal_segments_v2, vertical_segments_v2

    def extract_table_by_react(self):
        """
        通过PDF react 提取表格

        :return:
        """
        pass

    def save_image_with_name(self, image, name="line_mask.png"):
        save_image_file = f"{self.output_dir}/{self.raw_filename}_{name}"
        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, image)
        logger.info(f"save {name} image：{save_image_file}")

    def generate_table_cell(self, table_idx, bbox, line_tol=4):
        """
        select elements which lie within table_bbox

        :param table_idx:
        :param bbox:
        :param line_tol:
        :return:
        """
        text_bbox, h_s, v_s = self.get_text_in_table_bbox(bbox)

        bbox_scale = MathUtils.scale_pdf(bbox, self.image_scalers)
        text_bbox_v2, h_s_v2, v_s_v2, = self.get_text_in_table_bbox_v2(bbox_scale)

        raw_joint_point = self.table_bbox[bbox]
        # filter join point by image
        joint_point = self.filter_joint_point_match_image(raw_joint_point)
        logger.info(f"filter join point by image: {len(joint_point)} - {len(raw_joint_point)} "
                    f"-filter total：{len(raw_joint_point) - len(joint_point)} point.")

        cols = []
        rows = []

        for x, y in joint_point:
            cols.append(x)
            rows.append(y)

        # 添加边框的两个点
        for x in [bbox[0], bbox[2]]:
            if not equal_in(x, same_list=cols, diff=line_tol):
                cols.append(x)

        for y in [bbox[1], bbox[3]]:
            if not equal_in(y, same_list=rows, diff=line_tol):
                rows.append(y)

        # sort horizontal and vertical segments
        cols = PdfUtils.merge_close_lines(sorted(cols), line_tol=line_tol, last_merge_threold=20)
        rows = PdfUtils.merge_close_lines(sorted(rows, reverse=True), line_tol=line_tol, last_merge_threold=20)

        # 归一化 相交点
        new_joint_point = []
        for x, y in joint_point:
            new_x = PdfImageProcessor.find_close_norm_x(x, norm_list=cols, atol=line_tol)
            new_y = PdfImageProcessor.find_close_norm_x(y, norm_list=rows, atol=line_tol)

            new_joint_point.append((new_x, new_y))

        min_col = min(cols)
        max_col = max(cols)
        min_row = min(rows)
        max_row = max(rows)
        one_col_width = abs(max_col - min_col)
        one_row_height = abs(max_row - min_row)

        # make grid using x and y coord of shortlisted rows and cols
        col_pairs = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        row_pairs = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1)]

        col_avg_width = calc_cell_width(col_pairs, min_width=self.line_mark_tol)
        row_avg_width = calc_cell_width(row_pairs, min_width=self.line_mark_tol)

        # 生成所有的cell m x n
        cells = [[Cell(c[0], r[0], c[1], r[1], row_index=row_index, col_index=col_index)
                  for col_index, c in enumerate(col_pairs)]
                 for row_index, r in enumerate(row_pairs)]

        # 添加四个角点
        joint_point_cornets = [(min_col, min_row), (max_col, min_row), (max_col, max_row), (min_col, max_row)]
        for index, point in enumerate(joint_point_cornets):
            if point not in new_joint_point:
                new_joint_point.append(point)
                # logger.info(f"添加表格边框的四个角点：{index} - {point}")

        self.cell_line_mark_v2(cells, h_s=h_s_v2, v_s=v_s_v2,
                               diff=self.line_mark_tol,
                               col_avg_width=col_avg_width,
                               row_avg_width=row_avg_width)

        self.cell_line_mark_check(cells, )

        table_cell_image = ""
        if self.debug:
            table_cell_image = self.show_cell_line(table_idx=table_idx,
                                                   cells=cells, image_scalers=self.image_scalers)

        logger.info(f"col_pairs: {len(col_pairs)} - {col_pairs}")
        logger.info(f"row_pairs: {len(row_pairs)} - {row_pairs}")
        if self.debug:
            logger.info(f"cells: {len(cells)} - {cells}")

        sorted_joint_point = sorted(new_joint_point, key=lambda x: (x[1], x[0]))

        # 进行行合并
        column_merge_cells = self.merge_column_cell(sorted_joint_point,
                                                    h_s=h_s,
                                                    v_s=v_s,
                                                    row_pairs=row_pairs,
                                                    col_pairs=col_pairs,
                                                    cells=cells)

        row_merge_cells, remove_cells = self.merge_row_cell(sorted_joint_point=sorted_joint_point,
                                                            row_pairs=row_pairs,
                                                            col_pairs=col_pairs,
                                                            column_merge_cells=column_merge_cells)

        all_cells = self.get_merge_cell(column_merge_cells=column_merge_cells,
                                        row_merge_cells=row_merge_cells,
                                        remove_cells=remove_cells, )

        all_cell_results = TableProcessUtils.modify_cell_info(all_cells,
                                                              cols=cols,
                                                              rows=rows,
                                                              one_col_width=one_col_width,
                                                              one_row_height=one_row_height)
        if self.debug:
            table_cell_image = self.show_merge_cell(table_idx=table_idx,
                                                    all_cell_results=all_cell_results,
                                                    image_scalers=self.image_scalers)

        x1, y1, x2, y2 = MathUtils.scale_pdf(bbox, self.image_scalers)

        metric = {
            "table_idx": table_idx,
            "joint_point": joint_point,
            "new_joint_point": new_joint_point,
            "col_pairs": col_pairs,
            "row_pairs": row_pairs,
            "table_cell_image": table_cell_image,
            "image_table_bbox": [x1, y1, x2, y2],
            "pdf_table_bbox": bbox,
        }

        return text_bbox, all_cell_results, metric

    def get_text_in_table_bbox(self, bbox):
        """
        筛选 在表格区域中的文本

        :param bbox:
            Tuple (x1, y1, x2, y2) representing a bounding box where
            (x1, y1) -> lb and (x2, y2) -> rt in PDFMiner coordinate
            space.
        :return:
        """
        text_bbox = {}
        v_s, h_s = PdfUtils.segments_in_bbox(
            bbox, self.vertical_segments, self.horizontal_segments
        )
        return text_bbox, h_s, v_s

    def get_text_in_table_bbox_v2(self, bbox):
        """
        筛选 在表格区域中的文本

        :param bbox:
            Tuple (x1, y1, x2, y2) representing a bounding box where
            (x1, y1) -> lb and (x2, y2) -> rt in PDFMiner coordinate
            space.
        :return:
        """
        text_bbox = {}

        v_s, h_s = PdfUtils.segments_in_bbox_v2(
            bbox, self.vertical_segments_v2, self.horizontal_segments_v2
        )
        return text_bbox, h_s, v_s

    def cell_line_mark_v2(self, cells: List[List[Cell]], h_s: List[Line], v_s: List[Line], diff=6,
                          col_avg_width=None, row_avg_width=None):
        """
        根据线条 给每一个cell 设置四条边是否是实体线

        :param cells:
        :param h_s:
        :param v_s:
        :param diff:
        :param col_avg_width:
        :param row_avg_width:
        :return:
        """
        v_s.sort(key=lambda x: (x.min_y, x.min_x))
        h_s.sort(key=lambda x: (x.min_x, x.min_y))

        total_row = len(cells)
        total_col = len(cells[0])
        for index, row_cells in enumerate(cells):
            for col_index, cell in enumerate(row_cells):
                self.one_cell_mark_line_v2(cell=cell, h_s=h_s, v_s=v_s, diff=diff,
                                           col_avg_width=col_avg_width,
                                           row_avg_width=row_avg_width)
                if index == 0 and not cell.top:
                    # logger.info(f"第一行,top 属性强制设置为True: {cell}")
                    cell.top = True
                elif index == total_row - 1 and not cell.bottom:
                    # logger.info(f"最后一行,bottom 属性强制设置为True: {cell}")
                    cell.bottom = True

                if col_index == 0 and not cell.left:
                    # logger.info(f"第一列,left 属性强制设置为True: {cell}")
                    cell.left = True
                elif col_index == total_col - 1 and not cell.right:
                    # logger.info(f"第一列,right 属性强制设置为True: {cell}")
                    cell.right = True

    def one_cell_mark_line_v2(self, cell: Cell, h_s: List[Line], v_s: List[Line], diff=2,
                              need_sort=False,
                              col_avg_width=None, row_avg_width=None):
        """
        标记一个cell 的 边框

        :param cell:
        :param h_s:
        :param v_s:
        :param diff:
        :param need_sort:
        :param col_avg_width:
        :param row_avg_width:
        :return:
        """
        if need_sort:
            v_s.sort(key=lambda x: (x.min_y, x.min_x))
            h_s.sort(key=lambda x: (x.min_x, x.min_y))

        cell_scale = self.scale_cell_to_image(cell)

        line_top_exists, line_bottom_exists = cell_horizontal_line_exists_v2(cell=cell_scale,
                                                                             horizontal_line_list=h_s,
                                                                             diff=diff,
                                                                             row_avg_width=row_avg_width,
                                                                             avg_horizontal_height=self.avg_horizontal_height)
        if line_top_exists:
            cell.top = True
        if line_bottom_exists:
            cell.bottom = True
        line_left_exists, line_right_exists = cell_vertical_line_exists_v2(cell=cell_scale,
                                                                           vertical_line_list=v_s,
                                                                           diff=diff,
                                                                           col_avg_width=col_avg_width,
                                                                           avg_vertical_width=self.avg_vertical_width)
        if line_left_exists:
            cell.left = True
        if line_right_exists:
            cell.right = True

    def scale_cell_to_image(self, cell: Cell):
        """
        转换cell 到image 坐标

        :param cell:
        :return:
        """
        cell_scale = copy.deepcopy(cell)
        x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x1, cell.y1, cell.x2, cell.y2), self.image_scalers)
        cell_scale.x1 = x1
        cell_scale.y1 = y1
        cell_scale.x2 = x2
        cell_scale.y2 = y2
        return cell_scale

    def merge_column_cell(self, sorted_joint_point, h_s: List, v_s: List, row_pairs, col_pairs, cells):
        """
        进行每一行的 cell 合并

        :param sorted_joint_point:
        :param h_s:
        :param v_s:
        :param row_pairs:
        :param col_pairs:
        :param cells:
        :return:
        """
        new_cells = []
        remove_cells = []
        for index, (begin_y, end_y) in enumerate(row_pairs):
            old_row_cells = cells[index]

            # 合并列
            need_merge_cells, remain_cells = self.get_column_need_merge_cells(find_cells=old_row_cells)
            new_merge_cells = []
            if len(need_merge_cells) > 1:
                # 单独的cell
                new_merge_cells.extend(remain_cells)
                begin_cell = need_merge_cells[0]
                for start_index in range(1, len(need_merge_cells)):
                    next_cell = need_merge_cells[start_index]

                    if equal_point(begin_cell.rb, next_cell.lb) \
                            and equal_point(begin_cell.rt, next_cell.lt) \
                            and not begin_cell.right and not next_cell.left:
                        merge_cell = Cell(x1=begin_cell.x1, y1=begin_cell.y1,
                                          x2=next_cell.x2, y2=next_cell.y2,
                                          row_index=index, col_index=begin_cell.col_index,
                                          )
                        merge_cell.top = True if begin_cell.top and next_cell.top else False
                        merge_cell.bottom = True if begin_cell.bottom and next_cell.bottom else False
                        merge_cell.left = begin_cell.left
                        merge_cell.right = next_cell.right
                        merge_cell.inner_cells.extend(begin_cell.inner_cells)
                        merge_cell.inner_cells.append(next_cell)

                        # 需要删除的点
                        for cell_remove in [begin_cell, next_cell]:
                            if not cell_equal_in_list(cell_remove, remove_cells) \
                                    and cell_equal_in_list(cell_remove, need_merge_cells):
                                remove_cells.append(cell_remove)
                        begin_cell = merge_cell
                    else:
                        begin_cell = next_cell

                    # 可能没有右边线，人为添加的边
                    # point_equal_in(next_cell.rb, sorted_joint_point) \
                    #                             and point_equal_in(next_cell.rt, sorted_joint_point) \
                    #                             and
                    if begin_cell.left and begin_cell.right:
                        new_merge_cells.append(begin_cell)
                        if start_index < len(need_merge_cells) - 1:
                            begin_cell = need_merge_cells[start_index + 1]
            else:
                new_merge_cells = old_row_cells

            if len(new_merge_cells) > 0:
                new_merge_cells.sort(key=lambda x: x.x1)
                new_cells.append(new_merge_cells)

        logger.info(f"column_merge_cells: {len(new_cells)} - {new_cells}")
        return new_cells

    def get_column_need_merge_cells(self, find_cells: List[Cell]):
        """
        判断cell 的左边和右边是否存在边

        :param find_cells:
        :return:
        """
        need_merge_cells = []
        remain_cells = []
        for index, cell in enumerate(find_cells):
            if cell is not None and (not cell.left or not cell.right):
                need_merge_cells.append(cell)
            else:
                remain_cells.append(cell)
        return need_merge_cells, remain_cells

    def merge_row_cell(self, sorted_joint_point, row_pairs, col_pairs, column_merge_cells):
        """
        进行列 cell 合并
        :param sorted_joint_point:
        :param row_pairs:
        :param col_pairs:
        :param column_merge_cells:
        :return:
        """

        new_cells2 = []
        remove_cells = []
        for index, (begin_x, end_x) in enumerate(col_pairs):
            # 第一列 所有joint point

            x_joints = set()
            y_joints = set()
            find_joint_point = []
            for x, y in sorted_joint_point:
                if equal(x, begin_x) or equal(x, end_x):
                    find_joint_point.append((x, y))
                    x_joints.add(x)
                    y_joints.add(y)

            # 找到需要合并的列
            find_cells = self.find_cell_joint_point_list(find_joint_point, column_merge_cells,
                                                         begin_x=begin_x, )

            # 合并列
            need_merge_cells = self.get_row_need_merge_cells(find_cells)

            new_merge_cells = []
            if len(need_merge_cells) > 1:
                begin_cell = need_merge_cells[0]
                for start_index in range(1, len(need_merge_cells)):
                    next_cell = need_merge_cells[start_index]

                    if equal_point(begin_cell.lt, next_cell.lb) \
                            and equal_point(begin_cell.rt, next_cell.rb) \
                            and not begin_cell.bottom and not next_cell.top:
                        merge_cell = Cell(x1=begin_cell.x1, y1=begin_cell.y1,
                                          x2=next_cell.x2, y2=next_cell.y2,
                                          row_index=index, col_index=start_index,
                                          )
                        merge_cell.top = begin_cell.top
                        merge_cell.bottom = next_cell.bottom
                        merge_cell.left = True if begin_cell.left and next_cell.left else False
                        merge_cell.right = True if begin_cell.right and next_cell.right else False
                        merge_cell.inner_cells.extend(begin_cell.inner_cells)
                        merge_cell.inner_cells.append(next_cell)

                        # 需要删除的点
                        for cell_remove in [begin_cell, next_cell]:
                            if not cell_equal_in_list(cell_remove, remove_cells) \
                                    and cell_equal_in_list(cell_remove, need_merge_cells):
                                remove_cells.append(cell_remove)
                        begin_cell = merge_cell
                    else:
                        begin_cell = next_cell

                    # point_equal_in(next_cell.lt, sorted_joint_point) \
                    #                             and point_equal_in(next_cell.rt, sorted_joint_point) \
                    #                             and
                    if begin_cell.top and begin_cell.bottom:
                        new_merge_cells.append(begin_cell)
                        if start_index < len(need_merge_cells) - 1:
                            begin_cell = need_merge_cells[start_index + 1]

            if len(new_merge_cells) > 0:
                new_cells2.append(new_merge_cells)
        logger.info(f"row_merge_cells: {len(new_cells2)} -{new_cells2}")

        return new_cells2, remove_cells

    def find_cell_joint_point_list(self, find_joint_point: List, new_cells: List, begin_x) -> List:
        """
        根据 关键点 找到和其重叠的CELL

        :param find_joint_point:
        :param new_cells:
        :param begin_x:
        :return:
        """
        # 找到目前的cell
        find_cells = []
        for row_index, row_cells in enumerate(new_cells):
            for col_index, cell in enumerate(row_cells):
                if self.cell_in_point_list(cell, find_joint_point) or equal(cell.x1, begin_x):
                    find_cells.append(cell)

        if begin_x is not None:
            find_cells = [cell for cell in find_cells if equal(cell.x1, begin_x)]

        return find_cells

    def cell_in_point_list(self, cell, find_joint_point: List) -> bool:
        flag = False
        for point in find_joint_point:
            if cell_equal_in(point, cell):
                flag = True
                break
        return flag

    def get_row_need_merge_cells(self, find_cells: List[Cell]):
        """
        判断cell 的顶部和底部是否存在边

        :param find_cells:
        :return:
        """
        need_merge_cells = []
        for index, cell in enumerate(find_cells):
            if cell is not None and (not cell.top or not cell.bottom):
                need_merge_cells.append(cell)
        return need_merge_cells

    def get_merge_cell(self, column_merge_cells, row_merge_cells, remove_cells) -> List[Cell]:
        """
        汇总 行和列 的 cell

        :param column_merge_cells:
        :param row_merge_cells:
        :param remove_cells:
        :return:
        """
        all_cells: List[Cell] = []
        for row_cells in column_merge_cells:
            for col_cell in row_cells:
                if not cell_equal_in_list(col_cell, remove_cells) \
                        and not cell_equal_in_list(col_cell, all_cells):
                    all_cells.append(col_cell)
        for row_cells in row_merge_cells:
            for col_cell in row_cells:
                if not cell_equal_in_list(col_cell, remove_cells) \
                        and not cell_equal_in_list(col_cell, all_cells):
                    all_cells.append(col_cell)
        logger.info(f"all_cells: {len(all_cells)} - {all_cells}")

        return all_cells

    def show_merge_cell(self, table_idx, all_cell_results: List[Cell],
                        image_scalers,
                        sep_file_name="_table_cell_"):
        """
        显示表格的网格图片

        :param table_idx:
        :param all_cell_results:
        :param image_scalers:
        :param sep_file_name:
        :return:
        """
        src_im = copy.deepcopy(self.image)

        thickness = 2

        for index, cell in enumerate(all_cell_results):
            x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x1, cell.y1, cell.x2, cell.y2), image_scalers)

            start_point = (x1, y1)
            end_point = (x2, y2)

            color = self.color_list[cell.row_index % len(self.color_list)]

            cv2.rectangle(src_im, start_point, end_point, color, thickness)

        save_image_file = f"{self.output_dir}/{self.raw_filename}{sep_file_name}{table_idx + 1}.jpg"
        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, src_im)

        logger.info(f"保存表格cell图像：{len(all_cell_results)} - {save_image_file}")
        return save_image_file

    def show_cell_line(self, table_idx, cells: List[List[Cell]], image_scalers, sep_file_name="_table_cell_line_"):
        """
        显示表格的网格图片

        :param table_idx:
        :param cells:
        :param image_scalers:
        :param sep_file_name:
        :return:
        """
        src_im = copy.deepcopy(self.image)

        thickness = 2

        all_cells = []
        for cell_list in cells:
            all_cells.extend(cell_list)

        for index, cell in enumerate(all_cells):
            x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x1, cell.y1, cell.x2, cell.y2), image_scalers)

            lines = []
            if cell.top:
                lines.append([(x1, y1), (x2, y1)])
            if cell.bottom:
                lines.append([(x1, y2), (x2, y2)])
            if cell.left:
                lines.append([(x1, y1), (x1, y2)])
            if cell.right:
                lines.append([(x2, y1), (x2, y2)])

            color = self.color_list[cell.row_index % len(self.color_list)]

            for line in lines:
                # cv2.rectangle(src_im, start_point, end_point, color, thickness)
                cv2.line(src_im, line[0], line[1], color, thickness)

        save_image_file = f"{self.output_dir}/{self.raw_filename}{sep_file_name}{table_idx + 1}.jpg"
        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, src_im)

        logger.info(f"保存表格cell图像：{len(all_cells)} - {save_image_file}")
        return save_image_file

    def match_table_cell_and_text_cell(self, table_idx, table_cells: List[Cell], text_cells):
        """
        匹配 table cell 和 text cell

        :param table_idx:
        :param table_cells:
        :param text_cells:
        :return:
        """
        results = table_cells

        logger.info(f"match text cell results: {len(results)} -> ")
        results.sort(key=lambda x: (x.row_index, x.col_index))

        table_html, db_table_html = TableProcessUtils.cell_to_html(table_cells=results)
        table_html_str = "\n".join(table_html) + "\n"

        save_html_file = f"{self.output_dir}/{self.raw_filename}_table_{table_idx + 1}.html"
        FileUtils.save_to_text(save_html_file, table_html_str)

        save_db_html_file = save_html_file.replace(".html", "_db.html")
        FileUtils.save_to_text(save_db_html_file, "\n".join(db_table_html) + "\n")

        logger.info(f"save table cell html：{save_html_file}")

        metric = {
            "table_idx": table_idx,
            "save_html_file": save_html_file,
            "save_db_html_file": save_db_html_file
        }

        return results, table_html, metric, db_table_html

    def cell_line_mark_check(self, cells: List[List[Cell]], line_radio=0.7):
        """
        cell的边框修正（虚线，有点空白的线）

        :param cells:
        :param line_radio:
        :return:
        """

        total_row = len(cells)
        total_col = len(cells[0])

        height = round(self.avg_horizontal_height * 1)
        width = round(self.avg_vertical_width * 1)

        for index, row_cells in enumerate(cells):
            for col_index, cell in enumerate(row_cells):

                cell_scale = self.scale_cell_to_image(cell)

                x1 = cell_scale.x1_round
                x2 = cell_scale.x2_round
                y1 = cell_scale.y1_round
                y2 = cell_scale.y2_round

                if not cell.top:
                    roi_image = copy.deepcopy(self.image[y1 - height:y1 + height, x1:x2])
                    flag = PdfImageProcessor.find_cell_line_exist(image=roi_image, line_radio=line_radio,
                                                                  regions=self.regions)
                    if flag:
                        cell.top = True
                        logger.info(f"cell的边框修正,顶部修正：[{index},{col_index}] - {cell_scale}")
                if not cell.bottom:
                    roi_image = copy.deepcopy(self.image[y2 - height:y2 + height, x1:x2])
                    flag = PdfImageProcessor.find_cell_line_exist(image=roi_image, line_radio=line_radio,
                                                                  regions=self.regions)
                    if flag:
                        cell.bottom = True
                        logger.info(f"cell的边框修正,底部修正：[{index},{col_index}] - {cell_scale}")

                if not cell.left:
                    roi_image = copy.deepcopy(self.image[y1:y2, x1 - width:x1 + width])
                    flag = PdfImageProcessor.find_cell_line_exist(image=roi_image,
                                                                  is_horizontal=False,
                                                                  line_radio=line_radio,
                                                                  regions=self.regions)
                    if flag:
                        cell.left = True
                        logger.info(f"cell的边框修正,左边修正：[{index},{col_index}] - {cell_scale}")
                if not cell.right:
                    roi_image = copy.deepcopy(self.image[y1:y2, x2 - width:x2 + width])
                    flag = PdfImageProcessor.find_cell_line_exist(image=roi_image,
                                                                  is_horizontal=False,
                                                                  line_radio=line_radio,
                                                                  regions=self.regions)
                    if flag:
                        cell.right = True
                        logger.info(f"cell的边框修正,右边修正：[{index},{col_index}] - {cell_scale}")

    def scale_table_bbox_to_pdf(self, table_bbox):
        image_scalers, pdf_scalers = TableProcessUtils.get_pdf_scaler(image_shape=[self.image_height,
                                                                                   self.image_width],
                                                                      pdf_shape=[self.pdf_height,
                                                                                 self.pdf_width])

        # bbox 是 lt,rb 而 pdf 上的box 是lb,rt
        table_bbox_new = [table_bbox[0], table_bbox[3], table_bbox[2], table_bbox[1]]
        bbox = MathUtils.scale_image_bbox(bbox=table_bbox_new, factors=pdf_scalers)
        return bbox

    def scale_image_bbox_to_image(self, ):
        image_scalers, pdf_scalers = TableProcessUtils.get_pdf_scaler(image_shape=[self.image_height,
                                                                                   self.image_width],
                                                                      pdf_shape=[self.pdf_height,
                                                                                 self.pdf_width])

        for image in self.pdf_images:
            table_bbox = image["bbox"]
            table_bbox_new = [table_bbox[0], table_bbox[3], table_bbox[2], table_bbox[1]]
            bbox = MathUtils.scale_image_bbox(bbox=table_bbox_new, factors=image_scalers)
            bbox = [int(item) for item in bbox]
            image["image_bbox"] = bbox

            bbox_joint_point = [
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
            ]

            image["image_bbox_joint_point"] = bbox_joint_point

    def filter_joint_point_match_image(self, joint_point: List):
        """
        过滤交点 是图片的边缘交点

        :param joint_point:
        :return:
        """
        image_bboxs = []
        for image in self.pdf_images:
            bbox = image["image_bbox"]
            image_bboxs.append(bbox)

        new_joint_points = []
        for point in joint_point:
            new_point = MathUtils.scale_point(point, self.image_scalers)

            if not point_in_bbox_list(new_point, image_bboxs, diff=2):
                new_joint_points.append(point)

        return new_joint_points

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

    file_dir = f"{Constants.PDF_CACHE_BASE}/ocr_file/temp_dir"

    file_name = f"{file_dir}/page-27.pdf"
    # file_name = f"{output_dir}/pdf/page-8.pdf"
    # file_name = f"{file_dir}/page-27.png"
    # file_name = (f"{Constants.HTML_BASE_DIR}/pdf_html/pdf_check/main/2023-08-14/578/20230814_162337/page-9.pdf")

    page = 1
    parser = TableCellExtract(output_dir=output_dir, page=page, debug=True)

    parser.extract_cell(file_name)


if __name__ == '__main__':
    main()
