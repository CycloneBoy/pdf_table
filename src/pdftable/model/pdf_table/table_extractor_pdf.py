#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_extractor_pdf
# @Author  ：cycloneboy
# @Date    ：20xx/6/2 14:06
import copy
import os
import time
import traceback
from collections import defaultdict
from typing import List, Dict, cast

import cv2
import pandas as pd
from pdfminer.layout import LTTextLineHorizontal, LTChar, LTAnno, LTTextBoxVertical, LTText

from pdftable.utils.table.image_processing import PdfImageProcessor
from .table_common import (
    equal, equal_point, cell_equal_in_list,
    point_equal_in, cell_equal_in, distance, box_in_other_box,
    cell_horizontal_line_exists, cell_vertical_line_exists,
    equal_in, compute_iou_v2
)
from .table_core import Table, Cell, TableList
from .table_extractor_base import TableExtractorBase
from .table_result_compare import TableResultCompare
from ...entity import PdfTextCell
from ...utils import (
    logger,
    Constants,
    MathUtils,
    FileUtils,
    TimeUtils,
    PdfUtils,
    CommonUtils,
)

__all__ = [
    'TableExtractorPdf'
]

"""
提取PDF 中的表格
    - 有线表格提取
    - 表格合并
    - 优化：表格文字匹配错误
    - 优化：表格横竖排文字对齐
    - TODO: 跨页表格合并
    - 
"""


class TableExtractorPdf(TableExtractorBase):

    def __init__(
            self,
            table_regions=None,
            table_areas=None,
            process_background=False,
            line_scale=40,
            line_scale_vertical=50,
            copy_text=None,
            shift_text=["l", "t"],
            split_text=False,
            flag_size=False,
            strip_text="",
            line_tol=2,
            joint_tol=2,
            line_mark_tol=6,
            threshold_block_size=15,
            threshold_constant=-2,
            iterations=0,
            resolution=300,
            # backend="poppler_v2",
            # backend="poppler",
            backend="ghostscript",
            output_dir=None,
            filepath=None,
            debug=True,
            src_id=None,
            delete_check_success=True,
            update_version="v0.3",
            **kwargs,
    ):
        super().__init__(debug=debug)
        self.table_regions = table_regions
        self.table_areas = table_areas
        self.process_background = process_background
        self.line_scale = line_scale
        self.line_scale_vertical = line_scale_vertical
        self.copy_text = copy_text
        self.shift_text = shift_text
        self.split_text = split_text
        self.flag_size = flag_size
        self.strip_text = strip_text
        self.line_tol = line_tol
        self.line_mark_tol = line_mark_tol
        self.joint_tol = joint_tol
        self.threshold_block_size = threshold_block_size
        self.threshold_constant = threshold_constant
        self.iterations = iterations
        self.resolution = resolution

        self.output_dir = output_dir
        self.backend = self._get_pdf_to_image_backend(backend)
        self.filepath = filepath

        self.src_id = src_id
        self.delete_check_success = delete_check_success
        self.update_version = update_version

        self.result_http_server = self.get_result_http_server()

        self.check_table_html_compare = TableResultCompare()

    def get_result_http_server(self):
        result_http_server = "http://localhost:9100"

        prefix = f"{Constants.PDF_CACHE_BASE}/table_file/inference_results/"
        if self.output_dir is not None and str(self.output_dir).startswith(prefix):
            end = os.path.dirname(self.output_dir)[len(prefix):]
            result_http_server = f"{result_http_server}/pdf_html/{end}"

        return result_http_server

    def _generate_table_bbox(self):
        def scale_areas(areas):
            scaled_areas = []
            for area in areas:
                x1, y1, x2, y2 = area.split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                x1, y1, x2, y2 = MathUtils.scale_pdf((x1, y1, x2, y2), image_scalers)
                scaled_areas.append((x1, y1, abs(x2 - x1), abs(y2 - y1)))
            return scaled_areas

        self.image, self.threshold = PdfImageProcessor.adaptive_threshold(
            self.imagename,
            process_background=self.process_background,
            blocksize=self.threshold_block_size,
            c=self.threshold_constant,
        )

        image_width = self.image.shape[1]
        image_height = self.image.shape[0]
        image_width_scaler = image_width / float(self.pdf_width)
        image_height_scaler = image_height / float(self.pdf_height)
        pdf_width_scaler = self.pdf_width / float(image_width)
        pdf_height_scaler = self.pdf_height / float(image_height)
        image_scalers = (image_width_scaler, image_height_scaler, self.pdf_height)
        pdf_scalers = (pdf_width_scaler, pdf_height_scaler, image_height)

        self.image_scalers = image_scalers
        self.pdf_scalers = pdf_scalers

        if self.table_areas is None:
            regions = None
            if self.table_regions is not None:
                regions = scale_areas(self.table_regions)

            vertical_mask, vertical_segments = PdfImageProcessor.find_lines(
                self.threshold,
                regions=regions,
                direction="vertical",
                line_scale=self.line_scale_vertical,
                iterations=self.iterations,
            )
            horizontal_mask, horizontal_segments = PdfImageProcessor.find_lines(
                self.threshold,
                regions=regions,
                direction="horizontal",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )

            line_mask = vertical_mask + horizontal_mask

            if self.debug:
                save_image_file = f"{self.output_dir}/{self.raw_filename}_line_mask.png"
                FileUtils.check_file_exists(save_image_file)
                cv2.imwrite(save_image_file, line_mask)
                logger.info(f"保存line_mask：{save_image_file}")

            contours = PdfImageProcessor.find_contours(vertical_mask, horizontal_mask)
            table_bbox = PdfImageProcessor.find_joints(contours, vertical_mask, horizontal_mask)
        else:
            vertical_mask, vertical_segments = PdfImageProcessor.find_lines(
                self.threshold,
                direction="vertical",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )
            horizontal_mask, horizontal_segments = PdfImageProcessor.find_lines(
                self.threshold,
                direction="horizontal",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )

            areas = scale_areas(self.table_areas)
            table_bbox = PdfImageProcessor.find_joints(areas, vertical_mask, horizontal_mask)

        table_bbox = self.table_bbox_merge(table_bbox, diff=10)
        self.table_bbox_unscaled = copy.deepcopy(table_bbox)

        # 保存表格交点图像
        if self.debug:
            self.save_table_join_point(table_bbox_unscaled=self.table_bbox_unscaled)

        # 补充边框
        # for key, val in table_bbox.items():
        #     x1, y1, x2, y2 = val[0], val[1], val[2], val[3]
        #     horizontal_segments.append([x1, y1, x2, y1])
        #     horizontal_segments.append([x1, y2, x2, y2])
        #     vertical_segments.append([x1, y1, x1, y2])
        #     vertical_segments.append([x2, y1, x2, y2])

        self.table_bbox, self.vertical_segments, self.horizontal_segments = MathUtils.scale_image(
            table_bbox, vertical_segments, horizontal_segments, pdf_scalers
        )

    def save_table_join_point(self, table_bbox_unscaled):
        """
        保存表格交点图像
        :return:
        """
        src_im = copy.deepcopy(self.image)

        thickness = 2
        color1 = (255, 255, 0)
        color2 = (255, 0, 255)

        for key, val in table_bbox_unscaled.items():
            point = [int(item) for item in key]
            start_point = (point[0], point[1])
            end_point = (point[2], point[3])
            # logger.info(f"start_point {start_point} - {end_point}")
            cv2.rectangle(src_im, start_point, end_point, color1, thickness)
            #     cv2.circle(src_im, start_point, 10, color2, thickness)
            #     cv2.circle(src_im, end_point, 10, color2, thickness)

            for index, point in enumerate(val):
                cv2.circle(src_im, point, 20, color2, thickness)

        save_image_file = f"{self.output_dir}/{self.raw_filename}_table_joint_point.png"
        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, src_im)

        logger.info(f"保存表格交点图像：{save_image_file}")

    def extract_tables(self, filename, layout_kwargs=None, **kwargs):
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()
        self._generate_layout(filename, layout_kwargs)
        if self.debug:
            logger.info(f"Processing extract_tables : {self.basename}")

        if not self.horizontal_text:
            if self.images:
                logger.warn(f"{self.basename} is image-based, only works on text-based pages.")
            else:
                logger.warn("No tables found on {}".format(self.basename))
            return []

        # PDF TO PNG
        self.backend.convert(self.filename, self.imagename)

        self._generate_table_bbox()

        all_table_cell = []
        all_table_html = []
        all_db_table_html = []

        run_time = FileUtils.get_file_name(f"{self.output_dir}.txt")

        metric = {
            "run_time": run_time,
            "file_name": filename,
            "image_name": self.imagename,
            "image_scalers": self.image_scalers,
            "pdf_scalers": self.pdf_scalers,
            "result_dir_url": f"{self.result_http_server}/{run_time}",
            "show_url": f"{self.result_http_server}/{run_time}/{self.raw_filename}_show.html",
            "line_mask_image": f"{self.output_dir}/{self.raw_filename}_line_mask.png",
            "table_joint_point_image": f"{self.output_dir}/{self.raw_filename}_table_joint_point.png",
            "table_metric": []
        }

        _tables = []
        # sort tables based on y-coord
        for table_idx, tk in enumerate(
                sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=True)
        ):
            try:
                one_metric = {}
                cols, rows, v_s, h_s = self._generate_columns_and_rows(table_idx, tk)
                table = self._generate_table(table_idx, cols, rows, v_s=v_s, h_s=h_s)
                table._bbox = tk

                text_bbox, all_cell_results, table_cell_metric = self.generate_table_cell(table_idx, tk)

                # 过滤单一的表格
                if table_idx == 0 and len(all_cell_results) == 1:
                    logger.info(f"过滤只有一行和一列的表格：{table_idx} - {text_bbox}")
                    table_cell_metric.update(metric)
                    self.delete_not_found_table(table_cell_metric)
                    continue

                table_cells, table_html, match_metric, db_table_html = self.match_table_cell_and_text_cell(
                    table_idx=table_idx,
                    table_cells=all_cell_results,
                    text_cells=text_bbox)

                all_table_cell.append(table_cells)
                all_table_html.append(table_html)
                all_db_table_html.append(db_table_html)

                _tables.append(table)

                one_metric.update(table_cell_metric)
                one_metric.update(match_metric)
                metric["table_metric"].append(one_metric)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"提取表格异常：{self.src_id} - {table_idx} - {e}")

        end_time = time.time()
        use_time = end_time - begin_time

        end_time_str = TimeUtils.now_str()
        use_time_str = TimeUtils.calc_diff_time(begin_time_str, end_time_str)
        metric["begin_time"] = begin_time_str
        metric["end_time"] = end_time_str
        metric["use_time"] = use_time_str
        metric["use_time2"] = use_time

        all_valid_check = False
        try:
            all_valid_check = self.generate_table_extract_result(tables=_tables,
                                                                 all_table_cell=all_table_cell,
                                                                 all_table_html=all_table_html,
                                                                 all_db_table_html=all_db_table_html,
                                                                 metric=metric)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"产生表格结果异常：{self.src_id} - {e}")

        if len(self.table_bbox) == 0:
            self.delete_not_found_table(metric)

        # 删除PDF ,PNG
        all_valid_check = True
        if self.delete_check_success and all_valid_check:
            start_with = FileUtils.get_file_name(metric["image_name"])
            start_with1 = f"{start_with}."
            start_with2 = f"{start_with}_"
            raw_file_list1 = FileUtils.list_file_prefix(file_dir=self.output_dir, add_parent=True,
                                                        start_with=start_with1)
            raw_file_list2 = FileUtils.list_file_prefix(file_dir=self.output_dir, add_parent=True,
                                                        start_with=start_with2)
            raw_file_list = []
            raw_file_list.extend(raw_file_list1)
            raw_file_list.extend(raw_file_list2)
            file_list = [name for name in raw_file_list if name.endswith(".png") or name.endswith(".pdf")]
            FileUtils.delete_file_list(file_list, show_log=False)
            logger.info(f"所有表格提取正确：delete file_list: {len(file_list)}")

        return _tables

    def delete_not_found_table(self, metric):
        """
        删除 没有发现表格的文件

        :param metric:
        :return:
        """
        file_list = [
            metric["file_name"],
            metric["image_name"],
            metric["line_mask_image"],
            metric["table_joint_point_image"],
        ]
        FileUtils.delete_file_list(file_list, show_log=False)

    def generate_table_extract_result(self, tables, all_table_cell, all_table_html, all_db_table_html, metric):
        """
        生成最终 识别的结果
                - 对比结果
                - metric
                - 数据库

        :param tables:
        :param all_table_cell:
        :param all_table_html:
        :param all_db_table_html:
        :param metric:
        :return:
        """
        run_name = FileUtils.get_file_name(f"{self.output_dir}.txt")

        table_list = TableList(sorted(tables))
        if len(tables) == 0:
            logger.info(f"没有提取到表格：{self.raw_filename}")
            return

        out_file = f"{self.output_dir}/{FileUtils.get_file_name(self.filepath)}"
        table_list.export(f'{out_file}.html', f='html', compress=False)
        table_list.export(f'{out_file}.json', f='json', compress=False)
        table_list.export(f'{out_file}.xlsx', f='excel', compress=False)

        table_metric = metric["table_metric"]

        html_file = os.path.join(self.output_dir, f"{self.raw_filename}_show.html")
        logger.info(f"html show file : {html_file}")

        metric_json_file = os.path.join(self.output_dir, f"{self.raw_filename}_metric.json")
        FileUtils.dump_json(metric_json_file, metric)
        logger.info(f"metric json file : {metric_json_file}")

        html_content = [
            "<html>",
            "<body>",
            '<table border="1">',
            "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />",
            "<tr>",
            "<td>img name",
            "<td>ori image</td><td>table html</td><td>cell box</td>",
            "</tr>"
        ]
        self.write_html(html_file, html_content)

        save_table_html = []
        all_check_flag = []
        all_check_result = []
        for index, table in enumerate(tables):
            one_metric = table_metric[index]
            file_name = os.path.basename(metric["file_name"])
            table_joint_point_image = os.path.basename(metric["table_joint_point_image"])
            line_mask_image = os.path.basename(metric["line_mask_image"])
            src_image_name = os.path.basename(metric["image_name"])

            table_cell_image = os.path.basename(one_metric["table_cell_image"])
            cell_text_image_file = os.path.basename(one_metric["cell_text_image_file"])

            table_html = all_table_html[index]
            pdf_pred_html = "\n".join(table_html) + "\n"
            save_table_html.append(pdf_pred_html)

            db_table_html = all_db_table_html[index]
            pdf_pred_db_html = "\n".join(db_table_html) + "\n"

            check_flag = False
            check_metric = {}

            true_db_html_table = ''
            check_type = check_metric.get('check_type', "")
            flag_result = "错误！！！"
            flag_color = "red"
            if check_flag:
                flag_result = "正确"
                flag_color = "blue"

            check_msg_html = f'<p style="color:{flag_color};">{flag_result} - {check_type}</p>'

            src_pred_html_filename = f"{out_file}-page-{table.page}-table-{table.order}.html"
            src_pred_html = FileUtils.read_to_text(src_pred_html_filename)

            html_content = [
                "<tr>",
                f'<tr><td colspan=4 align=center><p style="color:red;">'
                f'表格解析结果：page-{table.page}-table-{table.order}: {check_msg_html}</p></td></tr>',
                "</tr>",

                "<tr>",
                f'<td> {file_name} <br/>',
                f'<td><img src="{table_joint_point_image}" width=640></td>',
                f'<td>{pdf_pred_html}</td>',
                f'<td></td>',
                "</tr>",

                "<tr>",
                f'<td> {file_name} <br/>',
                f'<td><img src="{table_cell_image}" width=640></td>',
                f'<td>{src_pred_html}</td>',
                f'<td></td>',
                "</tr>",

                "<tr>",
                f'<td> {file_name} <br/>',
                f'<td><img src="{src_image_name}" width=640></td>',
                f'<td><img src="{line_mask_image}" width=640></td>',
                f'<td></td>',
                "</tr>",

                "<tr>",
                f'<td> {file_name} <br/>',
                f'<td>cell text </td>',
                f'<td><img src="{cell_text_image_file}" width=640></td>',
                f'<td></td>',
                "</tr>",

            ]

            self.write_html(html_file, html_content)

            all_check_flag.append(check_flag)

        html_content = [
            "</table>",
            "</body>",
            "</html>",
        ]
        self.write_html(html_file, html_content)

        metric["check_flag"] = all_check_flag

        logger.info(f"解析结果目录链接：{metric['result_dir_url']}")
        logger.info(f"解析结果显示链接：{metric['show_url']}")

        all_valid_check = True

        return all_valid_check

    def write_html(self, file_name, content):
        FileUtils.save_to_text(file_name, "\n".join(content) + "\n", mode='a')

    def _generate_columns_and_rows(self, table_idx, tk):
        # select elements which lie within table_bbox
        t_bbox = {}
        v_s, h_s = PdfUtils.segments_in_bbox(
            tk, self.vertical_segments, self.horizontal_segments
        )
        t_bbox["horizontal"] = PdfUtils.text_in_bbox(tk, self.horizontal_text)
        t_bbox["vertical"] = PdfUtils.text_in_bbox(tk, self.vertical_text)

        t_bbox["horizontal"].sort(key=lambda x: (-x.y0, x.x0))
        t_bbox["vertical"].sort(key=lambda x: (x.x0, -x.y0))

        self.t_bbox = t_bbox

        cols = []
        rows = []
        for x, y in self.table_bbox[tk]:
            cols.append(x)
            rows.append(y)

        # cols, rows = zip(*self.table_bbox[tk])
        # cols, rows = list(cols), list(rows)
        cols.extend([tk[0], tk[2]])
        rows.extend([tk[1], tk[3]])
        # sort horizontal and vertical segments
        cols = PdfUtils.merge_close_lines(sorted(cols), line_tol=self.line_tol)
        rows = PdfUtils.merge_close_lines(sorted(rows, reverse=True), line_tol=self.line_tol)
        # make grid using x and y coord of shortlisted rows and cols
        cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        rows = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1)]

        return cols, rows, v_s, h_s

    def generate_table_cell(self, table_idx, bbox, line_tol=4):
        """
        select elements which lie within table_bbox

        :param table_idx:
        :param bbox:
        :param line_tol:
        :return:
        """
        text_bbox, h_s, v_s = self.get_text_in_table_bbox(bbox)

        joint_point = self.table_bbox[bbox]
        cols = []
        rows = []
        for x, y in joint_point:
            cols.append(x)
            rows.append(y)

        # cols, rows = zip(*joint_point)
        # cols, rows = list(cols), list(rows)
        # 添加边框的两个点
        for x in [bbox[0], bbox[2]]:
            if not equal_in(x, same_list=cols, diff=line_tol):
                cols.append(x)

        for y in [bbox[1], bbox[3]]:
            if not equal_in(y, same_list=rows, diff=line_tol):
                rows.append(y)

        # cols.extend([tk[0], tk[2]])
        # rows.extend([tk[1], tk[3]])
        # sort horizontal and vertical segments
        cols = PdfUtils.merge_close_lines(sorted(cols), line_tol=line_tol)
        rows = PdfUtils.merge_close_lines(sorted(rows, reverse=True), line_tol=6)

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

        # 生成所有的cell m x n
        cells = [[Cell(c[0], r[0], c[1], r[1], row_index=row_index, col_index=col_index)
                  for col_index, c in enumerate(col_pairs)]
                 for row_index, r in enumerate(row_pairs)]

        # 添加四个角点
        joint_point_cornets = [(min_col, min_row), (max_col, min_row), (max_col, max_row), (min_col, max_row)]
        for index, point in enumerate(joint_point_cornets):
            if point not in new_joint_point:
                new_joint_point.append(point)
                logger.info(f"添加表格边框的四个角点：{index} - {point}")

        self.cell_line_mark(cells, h_s=h_s, v_s=v_s, diff=self.line_mark_tol)

        logger.info(f"col_pairs: {col_pairs}")
        logger.info(f"row_pairs: {row_pairs}")
        logger.info(f"cells: {len(cells)}")

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

        all_cell_results = self.modify_cell_info(all_cells,
                                                 cols=cols,
                                                 rows=rows,
                                                 one_col_width=one_col_width,
                                                 one_row_height=one_row_height)

        table_cell_image = self.show_merge_cell(table_idx=table_idx,
                                                all_cell_results=all_cell_results, image_scalers=self.image_scalers)

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

    def show_merge_cell(self, table_idx, all_cell_results: List[Cell], image_scalers):
        """
        显示表格的网格图片

        :param all_cell_results:
        :param image_scalers:
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

        save_image_file = f"{self.output_dir}/{self.raw_filename}_table_cell_{table_idx + 1}.png"
        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, src_im)

        logger.info(f"保存表格cell图像：{save_image_file}")
        return save_image_file

    def modify_cell_info(self, all_cells: List[Cell], cols, rows, one_col_width, one_row_height):
        """
        修改cell 的属性

        :param all_cells:
        :param cols:
        :param rows:
        :param one_col_width:
        :param one_row_height:
        :return:
        """
        col_map = {col: index + 1 for index, col in enumerate(cols)}

        # pdf 和image 的y 轴相反
        rows_reverse = copy.deepcopy(rows)
        # rows_reverse.reverse()
        row_map = {row: index + 1 for index, row in enumerate(rows_reverse)}

        new_all_cells = sorted(all_cells, key=lambda x: (-x.y1, x.x1))
        for cell in new_all_cells:
            start_col_index = col_map.get(cell.x1)
            start_row_index = row_map.get(cell.y1)

            end_col_index = col_map.get(cell.x2)
            end_row_index = row_map.get(cell.y2)

            cell.col_index = start_col_index
            cell.row_index = start_row_index
            cell.col_span = abs(end_col_index - start_col_index)
            cell.row_span = abs(end_row_index - start_row_index)

            cell.width_ratio = cell.width / one_col_width
            cell.height_ratio = cell.height / one_row_height
        logger.info(f"new_all_cells: {len(new_all_cells)}")

        return new_all_cells

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
        logger.info(f"all_cells: {len(all_cells)} ")

        return all_cells

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
        logger.info(f"row_merge_cells: {len(new_cells2)}")

        return new_cells2, remove_cells

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

        logger.info(f"column_merge_cells: {len(new_cells)}")
        return new_cells

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
        text_bbox["horizontal"] = PdfUtils.text_in_bbox(bbox, self.horizontal_text)
        text_bbox["vertical"] = PdfUtils.text_in_bbox(bbox, self.vertical_text)
        text_bbox["horizontal"].sort(key=lambda x: (-x.y0, x.x0))
        text_bbox["vertical"].sort(key=lambda x: (x.x0, -x.y0))
        return text_bbox, h_s, v_s

    def _generate_table(self, table_idx, cols, rows, **kwargs):
        v_s = kwargs.get("v_s")
        h_s = kwargs.get("h_s")
        if v_s is None or h_s is None:
            raise ValueError("No segments found on {}".format(self.rootname))

        table = Table(cols, rows)
        # set table edges to True using ver+hor lines
        table = table.set_edges(v_s, h_s, joint_tol=self.joint_tol)
        # set table border edges to True
        table = table.set_border()
        # set spanning cells to True
        table = table.set_span()

        pos_errors = []
        # TODO: have a single list in place of two directional ones?
        # sorted on x-coordinate based on reading order i.e. LTR or RTL
        for direction in ["vertical", "horizontal"]:
            for t in self.t_bbox[direction]:
                indices, error = PdfUtils.get_table_index(
                    table,
                    t,
                    direction,
                    split_text=self.split_text,
                    flag_size=self.flag_size,
                    strip_text=self.strip_text,
                )
                if indices[:2] != (-1, -1):
                    pos_errors.append(error)
                    indices = self._reduce_index(
                        table, indices, shift_text=self.shift_text
                    )
                    for r_idx, c_idx, text in indices:
                        table.cells[r_idx][c_idx].text = text
        accuracy = PdfUtils.compute_accuracy([[100, pos_errors]])

        if self.copy_text is not None:
            table = self._copy_spanning_text(table, copy_text=self.copy_text)

        data = table.data
        table.df = pd.DataFrame(data)
        table.shape = table.df.shape

        whitespace = PdfUtils.compute_whitespace(data)
        table.flavor = "lattice"
        table.accuracy = accuracy
        table.whitespace = whitespace
        table.order = table_idx + 1
        table.page = int(os.path.basename(self.rootname).replace("page-", ""))

        # for plotting
        _text = []
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.horizontal_text])
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.vertical_text])
        table._text = _text
        table._image = (self.image, self.table_bbox_unscaled)
        table._segments = (self.vertical_segments, self.horizontal_segments)
        table._textedges = None

        return table

    def match_table_cell_and_text_cell(self, table_idx, table_cells: List[Cell], text_cells):
        """
        匹配 table cell 和 text cell

        :param table_idx:
        :param table_cells:
        :param text_cells:
        :return:
        """
        # text box 拆分: 将一行文本box 跨多个cell的拆分
        new_text_cells = self.text_box_split_to_cell(table_cells=table_cells, text_cells=text_cells)

        text_cells_h = new_text_cells["horizontal"]
        text_cells_v = new_text_cells["vertical"]

        table_bboxs = []
        for index, cell in enumerate(table_cells):
            cell_box = [cell.x1, self.pdf_height - cell.y1, cell.x2, self.pdf_height - cell.y2]
            table_bboxs.append(cell_box)

        all_text_cells = []
        all_text_cells.extend(text_cells_h)
        all_text_cells.extend(text_cells_v)
        cell_text_image_file = self.show_text_cell(table_idx=table_idx,
                                                   # text_cells=[text_cells_h, text_cells_v],
                                                   text_cells=[text_cells["horizontal"], text_cells["vertical"]],
                                                   image_scalers=self.image_scalers)

        text_bboxs = []
        for index, item in enumerate(all_text_cells):
            cell_box = item.bbox
            new_cell_box = (item.x0, self.pdf_height - item.y1, item.x1, self.pdf_height - item.y0)
            text_bboxs.append(new_cell_box)

        matched = {}
        for index, text_box in enumerate(text_bboxs):
            text_cell = all_text_cells[index]
            top1_index = self.find_top1_mach_box(text_box=text_box, table_bboxs=table_bboxs)
            top1_bbox = table_cells[top1_index]
            # msg = f"{}"
            if top1_index not in matched.keys():
                matched[top1_index] = [index]
            else:
                matched[top1_index].append(index)

        logger.info(f"cell and text matched: {matched}")

        results = []
        for key, val in matched.items():
            table_cell = table_cells[key]
            match_text_cells = [all_text_cells[index] for index in val]
            text_show, match_text_cells = self.get_one_cell_text(match_text_cells)
            table_cell.text_item = match_text_cells

            table_cell.text = '\n'.join(text_show)
            results.append(table_cell)

        not_matched_cells = [item for index, item in enumerate(table_cells) if index not in matched.keys()]
        results.extend(not_matched_cells)
        logger.info(f"cell and text matched matched end: match total: {len(matched)} "
                    f"-not match total {not_matched_cells}")

        logger.info(f"match text cell results: {len(results)} -> ")
        results.sort(key=lambda x: (x.row_index, x.col_index))
        # for index, cell in enumerate(results):
        #     logger.info(f"{index} - {cell}")

        # msg_show = []
        # for index, cell in enumerate(results):
        #     x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x1, cell.y1, cell.x2, cell.y2), self.image_scalers)
        #     msg_show.append([(x1, y1), (x2, y2), cell.col_index, cell.row_index, cell.text, ])
        #
        # for item in msg_show:
        #     # msg = " ".join()
        #     print(str(item))

        table_html, db_table_html = self.cell_to_html(table_cells=results)
        table_html_str = "\n".join(table_html) + "\n"

        save_html_file = f"{self.output_dir}/{self.raw_filename}_table_{table_idx + 1}.html"
        FileUtils.save_to_text(save_html_file, table_html_str)

        save_db_html_file = save_html_file.replace(".html", "_db.html")
        FileUtils.save_to_text(save_db_html_file, "\n".join(db_table_html) + "\n")

        logger.info(f"保存table cell html：{save_html_file}")

        metric = {
            "table_idx": table_idx,
            "cell_text_image_file": cell_text_image_file,
            "save_html_file": save_html_file,
            "save_db_html_file": save_db_html_file
        }

        return results, table_html, metric, db_table_html

    def get_one_cell_text(self, match_text_cells: List[LTTextLineHorizontal]):
        """
            获取一个table cell 中的文字
                - 按照阅读顺序排序，合并一行

        :param match_text_cells:
        :return:
        """

        match_text_cells_height = [item.height for item in match_text_cells]
        mean_height = sum(match_text_cells_height) / len(match_text_cells_height) * 1.0
        line_tol = mean_height / 3

        match_text_cells_y1 = [round(item.y1) for item in match_text_cells]

        norm_y1_list = PdfUtils.merge_close_lines(sorted(match_text_cells_y1, reverse=True),
                                                  line_tol=line_tol)

        pdf_text_cells = []
        for item in match_text_cells:
            new_y1 = PdfImageProcessor.find_close_norm_x(item.y1, norm_list=norm_y1_list, atol=line_tol)
            one_cell = PdfTextCell(text_cell=item, y_index=new_y1)
            pdf_text_cells.append(one_cell)

        # 文字排序
        pdf_text_cells.sort(key=lambda x: (-x.y_index, x.x1))

        text_show = []
        for item in pdf_text_cells:
            # 垂直文字
            if isinstance(item, LTTextBoxVertical):
                raw_text = "\n".join(
                    cast(LTText, obj).get_text() for obj in item if isinstance(obj, LTText)
                )
            else:
                raw_text = item.get_text().strip("\n")

            text_show.append(raw_text)

        sort_match_text_cells = [item.text_cell for item in pdf_text_cells]
        return text_show, sort_match_text_cells

    def find_top1_mach_box(self, text_box, table_bboxs: List):
        """
        查找对应的 top1 距离最近的box

        :param text_box:
        :param table_bboxs:
        :return:
        """
        distances = []
        find_cell_in_index = -1
        for index, pred_box in enumerate(table_bboxs):
            # 文本框在table 的cell 内
            if box_in_other_box(pred_box, text_box):
                dis = (0, 0)
                find_cell_in_index = index
                break
            else:
                # compute iou and l1 distance
                dis = (distance(text_box, pred_box), 1. - compute_iou_v2(text_box, pred_box))
            distances.append(dis)

        if find_cell_in_index > -1:
            return find_cell_in_index

        sorted_distances = distances.copy()
        # select det box by iou and l1 distance
        sorted_distances = sorted(sorted_distances, key=lambda item: (item[1], item[0]))
        top1_index = distances.index(sorted_distances[0])
        return top1_index

    def cell_to_html(self, table_cells: List[Cell], first_header=True):
        """
        cell to html

        :param table_cells:
        :param first_header:
        :return:
        """
        row_dict_sorted = self.convert_table_cell_to_dict(table_cells)

        first_rows = row_dict_sorted[1]
        first_row_row_spans = [cell for cell in first_rows if cell.row_span > 1]
        first_row_texts = [cell for cell in first_rows if len(cell.text) == 0]
        if first_header and (len(first_row_row_spans) >= 1 or len(first_row_texts) >= 1):
            first_header = False

        if len(row_dict_sorted) < 2:
            first_header = False

        html_row_list = []
        for row_index, cols in row_dict_sorted.items():
            one_cols = ['<tr>']
            table_cell_token = "td"
            if first_header and row_index == 1:
                table_cell_token = "th"

            all_row_span = [cell.row_span for col_index, cell in enumerate(cols) if cell.row_span > 1]
            all_row_span_same = True
            for row_span in all_row_span:
                if row_span != all_row_span[0]:
                    all_row_span_same = False

            fix_row_span_same = False
            if len(all_row_span) == len(cols) and len(cols) > 0 and all_row_span_same:
                fix_row_span_same = True

            for col_index, cell in enumerate(cols):
                colspan = f'colspan="{cell.col_span}" ' if cell.col_span > 1 else ""
                rowspan = f'rowspan="{cell.row_span}" ' if cell.row_span > 1 else ""
                width = f'width="{round(cell.width_ratio * 100)}%"' if cell.width > 0 else ""

                if fix_row_span_same:
                    rowspan = ""

                texts = cell.text.replace("\n", "<br/>")
                one_cell = f'<{table_cell_token} {colspan}{rowspan}{width}>{texts}</{table_cell_token}>'
                one_cols.append(one_cell)
            one_cols.append('</tr>')
            html_row_list.append(one_cols)

        # show html
        table_html = [
            '<table border="1" border="1" >',
        ]

        body_begin = 0
        if first_header:
            header_list = ['<thead>']
            header_list.extend(html_row_list[0])
            header_list.append('</thead>')
            body_begin = 1
            table_html.extend(header_list)

        table_html.append('<tbody>')
        for rows in html_row_list[body_begin:]:
            table_html.extend(rows)

        table_html.append('</tbody>')
        table_html.append('</table>')

        # db_table_html html
        db_table_html = [
            "<table class='pdf-table' border='1' width='100%'>",
        ]
        for rows in html_row_list:
            if rows[0] == "<tr>":
                rows[0] = '<tr align="center">'
            new_rows = [row.replace("<th ", "<td ").replace("</th>", "</td>") for row in rows]
            one_row = ''.join(new_rows)
            db_table_html.append(one_row)
        db_table_html.append('</table>')

        return table_html, db_table_html

    def convert_table_cell_to_dict(self, table_cells: List[Cell]) -> Dict[str, List[Cell]]:
        """
            将所有的table cell 转换成行列形式

        :param table_cells:
        :return:
        """
        table_cells.sort(key=lambda x: (x.row_index, x.col_index))

        row_dict = defaultdict(list)
        for cell in table_cells:
            row_dict[cell.row_index].append(cell)

        row_dict_sorted = CommonUtils.sorted_dict(row_dict, key=lambda x: x[0], reverse=False)
        return row_dict_sorted

    def show_text_cell(self, table_idx, text_cells, image_scalers):
        """
        显示表格的网格图片

        :param table_idx:
        :param text_cells:
        :param image_scalers:
        :return:
        """
        save_image_file = f"{self.output_dir}/{self.raw_filename}_text_cell_{table_idx + 1}.png"
        try:
            src_im = copy.deepcopy(self.image)

            thickness = 2

            for index, cell_list in enumerate(text_cells):
                for cell_index, cell in enumerate(cell_list):
                    x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x0, cell.y0, cell.x1, cell.y1), image_scalers)

                    start_point = (x1, y1)
                    end_point = (x2, y2)

                    color = self.color_list[index]

                    cv2.rectangle(src_im, start_point, end_point, color, thickness)

            FileUtils.check_file_exists(save_image_file)
            cv2.imwrite(save_image_file, src_im)

            logger.info(f"保存text cell图像：{save_image_file}")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"显示text cell图像异常：{e}")

        return save_image_file

    def cell_line_mark(self, cells: List[Cell], h_s: List, v_s: List, diff=6):
        """
        根据线条 给每一个cell 设置四条边是否是实体线

        :param cells:
        :param h_s:
        :param v_s:
        :param diff:
        :return:
        """
        v_s.sort(key=lambda x: (x[1], x[0]))
        h_s.sort(key=lambda x: (x[0], -x[1]))

        total_row = len(cells)
        total_col = len(cells[0])
        for index, row_cells in enumerate(cells):
            for col_index, cell in enumerate(row_cells):
                self.one_cell_mark_line(cell=cell, h_s=h_s, v_s=v_s, diff=diff)
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

    def one_cell_mark_line(self, cell: Cell, h_s: List, v_s: List, diff=2, need_sort=False):
        """
        标记一个cell 的 边框

        :param cell:
        :param h_s:
        :param v_s:
        :param diff:
        :param need_sort:
        :return:
        """
        if need_sort:
            v_s.sort(key=lambda x: (x[1], x[0]))
            h_s.sort(key=lambda x: (x[0], -x[1]))

        line_top_exists, line_bottom_exists = cell_horizontal_line_exists(cell=cell,
                                                                          horizontal_line_list=h_s,
                                                                          diff=diff)
        if line_top_exists:
            cell.top = True
        if line_bottom_exists:
            cell.bottom = True
        line_left_exists, line_right_exists = cell_vertical_line_exists(cell=cell,
                                                                        vertical_line_list=v_s,
                                                                        diff=diff)
        if line_left_exists:
            cell.left = True
        if line_right_exists:
            cell.right = True

    def one_column_cells_merge(self, cells: List[Cell]):
        """
        一行的cell 进行合并

        :param cells:
        :return:
        """
        cells.sort(key=lambda x: (x.x1, x.y1))
        merge_flag = False

        cell_flags = []
        for index, cell in enumerate(cells):
            flag = False
            if index == 0:
                if cell.left and not cell.right:
                    flag = True
            elif index == len(cells) - 1:
                if not cell.left and cell.right:
                    flag = True

            else:
                if not cell.left and not cell.right:
                    flag = True
            cell_flags.append(flag)

        if sum(cell_flags) == len(cells):
            merge_flag = True

        return merge_flag

    def text_box_split_to_cell(self, table_cells: List[Cell], text_cells, diff=2):
        """
        text box 拆分: 将一行文本box 跨多个cell的拆分

        :param table_cells:
        :param text_cells:
        :param diff:
        :return:
        """
        text_cells_h: List[LTTextLineHorizontal] = text_cells["horizontal"]
        text_cells_v: List[LTTextLineHorizontal] = text_cells["vertical"]

        new_text_cells = {
            "vertical": text_cells_v,
        }

        row_dict_sorted = self.convert_table_cell_to_dict(table_cells)

        # for row_index, row_cells in row_dict_sorted.items():
        #     begin_y = row_cells[0].y1
        #     end_y = row_cells[0].y2
        #
        #     row_texts = [for text_cell in row_cells if ]

        # 水平的拆分
        # for index, row_cells in enumerate(table_cells):
        #
        #     for col_index, cell in enumerate(row_cells):

        new_text_cells_h = []
        for index, text_cell in enumerate(text_cells_h):
            new_cell_box = (text_cell.x0, text_cell.y1, text_cell.x1, text_cell.y0)

            cell_in, begin_cell, end_cell = self.find_text_box_belong_cell(text_cell=new_cell_box,
                                                                           table_cells=table_cells)

            if cell_in is not None:
                new_text_cells_h.append(text_cell)
                continue
            elif begin_cell is not None or end_cell is not None:
                split_text_cell_list = self.split_text_cell(text_cell=text_cell,
                                                            table_cells=table_cells,
                                                            diff=diff)
                new_text_cells_h.extend(split_text_cell_list)
            else:
                new_text_cells_h.append(text_cell)
                logger.info(
                    f"拆分text cell失败,不进行拆分，直接保留原始text cell：{index} - {text_cell.get_text()} - {text_cell}")

        # 不存在 cell in 则查找开头的cell
        new_text_cells["horizontal"] = new_text_cells_h

        return new_text_cells

    def split_text_cell(self, text_cell: LTTextLineHorizontal,
                        table_cells: List[Cell], diff=2):
        """
        拆分 一个text cell 到多个

        :param text_cell:
        :param table_cells:
        :param diff:
        :return:
        """
        find_table_cells = self.find_cell_cross_text_cell(text_cell=text_cell, table_cells=table_cells, diff=diff)
        # 多列拆分
        new_text_cells = [LTTextLineHorizontal(word_margin=text_cell.word_margin) for _ in find_table_cells]

        before_cell_index = -1
        for index_char, char_object in enumerate(text_cell._objs):
            if isinstance(char_object, LTChar):
                cell_index = self.find_char_belong_cell(char_object, find_table_cells=find_table_cells, diff=diff)
                if cell_index > -1:
                    new_text_cells[cell_index].add(char_object)
                    before_cell_index = cell_index
                else:
                    # 没有找到对应的cell 则沿用之前的cell_index
                    if before_cell_index > -1:
                        new_text_cells[before_cell_index].add(char_object)
                    elif before_cell_index == -1 and index_char == 0:
                        new_text_cells[0].add(char_object)
                        logger.info(
                            f"未找到对应的cell【强制赋给第一个cell】：{index_char} - {char_object} - find_table_cells：{find_table_cells}")
                    else:
                        logger.info(
                            f"未找到对应的cell：{index_char} - {char_object} - find_table_cells：{find_table_cells}")
            elif isinstance(char_object, LTAnno):
                pass
                # logger.info(f"不是字符： {index_char} - {char_object}")
            else:
                logger.info(f"不是字符： {index_char} - {char_object}")

        results = []
        for chat_cell in new_text_cells:
            if len(chat_cell) > 0 and len(chat_cell.get_text()) > 0:
                results.append(chat_cell)

        src_text = text_cell.get_text().replace("\n", "\\n")
        logger.info(f"拆分text cell: [{len(text_cell)} = {' + '.join([str(len(char_cell)) for char_cell in results])}] "
                    f"- 【{src_text}】 -> 【 {' + '.join([char_cell.get_text() for char_cell in results])} 】")

        return results

    def find_cell_cross_text_cell(self, text_cell: LTTextLineHorizontal, table_cells: List[Cell], diff=2):
        """
        查找 text cell 和 table cell 相交的cell

        :param text_cell:
        :param table_cells:
        :param diff:
        :return:
        """
        text_cell_y_min = min(text_cell.y0, text_cell.y1)
        text_cell_y_max = max(text_cell.y0, text_cell.y1)

        find_table_cells = []
        for cell in table_cells:
            if cell.min_y - diff < text_cell_y_min < text_cell_y_max < cell.max_y + diff:
                if cell.x1 < cell.x2 < text_cell.x0 - diff \
                        or text_cell.x1 - diff < cell.x1 < cell.x2:
                    continue

                find_table_cells.append(cell)

        logger.info(f"find_table_cells: {text_cell} -> {len(find_table_cells)} -> {find_table_cells}")
        find_table_cells.sort(key=lambda x: x.x1)
        return find_table_cells

    def find_char_belong_cell(self, char_object: LTChar, find_table_cells: List[Cell], diff=2):
        """
        查找 char 属于哪一个table cell

        :param char_object:
        :param find_table_cells:
        :param diff:
        :return:
        """
        find_index = -1
        for index, cell in enumerate(find_table_cells):
            if cell.x1 - diff < char_object.x0 <= char_object.x1 < cell.x2 + diff:
                find_index = index
                break

            if cell.x1 - diff < char_object.x0 < cell.x2 <= char_object.x1:
                left_diff = cell.x2 - char_object.x0
                right_diff = char_object.x1 - cell.x2
                left_radio = left_diff / char_object.width

                if left_radio > 0.3:
                    find_index = index
                    break

        return find_index

    def find_text_box_belong_cell(self, text_cell, table_cells: List[Cell], diff=2):
        """
            查找 text box belong cell

        :param text_cell:
        :param table_cells:
        :param diff:
        :return:
        """

        cell_in = None
        begin_cell = None
        end_cell = None

        x0, y0, x1, y1 = text_cell
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        for index, cell in enumerate(table_cells):

            # 同一行：
            if cell.min_y - diff < min_y < max_y < cell.max_y + diff:
                if cell.x1 - diff < x0 < x1 < cell.x2 + diff:
                    cell_in = cell
                    break
                elif cell.x1 - diff < x0 < x1:
                    begin_cell = cell
                elif cell.x1 - diff < x1 < cell.x2 + diff:
                    end_cell = cell

        return cell_in, begin_cell, end_cell

    def table_bbox_merge(self, table_bboxs: Dict, diff=10):
        """
        表格合并
        :param table_bboxs:
        :param diff:
        :return:
        """
        if len(table_bboxs) < 2:
            return table_bboxs

        merge_bbox = {}
        # sort tables based on y-coord
        table_bbox_key = sorted(table_bboxs.keys(), key=lambda x: x[1], reverse=True)

        current_box = table_bbox_key[0]
        current_joint_coords = table_bboxs[current_box]
        for table_idx in range(1, len(table_bbox_key)):
            next_box = table_bbox_key[table_idx]
            next_joint_coords = table_bboxs[next_box]

            x1, y1, x2, y2 = current_box
            x3, y3, x4, y4 = next_box

            width1 = x2 - x1
            height1 = abs(y2 - y1)

            width2 = x4 - x3
            height2 = abs(y4 - y3)

            # 可以合并
            if equal_point((x1, y2), (x3, y3), diff=diff) \
                    and equal_point((x2, y2), (x4, y3), diff=diff):
                current_box = (x1, y1, x4, y4)

                merge_joint_coords = copy.deepcopy(current_joint_coords)
                for point in next_joint_coords:
                    if not point_equal_in(point, current_joint_coords, diff=diff):
                        merge_joint_coords.append(point)

                logger.info(
                    f"合并交点：{len(merge_joint_coords)} -> {len(current_joint_coords)} + {len(next_joint_coords)}")
                current_joint_coords = merge_joint_coords

            else:
                merge_bbox[current_box] = current_joint_coords
                current_box = next_box
                current_joint_coords = next_joint_coords

            if table_idx == len(table_bbox_key) - 1:
                merge_bbox[current_box] = current_joint_coords

        logger.info(f"合并表格：{len(merge_bbox)} - 原始： {len(table_bboxs)}")
        return merge_bbox

    def check_is_imaged_pdf(self, filename, layout_kwargs=None, **kwargs):
        """
        检测一个pdf 是否是图片型pdf

        :param filename:
        :param layout_kwargs:
        :param kwargs:
        :return:
        """
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()
        self._generate_layout(filename, layout_kwargs)
        # if self.debug:
        #     logger.info(f"Processing extract_tables : {self.basename}")

        is_imaged_pdf = False
        if not self.horizontal_text:
            if self.images:
                logger.warning(f"{self.basename} is image-based, only works on text-based pages.")
                is_imaged_pdf = True
            else:
                logger.warning("No tables found on {}".format(self.basename))

        return is_imaged_pdf


def demo_run_extract_table():
    output_dir = FileUtils.get_output_dir_with_time()

    file_dir = f"{Constants.PDF_CACHE_BASE}/ocr_file/temp_dir"

    file_name = f"{file_dir}/page-27.pdf"

    parser = TableExtractorPdf(output_dir=output_dir)
    tables = parser.extract_tables(file_name)

    logger.info(f"tables: {tables}")


if __name__ == '__main__':
    demo_run_extract_table()
