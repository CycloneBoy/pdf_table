#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_cell_extract_algo
# @Author  ：cycloneboy
# @Date    ：20xx/12/28 14:44

from typing import List, Dict

from pdfminer.layout import LTRect

from pdftable.entity.table_entity import Line
from pdftable.model.pdf_table.table_common import (equal_in,
                                                   calc_cell_width,
                                                   TableProcessUtils, point_in_bbox_list,
                                                   cell_horizontal_line_exists_v2, cell_vertical_line_exists_v2,
                                                   cell_equal_in_list, equal_point, equal, cell_equal_in
                                                   )
from pdftable.model.pdf_table.table_core import Cell
from pdftable.utils import (
    logger,
    PdfUtils, FileUtils,
)
from pdftable.utils.table.image_processing import PdfImageProcessor

__all__ = [
    "TableCellExtractAlgo"
]


class TableCellExtractAlgo(object):

    def __init__(
            self,
            line_tol=2,
            output_dir=None,
            debug=False,
            **kwargs,
    ):
        self.line_tol = line_tol
        self.output_dir = output_dir
        self.debug = debug

    def get_text_in_table_bbox_v2(self, bbox, v_segments: List[Line], h_segments: List[Line]):
        """
        筛选 在表格区域中的文本,水平线段，垂直线段

        :return:
        """
        text_bbox = {}

        v_s, h_s = PdfUtils.segments_in_bbox_v2(
            bbox, v_segments, h_segments
        )
        return text_bbox, h_s, v_s

    def filter_joint_point_match_image(self, joint_point: List, pdf_images: List):
        """
        过滤交点 是图片的边缘交点

        :param joint_point:
        :return:
        """
        image_bboxs = []
        for image in pdf_images:
            bbox = image["image_bbox"] if "image_bbox" in image else image["bbox"]
            image_bboxs.append(bbox)

        new_joint_points = []
        for point in joint_point:
            if not point_in_bbox_list(point, image_bboxs, diff=2):
                new_joint_points.append(point)

        logger.info(f"filter join point by image: {len(joint_point)} - {len(new_joint_points)} "
                    f"-filter total：{len(new_joint_points) - len(joint_point)} point.")

        return new_joint_points

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
        total_col = len(cells[0]) if len(cells) > 0 else 0
        for index, row_cells in enumerate(cells):
            for col_index, cell in enumerate(row_cells):
                self.one_cell_mark_line_v2(cell=cell, h_s=h_s, v_s=v_s, diff=diff,
                                           col_avg_width=col_avg_width,
                                           row_avg_width=row_avg_width)
                if index == 0 and not cell.top:
                    cell.top = True
                elif index == total_row - 1 and not cell.bottom:
                    cell.bottom = True

                if col_index == 0 and not cell.left:
                    cell.left = True
                elif col_index == total_col - 1 and not cell.right:
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

        line_top_exists, line_bottom_exists = cell_horizontal_line_exists_v2(cell=cell,
                                                                             horizontal_line_list=h_s,
                                                                             diff=diff,
                                                                             row_avg_width=row_avg_width)
        if line_top_exists:
            cell.top = True
        if line_bottom_exists:
            cell.bottom = True
        line_left_exists, line_right_exists = cell_vertical_line_exists_v2(cell=cell,
                                                                           vertical_line_list=v_s,
                                                                           diff=diff,
                                                                           col_avg_width=col_avg_width, )
        if line_left_exists:
            cell.left = True
        if line_right_exists:
            cell.right = True

    def merge_column_cell(self, row_pairs, cells):
        """
        进行每一行的 cell 合并

        :param row_pairs:
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

    def merge_row_cell(self, sorted_joint_point, col_pairs, column_merge_cells):
        """
        进行列 cell 合并
        :param sorted_joint_point:
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
        logger.info(f"all_cells: {len(all_cells)}")

        return all_cells

    def generate_table_cell(self, table_idx, bbox, table_bbox: Dict,
                            v_segments: List[Line], h_segments: List[Line],
                            pdf_images: List = [], line_tol=2, is_pdf=True):
        """
        select elements which lie within table_bbox

        :param table_idx:
        :param bbox:
        :param table_bbox:
        :param v_segments:
        :param h_segments:
        :param pdf_images:
        :param line_tol:
        :param is_pdf:
        :return:
        """
        text_bbox, h_s_v2, v_s_v2, = self.get_text_in_table_bbox_v2(bbox,
                                                                    v_segments=v_segments,
                                                                    h_segments=h_segments)

        raw_joint_point = table_bbox[bbox]
        # filter join point by image
        joint_point = self.filter_joint_point_match_image(raw_joint_point,
                                                          pdf_images=pdf_images)

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
        cols = PdfUtils.merge_close_lines(sorted(cols), line_tol=line_tol, last_merge_threold=10)
        rows = PdfUtils.merge_close_lines(sorted(rows, reverse=is_pdf), line_tol=line_tol, last_merge_threold=10)

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

        col_avg_width = calc_cell_width(col_pairs, )
        row_avg_width = calc_cell_width(row_pairs, )

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
                               diff=self.line_tol,
                               col_avg_width=col_avg_width,
                               row_avg_width=row_avg_width)

        table_cell_image = ""

        logger.info(f"col_pairs: {len(col_pairs)} - {col_pairs}")
        logger.info(f"row_pairs: {len(row_pairs)} - {row_pairs}")
        if self.debug:
            logger.info(f"cells: {len(cells)} - {cells}")

        sorted_joint_point = sorted(new_joint_point, key=lambda x: (x[1], x[0]))

        # 进行行合并
        column_merge_cells = self.merge_column_cell(row_pairs=row_pairs, cells=cells)

        row_merge_cells, remove_cells = self.merge_row_cell(sorted_joint_point=sorted_joint_point,
                                                            col_pairs=col_pairs,
                                                            column_merge_cells=column_merge_cells)

        all_cells = self.get_merge_cell(column_merge_cells=column_merge_cells,
                                        row_merge_cells=row_merge_cells,
                                        remove_cells=remove_cells, )

        all_cell_results = TableProcessUtils.modify_cell_info(all_cells,
                                                              cols=cols,
                                                              rows=rows,
                                                              one_col_width=one_col_width,
                                                              one_row_height=one_row_height,
                                                              is_pdf=is_pdf)

        metric = {
            "table_idx": table_idx,
            "joint_point": joint_point,
            "new_joint_point": new_joint_point,
            "col_pairs": col_pairs,
            "row_pairs": row_pairs,
            "table_cell_image": table_cell_image,
            "pdf_table_bbox": bbox,
        }

        return text_bbox, all_cell_results, metric

    def match_table_cell_and_text_cell(self, table_idx, table_cells: List[Cell], text_cells, save_html_file):
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

    @staticmethod
    def transform_bbox_and_logits_to_rect(table_bboxs, logits=None, image_file=None, line_diff=0.5):
        """
        对识别的表格cell 和 logit坐标进行后处理修正

        :param table_bboxs:
        :param logits:
        :param image_file:
        :param line_tol:
        :return:
        """
        linewidth = line_diff * 2
        line_rects = []
        joint_point = []
        for box in table_bboxs:
            points = box.reshape(4, 2)
            joint_point.extend(points)

            (x1, y1) = points[0]
            (x2, y2) = points[1]
            (x3, y3) = points[2]
            (x4, y4) = points[3]
            if abs(x1 - x4) < linewidth:
                x1 = x4 = (x1 + x4) / 2
            if abs(x2 - x3) < linewidth:
                x2 = x3 = (x2 + x3) / 2
            if abs(y1 - y2) < linewidth:
                y1 = y2 = (y1 + y2) / 2
            if abs(y3 - y4) < linewidth:
                y3 = y4 = (y3 + y4) / 2

            rect1 = LTRect(linewidth=line_diff, bbox=(x1, y1, x2, y1), )
            rect2 = LTRect(linewidth=line_diff, bbox=(x2, y2, x3, y3), )
            rect3 = LTRect(linewidth=line_diff, bbox=(x4, y4, x3, y3), )
            rect4 = LTRect(linewidth=line_diff, bbox=(x1, y1, x4, y4), )
            line_rects.extend([rect1, rect2, rect3, rect4])

        logger.info(f"total : {len(joint_point)}")

        joint_point_rects = []
        for x, y in joint_point:
            bbox = (x - line_diff, y, x + line_diff, y)
            point = LTRect(linewidth=linewidth, bbox=bbox, )
            joint_point_rects.append(point)

        return line_rects, joint_point_rects
