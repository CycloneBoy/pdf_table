#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_common
# @Author  ：cycloneboy
# @Date    ：20xx/6/5 15:00

import copy
import traceback
from collections import defaultdict
from typing import List, Dict

import cv2
import numpy as np
from pdfminer.layout import LTChar, LTTextLineHorizontal, LTAnno, LTImage, LTTextLineVertical, LTTextLine

from pdftable.entity.table_entity import Line, OcrCell, Point, TableEval, TableUnit
from pdftable.utils.ocr import OcrCommonUtils
from .table_core import Cell
from ...entity import LineDirectionType
from ...utils import (
    logger,
    MathUtils,
    FileUtils,
    MatchUtils,
    CommonUtils
)

__all__ = [
    'equal',
    'equal_in',
    'equal_point',
    'cell_equal_in',
    'point_equal_in',
    'cell_equal_in_list',
    'cell_all_in_points',
    'gen_lines',
    'point_in_bbox',
    'point_in_bbox_list',
    'box_in_other_box',
    'cell_horizontal_line_exists',
    'cell_vertical_line_exists',
    'line_in_line_pair',
    'find_cell_in_line_pairs',
    'distance',
    'compute_iou',
    'compute_iou_v2',
    'calc_cell_width',
    'cell_horizontal_line_exists_v2',
    'cell_vertical_line_exists_v2',
    'TableProcessUtils',
]


def equal(a, b, diff=2):
    return abs(a - b) < diff


def equal_in(x, same_list, diff=2):
    flag = False
    for item in same_list:
        if equal(x, item, diff=diff):
            flag = True
            break
    return flag


def equal_point(a, b, diff=2):
    return equal(a[0], b[0], diff=diff) and equal(a[1], b[1], diff=diff)


def cell_equal_in(x, cell: Cell, diff=2):
    flag = False
    for item in cell.points:
        if equal_point(x, item, diff=diff):
            flag = True
            break
    return flag


def point_equal_in(x, points: List, diff=2):
    flag = False
    for item in points:
        if equal_point(x, item, diff=diff):
            flag = True
            break
    return flag


def cell_equal_in_list(cell: Cell, cells: List[Cell], diff=2):
    flag = False
    for item in cells:
        if equal_point(cell.lb, item.lb, diff=diff) \
                and equal_point(cell.rt, item.rt, diff=diff):
            flag = True
            break
    return flag


def cell_all_in_points(cell: Cell, points: List, diff=2):
    flag = True
    for point in cell.points:
        if not point_equal_in(point, points, diff=diff):
            flag = False
            break
    return flag


def gen_lines(cols, is_row=False):
    reverse = False if not is_row else True
    cols = sorted(cols, reverse=reverse)
    cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
    return cols


def point_in_bbox(point, box, diff=2):
    x, y = point
    x1, y1, x2, y2 = box

    min_y = min(y1, y2)
    max_y = max(y1, y2)

    flag = False
    if x1 - diff <= x <= x2 + diff and min_y - diff <= y <= max_y + diff:
        flag = True
    return flag


def point_in_bbox_list(point, boxs: List, diff=2):
    flag = False
    for box in boxs:
        if point_in_bbox(point, box, diff=diff):
            flag = True
            break
    return flag


def box_in_other_box(box_1, box_2, diff=2):
    """
    box2 is in box1

    :param box_1:
    :param box_2:
    :param diff:
    :return:
    """
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2

    flag = False
    min_y_1 = min(y1, y2)
    max_y_1 = max(y1, y2)

    min_y_2 = min(y3, y4)
    max_y_2 = max(y3, y4)

    if x3 >= x1 - diff and x4 <= x2 + diff and min_y_1 - diff <= min_y_2 <= max_y_2 <= max_y_1 + diff:
        flag = True

    return flag


def cell_horizontal_line_exists(cell: Cell, horizontal_line_list: List, diff=2,
                                row_avg_width=None, avg_horizontal_height=None):
    """
    判断当前cell 水平是否已经存在

    :param cell:
    :param horizontal_line_list:
    :param diff:
    :param row_avg_width:
    :param avg_horizontal_height: 水平线的平均高度
    :return:
    """

    # x1, y1, x2, y2 = line1
    x1 = cell.x1
    y1 = cell.y1
    x2 = cell.x2
    y2 = cell.y2

    min_x = min(x1, x2)
    max_x = max(x1, x2)

    line_top_exists = False
    line_bottom_exists = False
    diff = avg_horizontal_height if avg_horizontal_height is not None else diff

    for index, (x3, y3, x4, y4) in enumerate(horizontal_line_list):
        min_target_x = min(x3, x4)
        max_target_x = max(x3, x4)
        # 水平线
        if abs(y3 - y4) < diff:
            # 判断top 当前行
            if abs(y1 - y3) < diff or abs(y1 - y4) < diff:
                if min_target_x - diff <= min_x < max_x <= max_target_x + diff:
                    line_top_exists = True
                    if line_bottom_exists:
                        break
            # 判断bottom 当前行
            elif abs(y2 - y3) < diff or abs(y2 - y4) < diff:
                if min_target_x - diff <= min_x < max_x <= max_target_x + diff:
                    line_bottom_exists = True
                    if line_top_exists:
                        break

        if line_top_exists and line_bottom_exists:
            break

    return line_top_exists, line_bottom_exists


def cell_horizontal_line_exists_v2(cell: Cell, horizontal_line_list: List[Line], diff=2, match_radio=0.8,
                                   row_avg_width=None, avg_horizontal_height=None):
    """
    判断当前cell 水平是否已经存在

    :param cell:
    :param horizontal_line_list:
    :param diff:
    :param row_avg_width:
    :param avg_horizontal_height: 水平线的平均高度
    :return:
    """

    line_top_exists = False
    line_bottom_exists = False
    diff = avg_horizontal_height if avg_horizontal_height is not None else diff

    for index, line in enumerate(horizontal_line_list):
        # 水平线
        if line.direction == LineDirectionType.HORIZONTAL:
            # 判断top 当前行
            if not line_top_exists and abs(cell.min_y - line.min_y) < diff or abs(
                    cell.min_y - line.min_y) < diff + line.height:
                if two_line_intersect(line1=(line.min_x, line.max_x),
                                      line2=(cell.min_x, cell.max_x),
                                      diff=diff,
                                      match_radio=match_radio):
                    line_top_exists = True
                    if line_bottom_exists:
                        break

            # 判断bottom 当前行
            elif not line_bottom_exists and abs(cell.max_y - line.max_y) < diff or abs(
                    cell.max_y - line.max_y) < diff + line.height:
                if two_line_intersect(line1=(line.min_x, line.max_x),
                                      line2=(cell.min_x, cell.max_x),
                                      diff=diff,
                                      match_radio=match_radio):
                    line_bottom_exists = True
                    if line_top_exists:
                        break

        if line_top_exists and line_bottom_exists:
            break

    return line_top_exists, line_bottom_exists


def two_line_intersect(line1, line2, diff=2, match_radio=0.8):
    """
    判断两条线段是否相交

    :param line1:
    :param line2:
    :param diff:
    :param match_radio: 相交程度
    :return:
    """
    intersect_flag = False
    line1_min_x = min(line1[0], line1[1])
    line1_max_x = max(line1[0], line1[1])
    line1_width = abs(line1_max_x - line1_min_x)

    line2_min_x = min(line2[0], line2[1])
    line2_max_x = max(line2[0], line2[1])
    line2_width = abs(line2_max_x - line2_min_x)

    # 不想交
    if line2_max_x < line1_min_x - diff or line2_min_x > line1_max_x + diff:
        return False

    if line2_min_x - diff <= line1_min_x < line1_max_x <= line2_max_x + diff:
        # 刚好靠中间
        return True
    elif line2_min_x - diff < line1_min_x <= line2_max_x + diff:
        # line2靠左
        flag = (line2_max_x - line1_min_x) / line2_width >= match_radio
        return flag
    elif line1_min_x - diff < line2_min_x - diff < line1_max_x + diff:
        # line2靠右
        flag = (line1_max_x - line2_min_x) / line2_width >= match_radio
        return flag

    return intersect_flag


def cell_vertical_line_exists(cell: Cell, vertical_line_list: List, diff=2,
                              col_avg_width=None, avg_vertical_width=None):
    """
    判断当前cell 垂直线 是否已经存在

    :param cell:
    :param vertical_line_list:
    :param diff:
    :param col_avg_width:
    :param avg_vertical_width:
    :return:
    """

    # x1, y1, x2, y2 = line1
    x1 = cell.x1
    y1 = cell.y1
    x2 = cell.x2
    y2 = cell.y2

    min_y = min(y1, y2)
    max_y = max(y1, y2)

    line_left_exists = False
    line_right_exists = False

    for index, (x3, y3, x4, y4) in enumerate(vertical_line_list):
        min_target_y = min(y3, y4)
        max_target_y = max(y3, y4)
        # 水平线
        diff = avg_vertical_width if avg_vertical_width is not None else diff
        if abs(x3 - x4) < diff:
            # 判断left 当前列
            if abs(x1 - x3) < diff or abs(x1 - x4) < diff:
                if min_target_y - diff <= min_y < max_y <= max_target_y + diff:
                    line_left_exists = True
                    if line_right_exists:
                        break
            # 判断right 当前列
            if abs(x2 - x3) < diff or abs(x2 - x4) < diff:
                if min_target_y - diff <= min_y < max_y <= max_target_y + diff:
                    line_right_exists = True
                    if line_left_exists:
                        break

        if line_left_exists and line_right_exists:
            break

    return line_left_exists, line_right_exists


def cell_vertical_line_exists_v2(cell: Cell, vertical_line_list: List[Line], diff=2, match_radio=0.8,
                                 col_avg_width=None, avg_vertical_width=None):
    """
    判断当前cell 垂直线 是否已经存在

    :param cell:
    :param vertical_line_list:
    :param diff:
    :param col_avg_width:
    :param avg_vertical_width:
    :return:
    """

    line_left_exists = False
    line_right_exists = False

    for index, line in enumerate(vertical_line_list):
        # 垂直线
        diff = avg_vertical_width if avg_vertical_width is not None else diff
        if line.direction == LineDirectionType.VERTICAL:
            # 判断left 当前列
            if not line_left_exists and abs(cell.min_x - line.min_x) < diff or abs(
                    cell.min_x - line.min_x) < diff + line.width:
                if two_line_intersect(line1=(line.min_y, line.max_y),
                                      line2=(cell.min_y, cell.max_y),
                                      diff=diff,
                                      match_radio=match_radio):
                    line_left_exists = True
                    if line_right_exists:
                        break
            # 判断right 当前列
            if not line_right_exists and abs(cell.max_x - line.max_x) < diff or abs(
                    cell.max_x - line.max_x) < diff + line.width:
                if two_line_intersect(line1=(line.min_y, line.max_y),
                                      line2=(cell.min_y, cell.max_y),
                                      diff=diff,
                                      match_radio=match_radio):
                    line_right_exists = True
                    if line_left_exists:
                        break

        if line_left_exists and line_right_exists:
            break

    return line_left_exists, line_right_exists


def line_in_line_pair(line, line_pairs: List, diff=2):
    """
    当前线段是否存在

    :param line:
    :param line_pairs:
    :param diff:
    :return:
    """
    line_pairs.sort(key=lambda x: x[0])
    flag = False
    x1, x2 = line
    for x3, x4 in line_pairs:
        if equal(x1, x3, diff=diff) and equal(x2, x4, diff=diff):
            flag = True
            break

    return flag


def find_cell_in_line_pairs(cells: List[Cell], line, diff=2):
    """
    查找 当前线段 中的cell

    :param cells:
    :param line:
    :param diff:
    :return:
    """
    results = []
    begin_x, end_x = line
    for index, cell in enumerate(cells):
        if begin_x - diff <= cell.x1 < cell.x2 <= end_x + diff:
            results.append(cell)

    results.sort(key=lambda x: (x.x1, x.y1))
    return results


def distance(box_1, box_2):
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def compute_iou_v2(boxes_preds, boxes_labels):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    box1_x1 = boxes_preds[0]
    box1_y1 = boxes_preds[1]
    box1_x2 = boxes_preds[2]
    box1_y2 = boxes_preds[3]
    box2_x1 = boxes_labels[0]
    box2_y1 = boxes_labels[1]
    box2_x2 = boxes_labels[2]
    box2_y2 = boxes_labels[3]

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    diff_x = (x2 - x1)
    if diff_x < 0:
        diff_x = 0

    diff_y = (y2 - y1)
    if diff_y < 0:
        diff_y = 0

    intersection = diff_x * diff_y
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def calc_cell_width(col_pairs: List, min_width=4):
    """
    计算cell 的宽度
    :param col_pairs:
    :param min_width:
    :return:
    """
    col_widths = []
    for col in col_pairs:
        width = round(abs(col[0] - col[1]))
        if width < min_width:
            continue
        col_widths.append(width)

    avg_width = np.average(col_widths)
    return avg_width


class TableProcessUtils(object):

    @staticmethod
    def convert_table_cell_to_dict(table_cells: List[Cell]) -> Dict[str, List[Cell]]:
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

    @staticmethod
    def convert_table_cell_to_image_scale(table_cells: List[Cell], image_scalers) -> List[Cell]:
        """
            将所有的table cell 转换Image上形式

        :param table_cells:
        :param image_scalers:
        :return:
        """

        for index, cell in enumerate(table_cells):
            x1, y1, x2, y2 = MathUtils.scale_pdf(cell.bbox, image_scalers)
            cell.image_bbox = [x1, y1, x2, y2]
            cell.x1 = x1
            cell.y1 = y1
            cell.x2 = x2
            cell.y2 = y2
            cell.clean_text()

        return table_cells

    @staticmethod
    def cell_to_html(table_cells: List[Cell], first_header=True,
                     add_width=True, add_text=True):
        """
        cell to html

        :param table_cells:
        :param first_header:
        :param add_width:
        :param add_text:
        :return:
        """
        row_dict_sorted = TableProcessUtils.convert_table_cell_to_dict(table_cells)

        first_header = False
        if len(row_dict_sorted) > 1:
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
                colspan = f'colspan="{int(cell.col_span)}" ' if cell.col_span > 1 else ""
                rowspan = f'rowspan="{int(cell.row_span)}" ' if cell.row_span > 1 else ""
                if add_width:
                    width = f'width="{round(cell.width_ratio * 100)}%"' if cell.width > 0 else ""
                else:
                    width = ""

                if fix_row_span_same:
                    rowspan = ""

                texts = cell.text.replace("\n", "<br/>") if add_text else ""
                one_cell = f'<{table_cell_token} {colspan}{rowspan}{width}>{texts}</{table_cell_token}>'
                one_cols.append(one_cell)
            one_cols.append('</tr>')
            html_row_list.append(one_cols)

        # show html
        table_html = [
            '<table border="1">',
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

    @staticmethod
    def write_html_result_header(file_name):
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
        FileUtils.append_to_text(file_name, html_content)

    @staticmethod
    def write_html_result_footer(file_name):
        html_content = [
            "</table>",
            "</body>",
            "</html>",
        ]
        FileUtils.append_to_text(file_name, html_content)

    @staticmethod
    def find_text_box_belong_cell(text_cell, table_cells: List[Cell], diff=2):
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

    @staticmethod
    def find_vertical_text_box_belong_cell(text_cell, table_cells: List[Cell], diff=2):
        """
            查找 vertical text box belong cell

        :param text_cell:
        :param table_cells:
        :param diff:
        :return:
        """

        cell_in = None
        begin_cell = None
        end_cell = None

        x0, y0, x1, y1 = text_cell
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        for index, cell in enumerate(table_cells):

            # 同一列：
            if cell.min_x - diff < min_x < max_x < cell.max_x + diff:
                if cell.y2 - diff < y1 < y0 < cell.y1 + diff:
                    cell_in = cell
                    break
                elif cell.y2 - diff < y0 < cell.y1 + diff:
                    begin_cell = cell
                elif cell.y2 - diff < y1 < cell.y1 + diff:
                    end_cell = cell

        return cell_in, begin_cell, end_cell

    @staticmethod
    def find_char_belong_cell(char_object: LTChar, find_table_cells: List[Cell], diff=2):
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

    @staticmethod
    def find_vertical_char_belong_cell(char_object: LTChar, find_table_cells: List[Cell], diff=2):
        """
        查找 char 属于哪一个table cell

        :param char_object:
        :param find_table_cells:
        :param diff:
        :return:
        """
        find_index = -1
        for index, cell in enumerate(find_table_cells):
            if cell.y2 - diff < char_object.y0 <= char_object.y1 < cell.y1 + diff:
                find_index = index
                break

            if cell.y2 - diff < char_object.y0 < cell.y1 <= char_object.y1:
                top_diff = cell.y1 - char_object.y0
                bottom_diff = char_object.y1 - cell.y1
                top_radio = top_diff / char_object.height

                if top_radio > 0.3:
                    find_index = index
                    break

        return find_index

    @staticmethod
    def table_bbox_merge(table_bboxs: Dict, diff=10):
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

    @staticmethod
    def split_text_cell_horizontal(text_cell: LTTextLineHorizontal,
                                   table_cells: List[Cell], diff=2):
        """
        拆分 一个horizontal text cell 到多个

        :param text_cell:
        :param table_cells:
        :param diff:
        :return:
        """
        find_table_cells = TableProcessUtils.find_cell_cross_text_cell(text_cell=text_cell,
                                                                       table_cells=table_cells,
                                                                       diff=diff)
        # 多列拆分
        new_text_cells = [LTTextLineHorizontal(word_margin=text_cell.word_margin) for _ in find_table_cells]

        if len(new_text_cells) == 0:
            logger.info(f"没有找打对应的cell: {text_cell}")
            return [text_cell]

        before_cell_index = -1
        for index_char, char_object in enumerate(text_cell._objs):
            if isinstance(char_object, LTChar):
                cell_index = TableProcessUtils.find_char_belong_cell(char_object, find_table_cells=find_table_cells,
                                                                     diff=diff)
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
        # logger.info(f"拆分text cell: [{len(text_cell)} = {' + '.join([str(len(char_cell)) for char_cell in results])}] "
        #             f"- 【{src_text}】 -> 【 {' + '.join([char_cell.get_text() for char_cell in results])} 】")

        return results

    @staticmethod
    def split_text_cell_vertical(text_cell: LTTextLineVertical, table_cells: List[Cell], diff=2):
        """
        拆分 一个垂直 text cell 到多个

        :param text_cell:
        :param table_cells:
        :param diff:
        :return:
        """
        find_table_cells = TableProcessUtils.find_cell_cross_vertical_text_cell(text_cell=text_cell,
                                                                                table_cells=table_cells,
                                                                                diff=diff)
        # 多列拆分
        new_text_cells = [LTTextLineVertical(word_margin=text_cell.word_margin) for _ in find_table_cells]

        if len(new_text_cells) == 0:
            logger.infof(f"没有找打对应的cell: {text_cell}")
            return [text_cell]

        before_cell_index = -1
        for index_char, char_object in enumerate(text_cell._objs):
            if isinstance(char_object, LTChar):
                cell_index = TableProcessUtils.find_vertical_char_belong_cell(char_object,
                                                                              find_table_cells=find_table_cells,
                                                                              diff=diff)
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
        logger.info(
            f"拆分text vertical cell: [{len(text_cell)} = {' + '.join([str(len(char_cell)) for char_cell in results])}] "
            f"- 【{src_text}】 -> 【 {' + '.join([char_cell.get_text() for char_cell in results])} 】")

        return results

    @staticmethod
    def find_cell_cross_vertical_text_cell(text_cell: LTTextLineVertical, table_cells: List[Cell], diff=2):
        """
        查找 text cell 和 table cell 相交的cell

        :param text_cell:
        :param table_cells:
        :param diff:
        :return:
        """
        text_cell_x_min = min(text_cell.x0, text_cell.x1)
        text_cell_x_max = max(text_cell.x0, text_cell.x1)

        find_table_cells = []
        for cell in table_cells:
            if cell.min_x - diff < text_cell_x_min < text_cell_x_max < cell.max_x + diff:
                if cell.y2 < cell.y1 < text_cell.y0 - diff \
                        or text_cell.y1 - diff < cell.y2 < cell.y1:
                    continue

                find_table_cells.append(cell)

        logger.info(f"find_table_cells: {text_cell} -> {len(find_table_cells)} -> {find_table_cells}")
        find_table_cells.sort(key=lambda x: -x.y1)
        return find_table_cells

    @staticmethod
    def find_cell_cross_text_cell(text_cell: LTTextLineHorizontal, table_cells: List[Cell], diff=2):
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

        # logger.info(f"find_table_cells: {text_cell} -> {len(find_table_cells)} -> {find_table_cells}")
        find_table_cells.sort(key=lambda x: x.x1)
        return find_table_cells

    @staticmethod
    def text_box_split_to_cell(table_cells: List[Cell], text_cells, diff=2):
        """
        text box 拆分: 将一行文本box 跨多个cell的拆分

        :param table_cells:
        :param text_cells:
        :param diff:
        :return:
        """
        new_text_cells_h = TableProcessUtils.split_horizontal_text_cell(text_cells["horizontal"],
                                                                        table_cells=table_cells,
                                                                        diff=diff)

        new_text_cells_v = TableProcessUtils.split_vertical_text_cell(text_cells["vertical"],
                                                                      table_cells=table_cells,
                                                                      diff=diff)

        new_text_cells = {
            "horizontal": new_text_cells_h,
            "vertical": new_text_cells_v,
        }

        return new_text_cells

    @staticmethod
    def split_horizontal_text_cell(text_cells_h: List[LTTextLineHorizontal], table_cells: List[Cell], diff=2):
        """
        拆分水平text cell

        :param text_cells_h:
        :param table_cells:
        :param diff:
        :return:
        """
        new_text_cells_h = []
        for index, text_cell in enumerate(text_cells_h):
            new_cell_box = (text_cell.x0, text_cell.y1, text_cell.x1, text_cell.y0)

            cell_in, begin_cell, end_cell = TableProcessUtils.find_text_box_belong_cell(text_cell=new_cell_box,
                                                                                        table_cells=table_cells)

            if cell_in is not None:
                new_text_cells_h.append(text_cell)
                continue
            elif begin_cell is not None or end_cell is not None:
                split_text_cell_list = TableProcessUtils.split_text_cell_horizontal(text_cell=text_cell,
                                                                                    table_cells=table_cells,
                                                                                    diff=diff)
                new_text_cells_h.extend(split_text_cell_list)
            else:
                new_text_cells_h.append(text_cell)
                logger.info(
                    f"拆分text cell失败,不进行拆分，直接保留原始text cell：{index} - {text_cell.get_text()} - {text_cell}")
        return new_text_cells_h

    @staticmethod
    def split_vertical_text_cell(text_cells_v: List[LTTextLineVertical], table_cells: List[Cell], diff=2):
        """
        拆分垂直text cell

        :param text_cells_v:
        :param table_cells:
        :param diff:
        :return:
        """
        new_text_cells_h = []
        for index, text_cell in enumerate(text_cells_v):
            new_cell_box = (text_cell.x0, text_cell.y1, text_cell.x1, text_cell.y0)

            cell_in, begin_cell, end_cell = TableProcessUtils.find_vertical_text_box_belong_cell(text_cell=new_cell_box,
                                                                                                 table_cells=table_cells)

            if cell_in is not None:
                new_text_cells_h.append(text_cell)
                continue
            elif begin_cell is not None or end_cell is not None:
                split_text_cell_list = TableProcessUtils.split_text_cell_vertical(text_cell=text_cell,
                                                                                  table_cells=table_cells,
                                                                                  diff=diff)
                new_text_cells_h.extend(split_text_cell_list)
            else:
                new_text_cells_h.append(text_cell)
                logger.info(
                    f"拆分text cell失败,不进行拆分，直接保留原始text cell：{index} - {text_cell.get_text()} - {text_cell}")
        return new_text_cells_h

    @staticmethod
    def show_text_cell(image, save_image_file, text_cells: List[List], image_scalers=None):
        """
        显示表格的网格图片

        :param image:
        :param save_image_file:
        :param text_cells:
        :param image_scalers:
        :return:
        """

        color_list = CommonUtils.get_color_list()
        try:
            src_im = copy.deepcopy(image)

            thickness = 2

            for index, cell_list in enumerate(text_cells):
                for cell_index, cell in enumerate(cell_list):
                    if isinstance(cell, LTTextLine):
                        x1, y1, x2, y2 = MathUtils.scale_pdf((cell.x0, cell.y0, cell.x1, cell.y1), image_scalers)
                    elif isinstance(cell, OcrCell):
                        box = [round(item) for item in cell.to_bbox()]
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                    start_point = (x1, y1)
                    end_point = (x2, y2)

                    color = color_list[index]

                    cv2.rectangle(src_im, start_point, end_point, color, thickness)

            FileUtils.check_file_exists(save_image_file)
            cv2.imwrite(save_image_file, src_im)

            logger.info(f"保存text cell图像：{save_image_file}")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"显示text cell图像异常：{e}")

        return save_image_file

    @staticmethod
    def modify_predict_table_html(table_html, diff_texts: List = None, diff_structures: List = None):
        """
        根据 对比结果 给cell 添加颜色

        :param table_html:
        :param diff_texts:
        :param diff_structures:
        :return:
        """
        if len(diff_texts) == 0:
            return table_html

        color_list = CommonUtils.get_html_color_list()

        row_begin_index = MatchUtils.find_text_span_index(table_html, span="<tr", total=-1)

        new_table_html = copy.deepcopy(table_html)
        find_index = []

        diff_cells = diff_texts if diff_structures is None or len(diff_structures) == 0 else diff_structures
        add_token_length_dict = {}
        for index, diff_text in enumerate(diff_cells):
            if "row_index" not in diff_text:
                continue
            row_index = diff_text["row_index"]
            column_index = diff_text["column_index"]

            start_index = row_begin_index[row_index - 1]
            end_index = row_begin_index[row_index] if len(row_begin_index) > row_index else -1
            row_span = table_html[start_index:end_index]

            column_begin_index = MatchUtils.find_text_span_index(row_span, span="<td", total=column_index)
            column_start_index = start_index + column_begin_index[column_index - 1]
            column_span = table_html[column_start_index:column_start_index + 50]
            # logger.info(f"{start_index} - {row_span}")
            # logger.info(f"{column_start_index} - {column_span}")

            find_index.append(column_start_index)

            add_color = color_list[index % len(color_list)]
            add_span = f" style='background-color:{add_color.lower()}'"

            add_token_length = 0
            for k, v in add_token_length_dict.items():
                if k < column_start_index:
                    add_token_length += v

            sep_index = column_start_index + add_token_length
            add_token_length_dict[column_start_index] = len(add_span)

            new_table_html_left = new_table_html[:sep_index]
            new_table_html_middle = new_table_html[sep_index:sep_index + 3]
            new_table_html_right = new_table_html[sep_index + 3:]

            new_table_html = f"{new_table_html_left}<td{add_span}{new_table_html_right}"

        # logger.info(row_begin_index)
        # print(new_table_html)
        return new_table_html

    @staticmethod
    def check_table_match_images(bbox, images: List[LTImage], iou_threshold=0.5):
        """
        判断表格是否和图片重合

        :param bbox:
        :param images:
        :param iou_threshold:
        :return:
        """
        match = False
        match_image = None
        remain_images = []
        for index, image in enumerate(images):
            if not match:
                image_bbox = image.bbox
                flag = box_in_other_box(image_bbox, bbox)
                if flag:
                    match = True
                    match_image = image
                else:
                    remain_images.append(image)
            else:
                remain_images.append(image)

        return match, match_image, remain_images

    @staticmethod
    def get_pdf_scaler(image_shape, pdf_shape):
        image_height, image_width = image_shape
        pdf_height, pdf_width = pdf_shape

        image_width_scaler = image_width / float(pdf_width)
        image_height_scaler = image_height / float(pdf_height)
        image_scalers = (image_width_scaler, image_height_scaler, pdf_height)

        pdf_width_scaler = pdf_width / float(image_width)
        pdf_height_scaler = pdf_height / float(image_height)
        pdf_scalers = (pdf_width_scaler, pdf_height_scaler, image_height)
        return image_scalers, pdf_scalers

    @staticmethod
    def filter_layout_figure(layout_result: List, table_bbox, label="figure", score_threshold=0.8):
        """
        过滤 图片误识别成表格

        :param layout_result:
        :param table_bbox:
        :param label:
        :param score_threshold:
        :return:
        """
        figures = TableProcessUtils.get_layout_by_type(layout_result=layout_result,
                                                       label=label,
                                                       score_threshold=score_threshold)

        match_flag = False
        match_figure = None
        for item in figures:
            figure_bbox = item["bbox"]
            if box_in_other_box(figure_bbox, table_bbox):
                match_flag = True
                match_figure = item
                continue

        return match_flag, match_figure

    @staticmethod
    def get_layout_by_type(layout_result: List, label="figure", score_threshold=0.8):
        """
        过滤 图片误识别成表格

        :param layout_result:
        :param label:
        :param score_threshold:
        :return:
        """
        results = [item for item in layout_result
                   if item["label"].lower() == label.lower() and item["score"] >= score_threshold]

        results.sort(key=lambda x: x["bbox"][1])
        return results

    @staticmethod
    def get_text_in_table_bbox(bbox, ocr_results: List[OcrCell], diff=2):
        """
        筛选 在表格区域中的文本

        :param bbox:
        :param ocr_results:
        :param diff:
        :return:
        """
        horizontal_cells = []
        remain_cells = []
        left_top = Point(x=bbox[0], y=bbox[1])
        right_bottom = Point(x=bbox[2], y=bbox[3])

        for cell in ocr_results:
            # y in box
            if left_top.y - diff <= cell.center_point.y <= right_bottom.y + diff \
                    and left_top.x - diff <= cell.center_point.x <= right_bottom.x + diff:
                horizontal_cells.append(cell)
            else:
                remain_cells.append(cell)

        return horizontal_cells, remain_cells

    @staticmethod
    def ocr_post_process(text: str):
        """
        ocr 结果后处理
            Oo -> 0
        :param text:
        :return:
        """
        new_text = text
        clean_text = CommonUtils.remove_space(text)
        if len(clean_text) == 1 and MatchUtils.match_pattern_flag(clean_text, pattern=MatchUtils.PATTERN_OCR_TEXT_0):
            new_text = "0"

        if CommonUtils.is_ocr_number(clean_text):
            dot_count = str(text).count(".")
            doc_right_index = text.rfind(".")
            if dot_count > 1 and doc_right_index > -1:
                left_text = text[:doc_right_index].replace(".", ",")
                new_text = f"{left_text}{text[doc_right_index:]}"

        if text != new_text:
            logger.info(f"进行OCR后处理：{text} -> {new_text}")

        return new_text

    @staticmethod
    def get_table_bbox_regions(ocr_system_output, diff=5):
        """
        获取表格区域

        :param ocr_system_output:
        :param diff:
        :return:
        """
        regions = None
        if ocr_system_output is None:
            return regions
        layout_result = ocr_system_output.layout_result
        if ocr_system_output is not None and layout_result is not None:
            layout_tables = TableProcessUtils.get_layout_by_type(layout_result=layout_result,
                                                                 label="table",
                                                                 score_threshold=0.2)
            layout_figures = TableProcessUtils.get_layout_by_type(layout_result=layout_result,
                                                                  label="figure",
                                                                  score_threshold=0.7)

            layout_texts = TableProcessUtils.get_layout_by_type(layout_result=layout_result,
                                                                label="text",
                                                                score_threshold=0.7)
            if len(layout_texts) == 0:
                layout_texts = [item for item in layout_result
                                if item["label"].lower() in ["figure", "text"] and item["score"] >= 0.7]

            min_text_x = []
            max_text_x = []
            for item in layout_texts:
                x1, y1, x2, y2 = [int(o) for o in item["bbox"]]
                min_text_x.append(min(x1, x2))
                max_text_x.append(max(x1, x2))

            image_width = ocr_system_output.image_shape[1]
            min_x = min(min_text_x) if len(min_text_x) > 0 else diff
            max_x = max(max_text_x) if len(max_text_x) > 0 else image_width - diff

            all_layout = []
            all_layout.extend(layout_tables)
            all_layout.extend(layout_figures)
            regions = []
            for item in all_layout:
                x1, y1, x2, y2 = [int(o) for o in item["bbox"]]
                if x1 > min_x:
                    x1 = min_x

                if x2 < max_x:
                    x2 = max_x

                regions.append([min(x1, x2) - diff, min(y1, y2) - diff,
                                abs(x2 - x1) + diff, abs(y2 - y1) + diff])
        return regions

    @staticmethod
    def cell_in_table_bboxes(cell_bbox, layout_tables: List = None, ):
        """

        :param cell_bbox:
        :param layout_tables:
        :return:
        """
        flag = False
        if layout_tables is None:
            return True

        for table in layout_tables:
            x, y, w, h = table
            table_bbox = x, y, x + w, y + h
            if box_in_other_box(table_bbox, cell_bbox):
                flag = True
                break
        return flag

    @staticmethod
    def add_box_to_image(threshold, bboxs: List, layout_tables: List = None,
                         color=(255, 255, 255), thickness=3,
                         diff_y=3, diff_x=10):
        """
        添加box 到 image

        :param threshold:
        :param bboxs:
        :param layout_tables:
        :param color:
        :param thickness:
        :param diff_y:
        :param diff_x:
        :return:
        """
        bbox_list = bboxs.tolist() if isinstance(bboxs, np.ndarray) else bboxs

        if layout_tables is None:
            filter_bboxs = bbox_list
        else:
            filter_bboxs = []
            for index, box in enumerate(bbox_list):
                new_point = [int(o) for o in box]
                cell_bbox = min(new_point[0], new_point[2]), new_point[1], max(new_point[4], new_point[6]), new_point[5]
                if TableProcessUtils.cell_in_table_bboxes(cell_bbox, layout_tables=layout_tables):
                    filter_bboxs.append(box)
                else:
                    # 构造新的box
                    match_box = None
                    x3, y3, x4, y4 = cell_bbox
                    for table in layout_tables:
                        x, y, w, h = table
                        x1, y1, x2, y2 = x, y, x + w, y + h

                        min_y_1 = min(y1, y2)
                        max_y_1 = max(y1, y2)

                        min_y_2 = min(y3, y4)
                        max_y_2 = max(y3, y4)
                        if min_y_1 - diff_y <= min_y_2 <= max_y_2 <= max_y_1 + diff_y:
                            match_box = table
                            break
                    if match_box is not None:
                        x, y, w, h = match_box
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        if x3 < x1 < x4:
                            if abs(new_point[0] - new_point[6]) < diff_x:
                                new_box = new_point
                            else:
                                max_x = max(new_point[0], new_point[6])
                                new_x = x1 if max_x > x1 else max_x
                                new_box = [new_x, y3, x4, y4, x4, y3, new_x, y4, ]
                        elif x3 < x2 < x4:
                            if abs(new_point[2] - new_point[4]) < diff_x:
                                new_box = new_point
                            else:
                                min_x = min(new_point[2], new_point[4])
                                new_x = x2 if min_x > x2 else min_x
                                new_box = [x3, y3, new_x, y3, new_x, y4, x3, y4, ]
                        else:
                            new_box = None
                        if new_box is not None:
                            filter_bboxs.append(new_box)

        logger.info(f"添加box到image: {len(filter_bboxs)} - 原始总共：{len(bbox_list)} 个box.")

        for index, box in enumerate(filter_bboxs):
            p = [int(o) for o in box]
            lt = Point(p[0], p[1])
            rt = Point(p[2], p[3])
            rb = Point(p[4], p[5])
            lb = Point(p[6], p[7])

            # 最右边的竖线
            if abs(rt.y - rb.y) < diff_y:
                new_y = round((rt.y + rb.y) / 2)
                rt.y = new_y
                rb.y = new_y

            # 水平线
            if abs(rt.y - lt.y) < diff_x:
                cv2.line(threshold, lt.to_cv(), rt.to_cv(), color, thickness)

            cv2.line(threshold, rt.to_cv(), rb.to_cv(), color, thickness)

            # 水平线
            if abs(rb.y - lb.y) < diff_x:
                cv2.line(threshold, rb.to_cv(), lb.to_cv(), color, thickness)

            cv2.line(threshold, lb.to_cv(), lt.to_cv(), color, thickness)

        return threshold

    @staticmethod
    def find_left_and_right_point(bboxes: List):
        """
        查找表格结构最左和最右的点

        :param bboxes:
        :return:
        """

    @staticmethod
    def check_pdf_text_need_rotate(texts: List, texts2: List):
        """
        判断pdf 是否需要旋转

        :param texts:
        :param texts2:
        :return:
        """
        # first_two_char = []
        # first_match = []
        # for index, text in enumerate(texts):
        #     if len(text) > 2:
        #         item = text[:2]
        #         first_two_char.append(item)
        #         if MatchUtils.match_pattern_flag(item, MatchUtils.PATTERN_PUNCTUATION2):
        #             first_match.append(item)
        #
        # total_two = len(first_two_char)
        # total_match = len(first_match)

        content = CommonUtils.remove_space("".join(texts))
        none_zh_content = "".join(MatchUtils.PATTERN_NONE_ZH.findall(content))

        content2 = CommonUtils.remove_space("".join(texts2))
        none_zh_content2 = "".join(MatchUtils.PATTERN_NONE_ZH.findall(content2))

        none_zh_radio1 = len(none_zh_content) / len(content)
        none_zh_radio2 = len(none_zh_content2) / len(content2)

        logger.info(f"判断pdf是否需要旋转: {none_zh_radio1} - {len(none_zh_content)} -{len(content)}")
        logger.info(f"判断pdf是否需要旋转: {none_zh_radio2} - {len(none_zh_content2)} -{len(content2)}")
        return none_zh_radio1 < none_zh_radio2

        # logger.info(f"判断pdf是否需要旋转: {total_two} - {total_match} -{total_match / total_two}")
        # if total_two > 5 and total_match / total_two > 0.2:
        #     return True

        # return False

    @staticmethod
    def build_table_cell_from_axis(bboxes: List, axis: List) -> List[Cell]:
        """
            通过逻辑坐标构造表格 cell

        :param bboxes:
        :param axis:
        :return:
        """
        pred_table = TableEval(bboxes=bboxes, axis=axis)
        return TableProcessUtils.build_table_cell_from_table_unit(pred_table.ulist)

    @staticmethod
    def build_table_cell_from_table_unit(table_units: List[TableUnit], text="test_text") -> List[Cell]:
        """
        table unit 转 table cell

        :param table_units:
        :param text:
        :return:
        """
        table_cells = []
        if len(table_units) == 0:
            return table_cells
        table_lt = table_units[0].bbox.point1[0]
        table_rb = table_units[-1].bbox.point3[0]
        table_width = table_rb[0] - table_lt[0]
        table_height = table_rb[1] - table_lt[1]
        for item in table_units:
            x1, y1 = item.bbox.point4[0]
            x2, y2 = item.bbox.point2[0]
            row_index = item.top_idx + 1
            col_index = item.left_idx + 1
            cell = Cell(x1, y1, x2, y2, row_index=row_index, col_index=col_index)
            cell.lt = item.bbox.point1[0]
            cell.rb = item.bbox.point3[0]
            cell.row_span = item.bottom_idx - item.top_idx + 1
            cell.col_span = item.right_idx - item.left_idx + 1
            cell.text = text

            cell.width_ratio = item.bbox.get_col_span() / table_width
            cell.height_ratio = item.bbox.get_row_span() / table_height

            table_cells.append(cell)
        return table_cells

    @staticmethod
    def check_pdf_text_need_rotate90(det_result):
        """
        根据 检测结果判断是否需要旋转90度

        :param det_result:
        :return:
        """
        widths = []
        heights = []
        for i in range(det_result.shape[0]):
            pts = OcrCommonUtils.order_point(det_result[i])
            width = abs(pts[0][0] - pts[2][0])
            height = abs(pts[0][1] - pts[2][1])
            widths.append(width)
            heights.append(height)

        radio = sum(widths) / sum(heights)

        flag = False
        if radio < 1:
            flag = True

        logger.info(f"根据检测结果判断是否需要旋转90度: {flag}- {radio} - {sum(widths)} - {sum(heights)}")
        return flag

    @staticmethod
    def get_table_cell_from_table_logit(table_bboxs, logits, save_html_file, add_end="_table_structure.html") -> List[
        Cell]:
        """
        table logit to table cell

        :param table_bboxs:
        :param logits:
        :param save_html_file:
        :param add_end:
        :return:
        """
        table_cells = TableProcessUtils.build_table_cell_from_axis(bboxes=table_bboxs, axis=logits)
        if save_html_file is None:
            return table_cells

        TableProcessUtils.table_cell_to_html_show(table_cells=table_cells,
                                                  save_html_file=save_html_file,
                                                  add_end=add_end)

        return table_cells

    @staticmethod
    def table_cell_to_html_show(table_cells: List[Cell], save_html_file, add_end="_table_structure.html"):
        if save_html_file.endswith(".html"):
            add_end = ""

        try:
            table_html, db_table_html = TableProcessUtils.cell_to_html(table_cells=table_cells)
            table_html_str = "\n".join(table_html) + "\n"
            save_html_file = f"{save_html_file}{add_end}"
            FileUtils.save_to_text(save_html_file, table_html_str)

            save_db_html_file = save_html_file.replace(".html", "_db.html")
            FileUtils.save_to_text(save_db_html_file, "\n".join(db_table_html) + "\n")
            logger.info(f"保存table structure to cell html：{save_html_file}")
        except Exception as e:
            traceback.print_exc()

        return save_html_file

    @staticmethod
    def modify_cell_info(all_cells: List[Cell], cols, rows,
                         one_col_width, one_row_height,
                         is_pdf=False, ):
        """
        修改cell 的属性

        :param all_cells:
        :param cols:
        :param rows:
        :param one_col_width:
        :param one_row_height:
        :param is_pdf:
        :return:
        """
        col_map = {col: index + 1 for index, col in enumerate(cols)}

        # pdf 和image 的y 轴相反
        rows_reverse = copy.deepcopy(rows)
        # rows_reverse.reverse()
        row_map = {row: index + 1 for index, row in enumerate(rows_reverse)}
        if not is_pdf:
            sort_cell_key = lambda x: (-x.y1, x.x1)
        else:
            sort_cell_key = lambda x: (x.y1, x.x1)

        new_all_cells = sorted(all_cells, key=sort_cell_key)
        for cell in new_all_cells:
            try:
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
            except Exception as e:
                # traceback.print_exc()
                pass
        logger.info(f"new_all_cells: {len(new_all_cells)}")

        return new_all_cells

    @staticmethod
    def save_table_join_point(image, table_bbox_unscaled, save_image_file,
                              pdf_images: List = None,
                              image_scalers=None,
                              thickness=2):
        """
         保存表格交点图像

        :param image:
        :param table_bbox_unscaled:
        :param save_image_file:
        :param pdf_images:
        :param image_scalers:
        :param thickness:
        :return:
        """
        src_im = copy.deepcopy(image)

        color_list = CommonUtils.get_color_list()
        for key, val in table_bbox_unscaled.items():
            if image_scalers:
                x1, y1, x2, y2 = MathUtils.scale_pdf(key, image_scalers)
            else:
                point = [round(item) for item in key]
                x1, y1, x2, y2 = point[0:4]

            cv2.rectangle(src_im, (x1, y1), (x2, y2), color_list[0], thickness)

            for index, point in enumerate(val):
                if image_scalers:
                    point = MathUtils.scale_point(point, image_scalers)
                new_point = (round(point[0]), round(point[1]))
                cv2.circle(src_im, new_point, 20, color_list[1], thickness)

        # draw pdf images
        if pdf_images is not None and len(pdf_images) > 0:
            for image in pdf_images:
                if image_scalers:
                    table_bbox = image["bbox"]
                    table_bbox_new = (table_bbox[0], table_bbox[3], table_bbox[2], table_bbox[1])
                    bbox = MathUtils.scale_pdf(table_bbox_new, image_scalers)
                    image["image_bbox"] = bbox
                    image["image_bbox_joint_point"] = [
                        [bbox[0], bbox[1]],
                        [bbox[0], bbox[3]],
                        [bbox[2], bbox[3]],
                        [bbox[2], bbox[1]],
                    ]

                x1, y1, x2, y2 = image["image_bbox"]
                cv2.rectangle(src_im, (x1, y1), (x2, y2), color_list[2], thickness)

                bbox_joint_point = image["image_bbox_joint_point"]
                for index, point in enumerate(bbox_joint_point):
                    cv2.circle(src_im, point, 10, color_list[3], thickness)

        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, src_im)

        logger.info(f"保存表格交点图像：{save_image_file}")

    @staticmethod
    def convert_table_sep_to_merge(outputs, result0):
        """
        将单独识别的表格结构转换成merge

        :param outputs:
        :return:
        """
        # return None
        results = []

        polygons_list = []
        logi_list = []
        structure_list = []
        logi_sep = []
        polygons_sep = []

        table_cells = []
        for one_result in outputs:
            table_bbox = one_result[0]
            x = round(table_bbox[0])
            y = round(table_bbox[1])

            table_result = one_result[1]
            structure_str_list = table_result.get("structure_str_list", [])
            structure_list.append(structure_str_list)

            if "logi" in table_result:
                logi_list.extend(table_result["logi"].tolist())
                logi_sep.append(table_result["logi"])

            polygons = OcrCommonUtils.box_list_move_point(table_result["polygons"], x=x, y=y)
            polygons_list.extend(polygons)
            try:
                polygons_sep.append(np.array(polygons))
            except Exception as e:
                pass

            if "table_cells" in table_result and table_result["table_cells"] is not None and len(
                    table_result["table_cells"]) > 0:
                one_table_cells = table_result["table_cells"][0]
                for k, rows in one_table_cells["table_cells"].items():
                    for cell in rows:
                        cell.x1 += x
                        cell.y1 += y
                        cell.x2 += x
                        cell.y2 += y

                new_table_cell = {
                    "index": len(table_cells),
                    "is_image": one_table_cells["is_image"],
                    "bbox": OcrCommonUtils.box_move_point(one_table_cells["bbox"], x=x, y=y),
                    "html": one_table_cells["html"],
                    "table_bbox": OcrCommonUtils.box_list_move_point(one_table_cells["table_bbox"], x=x, y=y),
                    # "table_image_bbox": OcrCommonUtils.box_list_move_point(one_table_cells["table_image_bbox"], x=x, y=y),
                    # "table_image_axis": one_table_cells["table_image_axis"],
                    "table_cells": one_table_cells["table_cells"]
                }
                table_cells.append(new_table_cell)
            else:
                raw_table_cells = {1: [Cell(x1=box[0],
                                            y1=box[1],
                                            x2=box[4] if len(box) > 5 else box[2],
                                            y2=box[5] if len(box) > 5 else box[3],
                                            col_index=1, row_index=1) for box in polygons]}
                new_table_cell = {
                    "index": len(table_cells),
                    "is_image": False,
                    "not_spit_text": True,
                    "bbox": table_bbox,
                    "html": structure_str_list,
                    "table_bbox": np.array(polygons),
                    "table_cells": raw_table_cells
                }
                table_cells.append(new_table_cell)

        result = {
            'polygons': np.array(polygons_list),
            'structure_str_list': structure_list,
            'logi': np.array(logi_list),
            'polygons_sep': polygons_sep,
            'logi_sep': logi_sep,
            # "inputs": inputs["inputs"],
            "table_cells": table_cells,
            "table_cell_metric": {},
        }

        results.append(result)

        return results
