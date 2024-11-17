#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_entity
# @Author  ：cycloneboy
# @Date    ：20xx/7/21 17:49
import os
import re
import traceback
from dataclasses import dataclass
from typing import Optional, List, Any, Dict

import numpy as np
import scipy
from pydantic import BaseModel
from transformers.utils import ModelOutput

from pdftable.entity.enum_entity import LineDirectionType, HtmlContentType, PdfLineType

"""
表格识别实体
"""

__all__ = [
    "Point",
    "Line",
    "OcrCell",
    "TableBBox",
    "TableUnit",
    "TableEval",
    "TableLabelRequest",
    "TablePageRequest",
    "TableContentResponse",
    "TablePageResult",
    "TableResponse",
    "TableListResponse",
    "ImagePreProcessOutput",
]


class Point(object):

    def __init__(self, x, y, is_joint=False):
        self.x = x
        self.y = y
        self.is_joint = is_joint

    def __repr__(self):
        x = round(self.x)
        y = round(self.y)

        return f"<Point x={x} y1={y} joint={self.is_joint}>"

    def to_point(self):
        msg = f"({self.x}, {self.y})"
        return msg

    def to_cv(self):
        return (self.x, self.y)

    def key(self):
        return f"{round(self.x)}_{round(self.y)}"

    @staticmethod
    def scale(point, factors):
        scaling_factor_x, scaling_factor_y, img_y = factors
        x = point.x * scaling_factor_x
        y = abs(point.y - img_y) * scaling_factor_x
        new_point = Point(x=x, y=y, is_joint=point.is_joint)
        return new_point


class LineInterval(object):
    def __init__(self, start, end):
        self.start = min(start, end)
        self.end = max(start, end)

    def __repr__(self):
        return f"<LineInterval [{self.start}, {self.end}] >"

    @staticmethod
    def merge_two_line(a, b):
        return LineInterval(min(a.start, b.start), max(a.end, b.end))

    @staticmethod
    def have_interaction(a, b):
        return max(a.start, b.start) <= min(a.end, b.end)

    @staticmethod
    def merge_line(line_list: List):
        if len(line_list) == 0:
            return []

        intervals = sorted(line_list, key=lambda x: x.start)
        result = []
        for interval in intervals:
            if len(result) == 0 or result[-1].end < interval.start:
                result.append(interval)
            else:
                result[-1].end = max(result[-1].end, interval.end)
        return result


class Line(object):

    def __init__(self, left: Point, right: Point,
                 direction: LineDirectionType = LineDirectionType.NONE, width=0, height=0):
        self.left = left
        self.right = right
        self.direction = direction
        self.width = width
        self.height = height

    def __repr__(self):
        return f"<Line left={self.left.to_point()} right={self.right.to_point()} direction={self.direction}>"

    @property
    def min_x(self):
        x = min(self.left.x, self.right.x)
        return x

    @property
    def max_x(self):
        x = max(self.left.x, self.right.x)
        return x

    @property
    def min_y(self):
        y = min(self.left.y, self.right.y)
        return y

    @property
    def max_y(self):
        y = max(self.left.y, self.right.y)
        return y

    @property
    def line_width(self):
        length = self.max_x - self.min_x
        return length

    @property
    def line_height(self):
        length = self.max_y - self.min_y
        return length

    @staticmethod
    def scale(line, factors):
        left = Point.scale(line.left, factors=factors)
        right = Point.scale(line.right, factors=factors)
        new_line = Line(left=left, right=right, direction=line.direction, width=line.width, height=line.height)
        return new_line

    @staticmethod
    def merge_two_line_vertical(line1, line2):
        left = Point(x=line1.min_x, y=min(line1.min_y, line2.min_y))
        right = Point(x=left.x, y=max(line1.max_y, line2.max_y))

        height = abs(right.y - left.y)
        one_line = Line(left=left, right=right,
                        direction=LineDirectionType.VERTICAL,
                        width=line1.width,
                        height=height)
        return one_line

    @staticmethod
    def merge_two_line_horizontal(line1, line2):
        left = Point(x=min(line1.min_x, line2.min_x), y=line1.min_y)
        right = Point(x=max(line1.min_x, line2.min_x), y=left.y)

        width = abs(right.x - left.x)
        one_line = Line(left=left, right=right,
                        direction=LineDirectionType.HORIZONTAL,
                        width=width,
                        height=line1.height)
        return one_line

    @staticmethod
    def merge_two_line(line1, line2, direction=LineDirectionType.HORIZONTAL):
        if direction == LineDirectionType.HORIZONTAL:
            return Line.merge_two_line_horizontal(line1=line1, line2=line2)
        else:
            return Line.merge_two_line_vertical(line1=line1, line2=line2)

    @staticmethod
    def line_can_vertical_merge(line1, line2, diff=2, ):
        intersect_flag = False

        if line2.max_y < line1.min_y - diff or line2.min_y > line1.max_y + diff:
            return False

        if line2.min_y - diff <= line1.min_y < line1.max_y <= line2.max_y + diff:
            # 刚好靠中间
            return True
        elif line2.min_y - diff < line1.min_y <= line2.max_y + diff:
            # line2靠左
            return True
        elif line1.min_y - diff < line2.min_y < line1.max_y + diff:
            # line2靠右
            return True

        return intersect_flag

    @staticmethod
    def line_can_horizontal_merge(line1, line2, diff=2, ):
        intersect_flag = False

        if line2.max_x < line1.min_x - diff or line2.min_x > line1.max_x + diff:
            return False

        if line2.min_x - diff <= line1.min_x < line1.max_x <= line2.max_x + diff:
            # 刚好靠中间
            return True
        elif line2.min_x - diff < line1.min_x <= line2.max_x + diff:
            # line2靠左
            return True
        elif line1.min_x - diff < line2.min_x < line1.max_x + diff:
            # line2靠右
            return True

        return intersect_flag

    @staticmethod
    def line_can_merge(line1, line2, diff=2, direction=LineDirectionType.HORIZONTAL):
        if direction == LineDirectionType.HORIZONTAL:
            return Line.line_can_horizontal_merge(line1=line1, line2=line2, diff=diff)
        else:
            return Line.line_can_vertical_merge(line1=line1, line2=line2, diff=diff)

    @staticmethod
    def merge_lines(vertical_lines: List, diff=2, direction=LineDirectionType.HORIZONTAL) -> List:
        """
        垂直线合并

        :param vertical_lines:
        :param diff:
        :param direction:
        :return:
        """
        if len(vertical_lines) == 0:
            return []

        if direction == LineDirectionType.HORIZONTAL:
            sort_key = lambda line: line.min_x
        else:
            sort_key = lambda line: line.min_y
        vertical_lines.sort(key=sort_key)

        result = []
        last_line = vertical_lines[0]
        for index in range(1, len(vertical_lines)):
            next_line = vertical_lines[index]
            if Line.line_can_merge(last_line, next_line, diff=diff, direction=direction):
                last_line = Line.merge_two_line(last_line, next_line, direction=direction)
            else:
                result.append(last_line)
                last_line = next_line

        result.append(last_line)
        return result


class OcrCell(object):

    def __init__(self, left_top: Point = None, right_bottom: Point = None, text=None, raw_data=None,
                 db_text=None,
                 cell_type=HtmlContentType.NONE, inner_cells=None):
        self.left_top = left_top
        self.right_bottom = right_bottom
        self.index = None
        self.text = text
        self.db_text = db_text
        self.bbox = None
        self.cell_type = cell_type
        self.is_image = False
        self.image_info = None

        self.text_number = 0
        self.text_width = 0

        self.line_type: PdfLineType = PdfLineType.NONE
        self.start_diff_error = 10

        # 内部包括的cell
        self.inner_cells = [] if inner_cells is None else inner_cells

        self.raw_data = raw_data
        if raw_data is not None:
            self.parse(raw_data)

        self.parse_width()

    def parse(self, raw_data):
        self.index = raw_data.get('index', None)
        self.text = raw_data.get('text', None)
        self.bbox = raw_data.get('bbox', None)
        self.is_image = raw_data.get('is_image', False)
        if self.is_image:
            self.cell_type = HtmlContentType.IMAGE
            self.image_info = raw_data.get('image_info', None)
        if self.bbox is not None:
            points = np.squeeze(self.bbox, axis=None)
            min_x = min(points[:, 0])
            min_y = min(points[:, 1])
            max_x = max(points[:, 0])
            max_y = max(points[:, 1])

            self.left_top = Point(x=min_x, y=min_y)
            self.right_bottom = Point(x=max_x, y=max_y)

    def to_json_save(self):
        result = {
            "index": self.index,
            "text": self.text,
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
        }
        return result

    def __repr__(self):
        text_show = self.text.replace('"', "“") if self.text is not None else ""
        return f"<OcrCell lt={self.left_top.to_point()} " \
               f"rb={self.right_bottom.to_point()} " \
               f"cell_type={self.cell_type.desc} " \
               f"line_type={self.line_type.desc} " \
               f"text={text_show}>"

    @property
    def x1(self):
        x = self.left_top.x
        return x

    @property
    def y1(self):
        y = self.left_top.y
        return y

    @property
    def x2(self):
        x = self.right_bottom.x
        return x

    @property
    def y2(self):
        y = self.right_bottom.y
        return y

    @property
    def max_x(self):
        x = max(self.left_top.x, self.right_bottom.x)
        return x

    @property
    def min_x(self):
        x = min(self.left_top.x, self.right_bottom.x)
        return x

    @property
    def max_x(self):
        x = max(self.left_top.x, self.right_bottom.x)
        return x

    @property
    def min_y(self):
        y = min(self.left_top.y, self.right_bottom.y)
        return y

    @property
    def max_y(self):
        y = max(self.left_top.y, self.right_bottom.y)
        return y

    @property
    def width(self):
        length = self.max_x - self.min_x
        return length

    @property
    def height(self):
        length = self.max_y - self.min_y
        return length

    @property
    def center_point(self):
        point = Point(round((self.min_x + self.max_x) / 2.0), round((self.min_y + self.max_y) / 2.0))
        return point

    def to_bbox(self):
        bbox = [self.left_top.x, self.left_top.y, self.right_bottom.x, self.right_bottom.y]
        return bbox

    def get_text(self):
        return self.text

    def to_key(self):
        bbox = self.to_bbox()
        key = "_".join([str(x) for x in bbox])
        return key

    def parse_width(self):
        if self.left_top is not None and self.right_bottom is not None:
            if self.text is not None:
                raw_text = str(self.text).strip("\n")
                self.text_number = len(list(raw_text))
                if raw_text.endswith(" ") and self.text_number > 0:
                    self.text_number -= 1

                if self.text_number > 0:
                    self.text_width = self.width / self.text_number

    def reset_line_type(self):
        self.line_type: PdfLineType = PdfLineType.NONE

    def parse_line_type(self, min_x, max_x, normal_font_size, reset_first=True):
        """
            判断是否是段落开始
                - 开始：跳过2个字符 ， 超过10
                - 结束
        :param min_x:
        :param max_x:
        :param normal_font_size:
        :param reset_first:
        :return:
        """
        if reset_first:
            self.reset_line_type()

        ads_start = abs(self.x1 - min_x)
        ads_end = abs(self.x2 - max_x)

        start_skip_font = ads_start / normal_font_size
        end_skip_font = ads_end / normal_font_size
        if ads_start >= self.start_diff_error and 1 <= start_skip_font:
            self.line_type = PdfLineType.PARAGRAPH_START

        if self.line_type == PdfLineType.NONE:
            if ads_start < self.start_diff_error < ads_end and 1 <= end_skip_font:
                self.line_type = PdfLineType.PARAGRAPH_END

        if self.line_type == PdfLineType.NONE:
            if ads_start < self.start_diff_error and ads_end < self.start_diff_error \
                    or (start_skip_font < 1 and end_skip_font < 1):
                self.line_type = PdfLineType.PARAGRAPH_MIDDLE

        if self.line_type == PdfLineType.NONE:
            #  中间状态
            if ads_start >= self.start_diff_error and start_skip_font > 2 \
                    and ads_end > self.start_diff_error and end_skip_font > 2:
                self.line_type = PdfLineType.PARAGRAPH_START

    def is_text(self):
        return self.cell_type == HtmlContentType.TXT

    def get_image_info(self, key, default=None):
        result = None
        if self.is_image:
            result = self.image_info.get(key, default)
        return result

    def get_image_name(self):
        return self.get_image_info('raw_name')

    def get_image_path(self):
        return self.get_image_info('save_name')

    def get_image_relative_path(self):
        return self.get_image_info('relative_dir')

    def get_image_width(self):
        return self.get_image_info('width')

    def get_image_height(self):
        return self.get_image_info('height')

    def get_show_html(self):
        text = self.get_text_to_show()
        if self.cell_type == HtmlContentType.TABLE:
            table = "".join(text)
            contents = ["\n<div class='div-table'>\n", table, "\n</div>\n"]
        elif self.cell_type == HtmlContentType.IMAGE:
            image_html = (f'<div class="div-image" align="center">'
                          f'{text}'
                          f'</div>')
            contents = [image_html]
        else:
            contents = ["<p>", text, "</p>"]

        content = "".join(contents)
        return content

    def get_text_to_show(self):
        text = self.get_text()
        if self.cell_type == HtmlContentType.IMAGE:
            image_text = "".join(text)
            image_path = self.get_image_path() if self.image_info is not None else image_text
            image_show = self.get_image_relative_path()
            # image_html = (f'<img src="{image_show}" '
            #               f'width="{self.get_image_width()}" '
            #               f'height="{self.get_image_height()}"  >')
            image_html = (f'<div "height: {self.get_image_height()}px">'
                          f'<img class="pdf-image" src="{image_show}" '
                          f'"height:100%" data-height="{self.get_image_height()}" '
                          f'data-width="{self.get_image_width()}" /></div>')

            contents = [image_html]
        else:
            contents = [text]

        content = "".join(contents)
        return content


class TableBBox(object):

    def __init__(self, bbox):
        self._bbox = bbox
        self.point1 = np.array([[bbox[0], bbox[1]]])
        self.point2 = np.array([[bbox[2], bbox[3]]])
        self.point3 = np.array([[bbox[4], bbox[5]]])
        self.point4 = np.array([[bbox[6], bbox[7]]])

        self.col_span = (self.computing_span(self.point1, self.point2) + self.computing_span(self.point3,
                                                                                             self.point4)) / 2
        self.row_span = (self.computing_span(self.point1, self.point4) + self.computing_span(self.point2,
                                                                                             self.point3)) / 2

    def computing_span(self, pointa, pointb):
        span = scipy.spatial.distance.cdist(pointa, pointb, metric="euclidean")
        return span

    def get_col_span(self):
        return self.col_span[0][0]

    def get_row_span(self):
        return self.row_span[0][0]

    def __repr__(self):
        x1 = round(self.point1[0][0])
        y1 = round(self.point1[0][1])
        x2 = round(self.point3[0][0])
        y2 = round(self.point3[0][1])
        return (f"<TableBBox x1={x1} y1={y1} x2={x2} y2={y2} " \
                f"span=[{round(self.get_row_span())},{round(self.get_col_span())}" \
                f">")


class TableUnit(object):
    def __init__(self, bbox, axis):
        self.bbox = TableBBox(bbox)
        self.axis = axis
        self.top_idx = axis[2]
        self.bottom_idx = axis[3]
        self.left_idx = axis[0]
        self.right_idx = axis[1]

    def __repr__(self):
        bbox_str = str(self.bbox)
        return (f"<TableUnit bbox_str={bbox_str} " \
                f"axis=[{self.axis}" \
                f">")

    def to_json(self):
        target = {
            "bbox": self.bbox._bbox,
            "axis": self.axis
        }
        return target


class TableEval(object):
    def __init__(self, bbox_path=None, axis_path=None, file_name=None, bboxes=None, axis=None):
        self.bboxes = bboxes
        self.axis = axis
        self.bbox_path = bbox_path
        self.axis_path = axis_path
        self.file_name = file_name

        self.ulist: List[TableUnit] = []
        self.read_bbox()

        self.ulist = TableEval.bubble_sort(self.ulist)

    def read_bbox(self):
        if self.bboxes is None and self.axis is None:
            bbox_dir = os.path.join(self.bbox_path, self.file_name)
            axis_dir = os.path.join(self.axis_path, self.file_name)

            self.ulist = TableEval.load_tabu(bbox_dir, axis_dir)
        else:
            for bbox, axis in zip(self.bboxes, self.axis):
                unit = TableUnit(bbox, axis)
                self.ulist.append(unit)

    @staticmethod
    def load_tabu(bbox_dir, axis_dir):
        unit_list = []
        f_b = open(bbox_dir)
        f_a = open(axis_dir)
        bboxs = f_b.readlines()
        axiss = f_a.readlines()

        for bbox, axis in zip(bboxs, axiss):
            try:
                bbox = list(map(float, re.split(';|,', bbox.strip())))
                axis = list(map(int, axis.strip().split(',')))
                unit = TableUnit(bbox, axis)

                unit_list.append(unit)
            except Exception as e:
                traceback.print_exc()
                print(f"error: {bbox_dir}")
        return unit_list

    @staticmethod
    def compute_iou(bbox1, bbox2):
        rec1 = (bbox1.point1[0][0], bbox1.point1[0][1], bbox1.point3[0][0], bbox1.point3[0][1])
        rec2 = (bbox2.point1[0][0], bbox2.point1[0][1], bbox2.point3[0][0], bbox2.point3[0][1])
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])

        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            return S_cross / (S1 + S2 - S_cross)

    @staticmethod
    def bubble_sort(unit_list: List[TableUnit]) -> List[TableUnit]:
        length = len(unit_list)
        for index in range(length):
            for j in range(1, length - index):
                if TableEval.is_priori(unit_list[j], unit_list[j - 1]):
                    unit_list[j - 1], unit_list[j] = unit_list[j], unit_list[j - 1]
        return unit_list

    @staticmethod
    def is_priori(unit_a: TableUnit, unit_b: TableUnit) -> bool:
        if unit_a.top_idx < unit_b.top_idx:
            return True
        elif unit_a.top_idx > unit_b.top_idx:
            return False
        if unit_a.left_idx < unit_b.left_idx:
            return True
        elif unit_a.left_idx > unit_b.left_idx:
            return False
        if unit_a.bottom_idx < unit_b.bottom_idx:
            return True
        elif unit_a.bottom_idx > unit_b.bottom_idx:
            return False
        if unit_a.right_idx < unit_b.right_idx:
            return True
        elif unit_a.right_idx > unit_b.right_idx:
            return False


class TableLabelRequest(BaseModel):
    run_item: Optional[Any] = None
    file_name: Optional[str] = None
    file_dir: Optional[str] = None
    src_id: Optional[int] = None
    page: str


class TablePageRequest(BaseModel):
    id: int
    content: Optional[str] = None
    new_content: Optional[str] = None
    src_id: Optional[int] = None
    page: Optional[str] = None
    source: Optional[int] = None
    table_metric: Optional[Dict[str, Any]] = None


class TablePageResult(BaseModel):
    id: int
    src_id: Optional[int] = None
    run_time: Optional[str] = None
    current_page: Optional[int] = None
    current_number: Optional[int] = None
    content: Optional[str]
    new_content: Optional[str] = None
    file_dir: Optional[str] = None
    run_metric: Optional[Dict] = None


class TableContentResponse(BaseModel):
    src_id: Optional[int] = None
    page: Optional[str] = None
    object: str = "list"
    tables: List[TablePageResult]


class TableListResponse(BaseModel):
    src_id: Optional[int] = None
    page: Optional[str] = None
    object: str = "list"
    data: List[Any] = None


class TableResponse(BaseModel):
    object: str = "success"
    message: Optional[str] = None
    code: int = 200


@dataclass
class ImagePreProcessOutput(ModelOutput):
    """
    image 预处理 输出结果
    """
    image_orientation: str = None
    image_orientation_score: float = None
    table_attribute: List[str] = None
    table_attribute_score: List[int] = None

    def check_rotate(self):
        if self.image_orientation_score is not None and self.image_orientation_score > 0.6 \
                and self.image_orientation is not None and self.image_orientation not in ["0"]:
            return True

        return False

    def get_image_orientation(self):
        return {"angle": self.image_orientation, "score": self.image_orientation_score}
