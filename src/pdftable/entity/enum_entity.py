#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : enum_entity.py
# @Author   : cycloneboy
# @Date     : 20xx/8/19 - 14:59
from enum import Enum, unique
from typing import Dict

__all__ = [
    "HtmlContentType",
    "HtmlTableCompareType",
    "LineDirectionType",
    "PdfLineType",
    "ModelType"
]


@unique
class HtmlContentType(Enum):
    """
    HTML 页面类型
    """

    TXT = "文本"
    TABLE = "表格"
    IMAGE = "图片"
    HYPERLINK = "超链接"
    NONE = "未知"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc

    @staticmethod
    def parse(task_type):
        for name, member in HtmlContentType.__members__.items():
            if str(task_type).lower() == str(member.desc).lower():
                return member
        return HtmlContentType.NONE


@unique
class HtmlTableCompareType(Enum):
    """
    HTML 表格对比结果类型
    """

    DIFF = "不同"
    SAME = "完全相同"
    REMOVE_WIDTH_SAME = "相同（移除宽度信息）"
    SAME_LABEL_MISSING_ONE_CHARACTER = "相同（标签缺少一个字符）"
    SAME_LABEL_GARBLED_ONE_CHARACTER = "相同（标签一个字符乱码）"

    DIFF_TEXT_ORDER = "不同（字序错乱）"
    DIFF_TEXT_INCONSISTENT = "不同（内容不一致）"
    DIFF_TEXT_PREDICT_LESS_WORDS = "不同（预测内容少字）"
    DIFF_TEXT_LABEL_LESS_WORDS = "不同（标签内容少字）"

    DIFF_CELL_SPAN_SAME = "相同（cell）"
    DIFF_CELL_ROW_SPAN = "不同（cell跨行）"
    DIFF_CELL_COL_SPAN = "不同（cell跨列）"
    DIFF_CELL_ROW_COL_SPAN = "不同（cell跨行又跨列）"

    DIFF_CELL_DIFF_ROW = "不同（cell不同行）"

    NONE = "未知"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc

    @staticmethod
    def parse(task_type):
        for name, member in HtmlTableCompareType.__members__.items():
            if str(task_type).lower() == str(name).lower():
                return member
        return HtmlTableCompareType.NONE

    @staticmethod
    def parse_desc(task_type):
        for name, member in HtmlTableCompareType.__members__.items():
            if str(task_type).lower() == member.desc.lower():
                return member
        return HtmlTableCompareType.NONE

    @staticmethod
    def mapping(task_type):
        for index, (name, member) in enumerate(HtmlTableCompareType.__members__.items()):
            if task_type == member:
                return index
        return 0

    @staticmethod
    def get_mapping() -> Dict:
        check_type_mapping = {
            "diff": 0,
            "same": 1,
            "remove_width_same": 2,
            "same_label_missing_one_character": 3,
            "same_label_garbled_one_character": 4,

            "diff_text_order": 5,
            "diff_text_inconsistent": 6,
            "diff_text_predict_less_words": 7,
            "diff_text_label_less_words": 8,

            "diff_cell_span_same": 9,
            "diff_cell_row_span": 10,
            "diff_cell_col_span": 11,
            "diff_cell_row_col_span": 12,
            "diff_cell_diff_row": 13,
            "none": 0,
        }
        return check_type_mapping

    def get_check_type(self):
        return self.name.lower()


@unique
class LineDirectionType(Enum):
    """
    线的方向
    """

    HORIZONTAL = "水平"
    VERTICAL = "垂直"
    NONE = "未知"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc

    @staticmethod
    def parse(task_type):
        for name, member in LineDirectionType.__members__.items():
            if str(task_type).lower() == str(member).lower():
                return member
        return LineDirectionType.NONE


@unique
class PdfLineType(Enum):
    """
    pdf 一行的类别
    """

    PARAGRAPH_START = "段落开始"
    PARAGRAPH_END = "段落结束"
    PARAGRAPH_MIDDLE = "段落中间"

    ALIGN_LEFT = "左对齐"
    ALIGN_RIGHT = "右对齐"
    ALIGN_CENTER = "中对齐"

    NONE = "未知"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc


class LayoutLabelEnum(Enum):
    """
    layout label enum
    """
    TEXT = "text"
    TITLE = "title"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    HEADER = "header"
    FOOTER = "footer"
    REFERENCE = "reference"
    EQUATION = "equation"
    LIST = "list"
    PAGE_NUMBER = "page_number"
    FOOTNOTE = "footnote"
    FULL_COLUMN = "full_column"
    SUB_COLUMN = "sub_column"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc

    def __str__(self):
        return "{}:{}".format(self.name, self.desc)


@unique
class ModelType(Enum):
    """
    Model name
    """

    LAYOUT_DOCX_LAYOUT = "DocXLayout"
    LAYOUT_PICODET = "picodet"

    TSR_CENTER_NET = "CenterNet"
    TSR_SLANet = "SLANet"
    TSR_LORE = "Lore"
    TSR_LGPMA = "Lgpma"
    TSR_MTL_TAB_NET = "MtlTabNet"
    TSR_TABLE_MASTER = "TableMaster"
    TSR_LINE_CELL = "LineCell"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc

    def __str__(self):
        return "{}:{}".format(self.name, self.desc)
