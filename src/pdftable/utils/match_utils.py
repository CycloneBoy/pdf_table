#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：match_utils
# @Author  ：cycloneboy
# @Date    ：20xx/6/21 15:07
import re

from . import BaseUtil

"""
匹配工具类
"""


class MatchUtils(BaseUtil):
    """
    匹配工具类类

    """

    PATTERN_REPLACE_LAW_PAGE = re.compile(r"[—－_-]([0-9]{1,})[—－_-]")

    PATTERN_VIOLATION_FILTER_LINE_2 = re.compile(r"(【打印】|【关闭窗口】)")

    ENGLISH_PATTERN = re.compile(r"[a-zA-Z]")
    PATTERN_ENGLISH_AND_NUMBER = re.compile(r"[a-zA-Z0-9]")
    PUNCTUATION_PATTERN = re.compile(r"[#（）%]")

    DATE_PATTERN = re.compile(r"[年月日]")
    # [节次个号条款届]
    OTHER_PATTERN = re.compile(r"([%十\d])|(第[一二三四五六七八九十\d])")
    # 中英文数字
    PATTERN_NUMBER = re.compile(r'[一二三四五六七八九十\d]')
    # 中英文标点
    PATTERN_PUNCTUATION = re.compile(r'[#（）%“”*，：。、╮≥►〉]')
    PATTERN_PUNCTUATION_FILTER = re.compile(r'[：。“”]')
    PATTERN_OTHER_FILTER = re.compile(r'(指$|^《|》$|（[一二三四五六七八九十\d]）)')

    # 匹配非汉子
    PATTERN_NONE_ZH = re.compile(r"[^\u4e00-\u9fa5]")
    PATTERN_ZH = re.compile(r"[\u4e00-\u9fa5]+")

    # 知识库文件
    PATTERN_LC_KG_FILE = re.compile(r"\.(txt|md|csv|png|jpg|pdf)$", flags=re.I)

    PATTERN_LAW_HTML_IMAGE_1 = re.compile(r'(<img)(.*)(/>|</img>)')

    # OCR 后处理
    PATTERN_OCR_TEXT_0 = re.compile(r'[oO]')

    PATTERN_OCR_TEXT_ZH_NUMBER = re.compile(r'[-0-9\.,]')

    def init(self):
        pass

    @staticmethod
    def match_none_zh(text):
        """
        匹配中文汉子
        :param text:
        :return:
        """
        return MatchUtils.match_pattern_flag(text, MatchUtils.PATTERN_NONE_ZH)

    @staticmethod
    def match_pattern_extract(text, pattern, return_str=False):
        """
        匹配单个位置
        :param text:
        :param pattern:
        :param return_str:
        :return:
        """
        if isinstance(pattern, str):
            match_pattern = re.compile(pattern)
        else:
            match_pattern = pattern

        match_result = match_pattern.findall(text)
        if len(match_result) > 0:
            if return_str:
                res = str(match_result[0]).rstrip()
            else:
                res = match_result
        else:
            res = ""
        return res

    @staticmethod
    def match_pattern_flag(text, pattern):
        """
        匹配结果
        :param text:
        :param pattern:
        :return:
        """
        flag = False
        filter_error_type = len(MatchUtils.match_pattern_extract(text, pattern)) > 0
        if filter_error_type:
            flag = True
        return flag

    @staticmethod
    def match_pattern_result(text, pattern):
        """
        匹配
        :param text:
        :param pattern:
        :return:
        """
        match_result = MatchUtils.match_pattern_extract(text, pattern)

        flag = False
        filter_error_type = len(match_result) > 0
        if filter_error_type:
            flag = True
        return flag, match_result

    @staticmethod
    def match_pattern_list(texts, pattern_list):
        """
        匹配  正则列表
        """
        if not isinstance(pattern_list, list):
            pattern_list = [pattern_list]

        match_result = []
        for pattern in pattern_list:
            raw_match_result = pattern.findall(texts)
            if len(raw_match_result) > 0:
                match_result = raw_match_result
                break

        return match_result

    @staticmethod
    def clean_html_table_width(html_table):
        """
        移除表格属性

        :param html_table:
        :return:
        """
        html_res = re.sub(r'(width=["\']\d+%["\'])', "", html_table)
        html_res = re.sub(r'(width=['"']\d+%['"'])', "", html_res)
        html_res = re.sub(r'(align=["\']?center["\']?)', "", html_res)

        html_res = MatchUtils.PATTERN_LAW_HTML_IMAGE_1.sub("<img></img>", html_res)

        return html_res

    @staticmethod
    def remove_html_table_text(html_table):
        """
        移除表格文字 <td>7</td>

        :param html_table:
        :return:
        """
        html_res = re.sub(r'(>[^<>]*</td>)', ">测试</td>", html_table)
        return html_res

    @staticmethod
    def find_text_span_index(text: str, span: str, total=1):
        """
        找到span index

        :param text:
        :param span:
        :param total:
        :return:
        """
        begin_index = []

        all_total = text.count(span)
        if all_total == 0:
            return begin_index

        find_total = len(begin_index)

        if total < 0:
            total = all_total

        start_index = 0
        while find_total < min(all_total, total):
            index = text.find(span, start_index)
            find_span = text[index:index + 50]
            # logger.info(f"find_span: {find_span} - {index} - {start_index}")
            if index == -1:
                break

            start_index = index + len(span)
            begin_index.append(index)
            find_total = len(begin_index)

        return begin_index
