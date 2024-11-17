#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：pdf_utils
# @Author  ：cycloneboy
# @Date    ：20xx/6/21 14:30
import os
import random
import re
import shutil
import time
import traceback
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from operator import itemgetter
from typing import Tuple, List, Union, Dict

import cv2
import fitz
import numpy as np
import requests
from bs4 import BeautifulSoup
from pdfminer.converter import PDFPageAggregator
from pdfminer.image import ImageWriter
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTTextLineHorizontal,
    LTTextLineVertical,
    LTImage, LTPage, LTTextLine, LTRect,
)
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser
from pypdf import PdfReader, PdfWriter, PageObject

from pdftable.entity.table_entity import OcrCell, Line, Point
from . import BaseUtil, logger, FileUtils, TimeUtils, MathUtils, CommonUtils, Constants
from .table.image_processing import PdfImageProcessor
from ..entity import PdfLineType, HtmlContentType

'''
PDFDF文件处理的工具类

'''


class PdfUtils(BaseUtil):
    """
    文件工具类
    """

    def init(self):
        pass

    @staticmethod
    def get_user_agent():
        ua_list = [
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .\
            NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; InfoPath.3)', ]

        headers = {'User-Agent': random.choice(ua_list)}
        return headers

    @staticmethod
    def clean_sentence(sentence):
        "clean one line from pdf"
        # result = re.sub("\u3000*\n*", "", sentence).replace("\x82", "")
        result = re.sub(r"\x82", "", sentence)
        result = re.sub(r"\u200b", "", result)
        result = re.sub(r"\ue004", "", result)
        result = re.sub(r"\ue008", "", result)
        result = re.sub(r"\ue009", "", result)
        result = re.sub(r"\u3000", " ", result)
        result = result.strip("\r")
        result = result.strip("\n")
        result = result.strip("\r")

        return result

    @staticmethod
    def clean_sentence_space_line(content):
        """
        清洗
            - 移除空白行
        :param content:
        :return:
        """

        raw_text = []
        for item in content:
            item_clean = PdfUtils.clean_sentence(item)
            result = str(item_clean).replace(" ", "").replace("\r", "").replace("\n", "")
            if len(result) > 0:
                raw_text.append(item_clean)

        return raw_text

    @staticmethod
    def extract_use_pdfmup(file_name, show_number=50, keep_page_number=True):
        """
        提取pdf 中的文字

        :param file_name:
        :param show_number:
        :param keep_page_number:
        :return:
        """
        page_result = []
        page_metric = {}
        total_word = 0
        doc = fitz.open(file_name)
        for index, page in enumerate(doc):
            page_text = []
            """
            The first four entries are the block’s bbox coordinates, 
            block_type is 1 for an image block, 0 for text. 
            block_no is the block sequence number. 
            Multiple text lines are joined via line breaks.
            """

            # (x0, y0, x1, y1, "lines in the block", block_no, block_type)
            page1text = page.get_text("blocks", sort=True)
            # 只取文字 https://pymupdf.readthedocs.io/en/latest/textpage.html#TextPage.extractText

            page_text_raw = [i[-3:] for i in page1text if i[-1] == 0]
            page_text_raw.sort(key=lambda x: x[1])
            page_text_clean = [PdfUtils.clean_sentence(j[0]) for j in page_text_raw]

            for element in page_text_clean:
                texts = element
                if not keep_page_number and str(texts).strip() == str(index + 1):
                    continue

                total_word += len(texts)
                # print(texts)
                page_text.append(texts)
            page_result.append("\n".join(page_text))
            if index % show_number == 0:
                logger.info(f"page: {index} - page_text: {len(page_text)}")
            page_metric[index + 1] = len(page_text)
        return page_result, page_metric, total_word

    @staticmethod
    def extract_pdf_to_text(file_name, text_save_dir, metric_save_dir, show_number=50, keep_page_number=True):
        """
        提取 PDF文件 到txt

        :param file_name:
        :param text_save_dir:
        :param metric_save_dir:
        :param show_number:
        :param keep_page_number:
        :return:
        """
        logger.info(f"开始提取pdf2text: {file_name}")

        # temp_file_name = f"{FileUtils.get_parent_dir_name(file_name)}/{FileUtils.get_file_name(file_name)}"
        temp_file_name = FileUtils.get_file_name(file_name)
        if text_save_dir.endswith(".txt"):
            text_save_file = text_save_dir
        else:
            text_save_file = f"{text_save_dir}/{temp_file_name}.txt"

        metric_json_file = f"{metric_save_dir}/{temp_file_name}.json"

        eval_begin_time = TimeUtils.now_str()

        # page_result, page_metric, total_word = self.extract_use_pdfminer()

        page_result, page_metric, total_word = PdfUtils.extract_use_pdfmup(file_name=file_name,
                                                                           show_number=show_number,
                                                                           keep_page_number=keep_page_number)

        extract_success = False
        if len(page_result) > 1:
            extract_success = True

        eval_end_time = TimeUtils.now_str()
        all_metric = {
            "extract_success": extract_success,
            "pdf_file_path": file_name,
            "text_file_path": text_save_file,
            "metric_file_path": metric_json_file,
            "total_page": len(page_metric),
            "total_word": total_word,
            "avg_page_word": total_word / len(page_metric) if len(page_result) > 0 else 0,
            "eval_begin_time": eval_begin_time,
            "eval_end_time": eval_end_time,
            "use_time": TimeUtils.calc_diff_time(eval_begin_time, eval_end_time),
            "run_metric": page_metric
        }

        FileUtils.save_to_text(text_save_file, "\n".join(page_result))

        FileUtils.dump_json(metric_json_file, all_metric)

        logger.info(
            f"提取pdf:{'成功' if extract_success else '失败'} -  {file_name},耗时：{all_metric['use_time']} s text: {text_save_file} metric: {metric_json_file}")
        # logger.info(f"all_metric：{all_metric}")

        return extract_success, page_result, all_metric

    @staticmethod
    def download_pdf(file_url, file_name):
        """
        下载PDF 文件
        :param file_url:
        :param file_name:
        :return:
        """
        usetime = 0
        try:
            start_time = time.time()
            logger.info(f"开始下载文件: {file_url} -> {file_name}")
            FileUtils.check_file_exists(filename=file_name)
            headers = PdfUtils.get_user_agent()
            # urllib.request.urlretrieve(url=file_url, filename=file_name)

            r = requests.get(file_url, headers=headers, timeout=(10, 30))
            if r.status_code == 200 and r.content:
                with open(file_name, 'wb+') as f:
                    f.write(r.content)
            else:
                logger.info(f"文件不存在：{file_url}")

            end_time = time.time()
            usetime = end_time - start_time
            logger.info(f"下载完成一个文件,耗时：{usetime} -> {file_url}")

        except Exception as e:
            traceback.print_exc()
            logger.info(f"下载文件出错: {file_url} -> {file_name}")
            logger.error(e)

        return usetime

    @staticmethod
    def download_pdf_before_check(file_url, pdf_dir=None, file_name=None, ):
        """
        下载PDF
        :param file_url:
        :param pdf_dir:
        :param file_name:
        :return:
        """
        if file_url.startswith("http"):
            raw_file_name = FileUtils.get_raw_file_name(file_url, )
            pdf_file = f"{pdf_dir}/{raw_file_name}" if pdf_dir is not None else file_name

            if not FileUtils.check_file_exists(pdf_file):
                PdfUtils.download_pdf(file_url, pdf_file)
            return pdf_file
        else:
            return file_url

    @staticmethod
    def html_to_text_use_beautiful_soup(src_html):
        """
         HTML to plain text
        :param src_html:
        :return:
        """
        soup = PdfUtils.html_to_tag_list_use_beautiful_soup(src_html)
        src_text = soup.get_text()
        return src_text

    @staticmethod
    def html_to_tag_list_use_beautiful_soup(src_html) -> BeautifulSoup:
        """
         HTML to soup
        :param src_html:
        :return:
        """
        soup = BeautifulSoup(src_html, 'html.parser')
        [script.extract() for script in soup.findAll('script')]
        [style.extract() for style in soup.findAll('style')]

        return soup

    @staticmethod
    def html_to_text_use_scrapy_selector(src_html):
        """
         HTML to plain text
        :param src_html:
        :return:
        """
        response = HtmlSelector(text=src_html)
        src_text = response.xpath('//*/text()').extract()

        return src_text

    @staticmethod
    def html_to_text_list(src_html, use_beautiful_soup=True, use_scrapy_selector=False):
        """
        提取TXT
        :param src_html:
        :param use_beautiful_soup:
        :param use_scrapy_selector:
        :return:
        """
        if not src_html:
            return ""

        if use_beautiful_soup:
            src_text = PdfUtils.html_to_text_use_beautiful_soup(src_html)
            src_text_list = src_text.split('\n')
        else:
            src_text_list = PdfUtils.html_to_text_use_scrapy_selector(src_html)

        return src_text_list

    @staticmethod
    def html_to_text_pure(src_html, filter_space_line=False, origin_text_content=None) -> List:
        """
        HTML to plain text

        :param src_html:
        :param filter_space_line:
        :param origin_text_content:
        :return:
        """

        src_text_list_beautiful_soup = PdfUtils.html_to_text_list(src_html, use_beautiful_soup=True)

        src_text_list = src_text_list_beautiful_soup
        if filter_space_line:
            src_text_list = PdfUtils.clean_sentence_space_line(src_text_list)

        return src_text_list

    @staticmethod
    def get_page_layout_all(filename: str,
                            line_overlap: float = 0.5,
                            char_margin: float = 1.0,
                            line_margin: float = 0.5,
                            word_margin: float = 0.1,
                            boxes_flow: float = 0.5,
                            detect_vertical: bool = True,
                            all_texts: bool = True,
                            ) -> List[Tuple[LTPage, Tuple]]:
        """Returns a PDFMiner LTPage object and page dimension of a single
        page pdf. To get the definitions of kwargs, see
        https://pdfminersix.rtfd.io/en/latest/reference/composable.html.

        Parameters
        ----------
        :param filename :  Path to pdf file.
        :param line_overlap :
        :param char_margin :
        :param line_margin :
        :param word_margin :
        :param boxes_flow :
        :param detect_vertical :
        :param all_texts :

        :return
        -------
        layout : object
            PDFMiner LTPage object.
        dim : tuple
            Dimension of pdf page in the form (width, height).

        """
        with open(filename, "rb") as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)
            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed(
                    f"Text extraction is not allowed: {filename}"
                )
            laparams = LAParams(
                line_overlap=line_overlap,
                char_margin=char_margin,
                line_margin=line_margin,
                word_margin=word_margin,
                boxes_flow=boxes_flow,
                detect_vertical=detect_vertical,
                all_texts=all_texts,
            )
            rsrcmgr = PDFResourceManager()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            results = []
            for page in PDFPage.create_pages(document):
                interpreter.process_page(page)
                layout = device.get_result()
                width = layout.bbox[2]
                height = layout.bbox[3]
                dim = (width, height)
                results.append((layout, dim))
            return results

    @staticmethod
    def get_page_layout_first(filename: str,
                              line_overlap: float = 0.5,
                              char_margin: float = 1.0,
                              line_margin: float = 0.5,
                              word_margin: float = 0.1,
                              boxes_flow: float = 0.5,
                              detect_vertical: bool = True,
                              all_texts: bool = True,
                              ) -> Tuple[LTPage, Tuple]:
        """Returns a PDFMiner LTPage object and page dimension of a single
        page pdf. To get the definitions of kwargs, see
        https://pdfminersix.rtfd.io/en/latest/reference/composable.html.

        Parameters
        ----------
        :param filename :  Path to pdf file.
        :param line_overlap :
        :param char_margin :
        :param line_margin :
        :param word_margin :
        :param boxes_flow :
        :param detect_vertical :
        :param all_texts :

        :return
        -------
        layout : object
            PDFMiner LTPage object.
        dim : tuple
            Dimension of pdf page in the form (width, height).

        """
        with open(filename, "rb") as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)
            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed(
                    f"Text extraction is not allowed: {filename}"
                )
            laparams = LAParams(
                line_overlap=line_overlap,
                char_margin=char_margin,
                line_margin=line_margin,
                word_margin=word_margin,
                boxes_flow=boxes_flow,
                detect_vertical=detect_vertical,
                all_texts=all_texts,
            )
            rsrcmgr = PDFResourceManager()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            results = []
            for index, page in enumerate(PDFPage.create_pages(document)):
                interpreter.process_page(page)
                layout = device.get_result()
                width = layout.bbox[2]
                height = layout.bbox[3]
                dim = (width, height)
                results.append((layout, dim))
                break
            return results[0]

    @staticmethod
    def get_page_layout(filename: str,
                        line_overlap: float = 0.5,
                        char_margin: float = 1.0,
                        line_margin: float = 0.5,
                        word_margin: float = 0.1,
                        boxes_flow: float = 0.5,
                        detect_vertical: bool = True,
                        all_texts: bool = True,
                        ) -> Tuple[LTPage, Tuple]:
        results = PdfUtils.get_page_layout_first(filename, line_overlap, char_margin,
                                                 line_margin, word_margin, boxes_flow,
                                                 detect_vertical, all_texts)

        return results

    @staticmethod
    def get_text_objects(layout: LTPage, ltype: str = "char", t=None):
        """Recursively parses pdf layout to get a list of
        PDFMiner text objects.

        Parameters
        ----------
        layout : object
            PDFMiner LTPage object.
        ltype : string
            Specify 'char', 'lh', 'lv' to get LTChar, LTTextLineHorizontal,
            and LTTextLineVertical objects respectively.
        t : list

        Returns
        -------
        t : list
            List of PDFMiner text objects.

        """
        if ltype == "char":
            LTObject = LTChar
        elif ltype == "image":
            LTObject = LTImage
        elif ltype == "horizontal_text":
            LTObject = LTTextLineHorizontal
        elif ltype == "vertical_text":
            LTObject = LTTextLineVertical
        elif ltype == "rect":
            LTObject = LTRect
        if t is None:
            t = []
        try:
            for obj in layout._objs:
                if isinstance(obj, LTObject):
                    t.append(obj)
                else:
                    t += PdfUtils.get_text_objects(obj, ltype=ltype)
        except AttributeError:
            pass
        return t

    @staticmethod
    def get_rotation(chars: List, horizontal_text: List, vertical_text: List):
        """Detects if text in table is rotated or not using the current
        transformation matrix (CTM) and returns its orientation.

        Parameters
        ----------
        horizontal_text : list
            List of PDFMiner LTTextLineHorizontal objects.
        vertical_text : list
            List of PDFMiner LTTextLineVertical objects.
        ltchar : list
            List of PDFMiner LTChar objects.

        Returns
        -------
        rotation : string
            '' if text in table is upright, 'anticlockwise' if
            rotated 90 degree anticlockwise and 'clockwise' if
            rotated 90 degree clockwise.

        """
        rotation = ""
        hlen = len([t for t in horizontal_text if t.get_text().strip()])
        vlen = len([t for t in vertical_text if t.get_text().strip()])
        if hlen < vlen:
            clockwise = sum(t.matrix[1] < 0 and t.matrix[2] > 0 for t in chars)
            anticlockwise = sum(t.matrix[1] > 0 and t.matrix[2] < 0 for t in chars)
            rotation = "anticlockwise" if clockwise < anticlockwise else "clockwise"
        return rotation

    @staticmethod
    def read_pdf(file_name, password=None) -> PdfReader:
        """
        read a PDF file

        :param file_name:
        :param password:
        :return:
        """
        infile = PdfReader(file_name, strict=False)
        if infile.is_encrypted:
            password = password if password is not None else ""
            infile.decrypt(password)

        return infile

    @staticmethod
    def write_pdf_page(file_name: str, pages: Union[PageObject, List[PageObject]]):
        """
        write a PDF page

        :param file_name:
        :param pages:
        :return:
        """
        if not isinstance(pages, List):
            pages = [pages]
        outfile = PdfWriter()

        for page in pages:
            outfile.add_page(page)
        with open(file_name, "wb") as f:
            outfile.write(f)

    @staticmethod
    def get_pdf_total_page(file_name, password=None):
        infile = PdfUtils.read_pdf(file_name, password)

        total = len(infile.pages)
        return total

    @staticmethod
    def build_pdf_pages_list(file_name: str, password=None, pages="1") -> List:
        """
        Converts pages string to list of ints.

        :param file_name:
        :param password:
        :param pages:  Comma-separated page numbers.
                        Example: '1,3,4' or '1,4-end' or 'all'.
        :return:  List of int page numbers.
        """
        page_numbers = []

        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            total_page = PdfUtils.get_pdf_total_page(file_name=file_name, password=password)

            if pages == "all":
                page_numbers.append({"start": 1, "end": total_page})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = total_page
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})

        result_pages = []
        for p in page_numbers:
            result_pages.extend(range(p["start"], p["end"] + 1))
        sort_pages = sorted(set(result_pages))
        return sort_pages

    @staticmethod
    def segments_in_bbox(bbox, v_segments, h_segments):
        """Returns all line segments present inside a bounding box.

        Parameters
        ----------
        bbox : tuple
            Tuple (x1, y1, x2, y2) representing a bounding box where
            (x1, y1) -> lb and (x2, y2) -> rt in PDFMiner coordinate
            space.
        v_segments : list
            List of vertical line segments.
        h_segments : list
            List of vertical horizontal segments.

        Returns
        -------
        v_s : list
            List of vertical line segments that lie inside table.
        h_s : list
            List of horizontal line segments that lie inside table.

        """
        lb = (bbox[0], bbox[1])
        rt = (bbox[2], bbox[3])
        v_s = [
            v
            for v in v_segments
            if v[1] > lb[1] - 2 and v[3] < rt[1] + 2 and lb[0] - 2 <= v[0] <= rt[0] + 2
        ]
        h_s = [
            h
            for h in h_segments
            if h[0] > lb[0] - 2 and h[2] < rt[0] + 2 and lb[1] - 2 <= h[1] <= rt[1] + 2
        ]
        return v_s, h_s

    @staticmethod
    def segments_in_bbox_v2(bbox, v_segments: List[Line], h_segments: List[Line], diff=2):
        """Returns all line segments present inside a bounding box.

        Parameters
        ----------
        bbox : tuple
            Tuple (x1, y1, x2, y2) representing a bounding box where
            (x1, y1) -> lb and (x2, y2) -> rt in PDFMiner coordinate
            space.
        v_segments : list
            List of vertical line segments.
        h_segments : list
            List of vertical horizontal segments.

        Returns
        -------
        v_s : list
            List of vertical line segments that lie inside table.
        h_s : list
            List of horizontal line segments that lie inside table.

        """

        v_s = PdfUtils.line_in_box(bbox=bbox, segments=v_segments, diff=diff)
        h_s = PdfUtils.line_in_box(bbox=bbox, segments=h_segments, diff=diff)
        return v_s, h_s

    @staticmethod
    def line_in_box(bbox, segments: List[Line], diff=2):
        """
        line in bbox

        :param bbox:
        :param segments:
        :param diff:
        :return:
        """
        lb = Point(bbox[0], bbox[1])
        rt = Point(bbox[2], bbox[3])

        min_x = min(lb.x, rt.x)
        max_x = max(lb.x, rt.x)

        min_y = min(lb.y, rt.y)
        max_y = max(lb.y, rt.y)

        v_s_v2 = []
        for line in segments:
            if min_x - diff <= line.min_x <= line.max_x <= max_x + diff \
                    and min_y - diff <= line.min_y <= line.max_y <= max_y + diff:
                v_s_v2.append(line)

        return v_s_v2

    @staticmethod
    def text_in_bbox(bbox, text: List[LTTextLine], diff=2):
        """Returns all text objects present inside a bounding box.

        Parameters
        bbox : tuple
            Tuple (x1, y1, x2, y2) representing a bounding box where
            (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
            space.
        text : List of PDFMiner text objects.

        Returns
        t_bbox : list
            List of PDFMiner text objects that lie inside table, discarding the overlapping ones

            :param bbox: Tuple (x1, y1, x2, y2) representing a bounding box where
            (x1, y1) -> lb and (x2, y2) -> rt in the PDF coordinate
            space.
            :param text: List of PDFMiner text objects.
            :param diff:

        """
        lb = (bbox[0], bbox[1])
        rt = (bbox[2], bbox[3])
        t_bbox = []
        remain_t_bbox = []

        for index, text_box in enumerate(text):
            # y in box
            if lb[1] - diff <= (text_box.y0 + text_box.y1) / 2.0 <= rt[1] + diff:

                if lb[0] - diff <= (text_box.x0 + text_box.x1) / 2.0 <= rt[0] + diff:
                    t_bbox.append(text_box)
                elif lb[0] - diff <= text_box.x0:
                    # 超出边框的box 需要拆分
                    # 拆分text_box
                    new_text_cell = LTTextLineHorizontal(word_margin=text_box.word_margin)
                    other_text_cell = LTTextLineHorizontal(word_margin=text_box.word_margin)
                    for index_char, char_object in enumerate(text_box._objs):
                        if isinstance(char_object, LTChar):
                            if lb[0] - diff <= char_object.x0 < char_object.x1 <= rt[0] + diff:
                                new_text_cell.add(char_object)
                            else:
                                other_text_cell.add(char_object)
                        elif isinstance(char_object, LTAnno):
                            pass
                            # logger.info(f"不是字符： {index_char} - {char_object}")
                        else:
                            logger.info(f"不是字符： {index_char} - {char_object}")

                    results = [new_text_cell, other_text_cell]
                    src_text = text_box.get_text().replace("\n", "\\n")
                    logger.info(
                        f"不在表格范围之内：拆分text_box: [{len(text_box)} = {' + '.join([str(len(char_cell)) for char_cell in results])}] "
                        f"- 【{src_text}】 -> 【 {' + '.join([char_cell.get_text() for char_cell in results])} 】")

                    t_bbox.append(new_text_cell)
                    remain_t_bbox.append(other_text_cell)
                else:
                    remain_t_bbox.append(text_box)
            else:
                remain_t_bbox.append(text_box)

        # Avoid duplicate text by discarding overlapping boxes
        rest = {t for t in t_bbox}
        for ba in t_bbox:
            for bb in rest.copy():
                if ba == bb:
                    continue
                if MathUtils.bbox_intersect(ba, bb):
                    # if the intersection is larger than 80% of ba's size, we keep the longest
                    if (MathUtils.bbox_intersection_area(ba, bb) / MathUtils.bbox_area(ba)) > 0.8:
                        if MathUtils.bbox_longer(bb, ba):
                            rest.discard(ba)
        unique_boxes = list(rest)

        return unique_boxes, remain_t_bbox

    @staticmethod
    def merge_close_lines(ar, line_tol=2, last_merge_threold=-1):
        """Merges lines which are within a tolerance by calculating a
        moving mean, based on their x or y axis projections.

        Parameters
        ----------
        ar : list
        line_tol : int, optional (default: 2)

        Returns
        -------
        ret : list

        """
        ret = []
        for a in ar:
            if not ret:
                ret.append(a)
            else:
                temp = ret[-1]
                if np.isclose(temp, a, atol=line_tol):
                    temp = (temp + a) / 2.0
                    ret[-1] = temp
                else:
                    ret.append(a)

        # 判断每一行的宽度
        total = len(ret)
        if total > 2 and last_merge_threold > 0:
            avg_width = abs(max(ret) - min(ret)) / total

            col_pairs = [ret[i + 1] - ret[i] for i in range(0, total - 1)]

            first_width = abs(col_pairs[0])
            if first_width < last_merge_threold and first_width < avg_width * 0.2:
                logger.info(f"第一个宽度太小过滤掉：{first_width} - total:{total} - avg_width: {avg_width:.2f} - {ret}")
                ret = ret[1:]

            last_width = abs(col_pairs[-1])
            if last_width < last_merge_threold and last_width < avg_width * 0.2:
                logger.info(f"最后一个宽度太小过滤掉：{last_width} - total:{total} - avg_width: {avg_width:.2f} - {ret}")
                ret = ret[:-1]

        return ret

    @staticmethod
    def text_strip(text, strip=""):
        """Strips any characters in `strip` that are present in `text`.
        Parameters
        ----------
        text : str
            Text to process and strip.
        strip : str, optional (default: '')
            Characters that should be stripped from `text`.
        Returns
        -------
        stripped : str
        """
        if not strip:
            return text

        stripped = re.sub(
            fr"[{''.join(map(re.escape, strip))}]", "", text, flags=re.UNICODE
        )
        return stripped

    @staticmethod
    def flag_font_size(textline, direction, strip_text=""):
        """Flags super/subscripts in text by enclosing them with <s></s>.
        May give false positives.

        Parameters
        ----------
        textline : list
            List of PDFMiner LTChar objects.
        direction : string
            Direction of the PDFMiner LTTextLine object.
        strip_text : str, optional (default: '')
            Characters that should be stripped from a string before
            assigning it to a cell.

        Returns
        -------
        fstring : string

        """
        if direction == "horizontal":
            d = [
                (t.get_text(), np.round(t.height, decimals=6))
                for t in textline
                if not isinstance(t, LTAnno)
            ]
        elif direction == "vertical":
            d = [
                (t.get_text(), np.round(t.width, decimals=6))
                for t in textline
                if not isinstance(t, LTAnno)
            ]
        l = [np.round(size, decimals=6) for text, size in d]
        if len(set(l)) > 1:
            flist = []
            min_size = min(l)
            for key, chars in groupby(d, itemgetter(1)):
                if key == min_size:
                    fchars = [t[0] for t in chars]
                    if "".join(fchars).strip():
                        fchars.insert(0, "<s>")
                        fchars.append("</s>")
                        flist.append("".join(fchars))
                else:
                    fchars = [t[0] for t in chars]
                    if "".join(fchars).strip():
                        flist.append("".join(fchars))
            fstring = "".join(flist)
        else:
            fstring = "".join([t.get_text() for t in textline])
        return PdfUtils.text_strip(fstring, strip_text)

    @staticmethod
    def split_textline(table, textline, direction, flag_size=False, strip_text=""):
        """Splits PDFMiner LTTextLine into substrings if it spans across
        multiple rows/columns.

        Parameters
        ----------
        table : camelot.core.Table
        textline : object
            PDFMiner LTTextLine object.
        direction : string
            Direction of the PDFMiner LTTextLine object.
        flag_size : bool, optional (default: False)
            Whether or not to highlight a substring using <s></s>
            if its size is different from rest of the string. (Useful for
            super and subscripts.)
        strip_text : str, optional (default: '')
            Characters that should be stripped from a string before
            assigning it to a cell.

        Returns
        -------
        grouped_chars : list
            List of tuples of the form (idx, text) where idx is the index
            of row/column and text is the an lttextline substring.

        """
        idx = 0
        cut_text = []
        bbox = textline.bbox
        try:
            if direction == "horizontal" and not textline.is_empty():
                x_overlap = [
                    i
                    for i, x in enumerate(table.cols)
                    if x[0] <= bbox[2] and bbox[0] <= x[1]
                ]
                r_idx = [
                    j
                    for j, r in enumerate(table.rows)
                    if r[1] <= (bbox[1] + bbox[3]) / 2 <= r[0]
                ]
                r = r_idx[0]
                x_cuts = [
                    (c, table.cells[r][c].x2) for c in x_overlap if table.cells[r][c].right
                ]
                if not x_cuts:
                    x_cuts = [(x_overlap[0], table.cells[r][-1].x2)]
                for obj in textline._objs:
                    row = table.rows[r]
                    for cut in x_cuts:
                        if isinstance(obj, LTChar):
                            if (
                                    row[1] <= (obj.y0 + obj.y1) / 2 <= row[0]
                                    and (obj.x0 + obj.x1) / 2 <= cut[1]
                            ):
                                cut_text.append((r, cut[0], obj))
                                break
                            else:
                                # TODO: add test
                                if cut == x_cuts[-1]:
                                    cut_text.append((r, cut[0] + 1, obj))
                        elif isinstance(obj, LTAnno):
                            cut_text.append((r, cut[0], obj))
            elif direction == "vertical" and not textline.is_empty():
                y_overlap = [
                    j
                    for j, y in enumerate(table.rows)
                    if y[1] <= bbox[3] and bbox[1] <= y[0]
                ]
                c_idx = [
                    i
                    for i, c in enumerate(table.cols)
                    if c[0] <= (bbox[0] + bbox[2]) / 2 <= c[1]
                ]
                c = c_idx[0]
                y_cuts = [
                    (r, table.cells[r][c].y1) for r in y_overlap if table.cells[r][c].bottom
                ]
                if not y_cuts:
                    y_cuts = [(y_overlap[0], table.cells[-1][c].y1)]
                for obj in textline._objs:
                    col = table.cols[c]
                    for cut in y_cuts:
                        if isinstance(obj, LTChar):
                            if (
                                    col[0] <= (obj.x0 + obj.x1) / 2 <= col[1]
                                    and (obj.y0 + obj.y1) / 2 >= cut[1]
                            ):
                                cut_text.append((cut[0], c, obj))
                                break
                            else:
                                # TODO: add test
                                if cut == y_cuts[-1]:
                                    cut_text.append((cut[0] - 1, c, obj))
                        elif isinstance(obj, LTAnno):
                            cut_text.append((cut[0], c, obj))
        except IndexError:
            return [(-1, -1, textline.get_text())]
        grouped_chars = []
        for key, chars in groupby(cut_text, itemgetter(0, 1)):
            if flag_size:
                grouped_chars.append(
                    (
                        key[0],
                        key[1],
                        PdfUtils.flag_font_size(
                            [t[2] for t in chars], direction, strip_text=strip_text
                        ),
                    )
                )
            else:
                gchars = [t[2].get_text() for t in chars]
                grouped_chars.append(
                    (key[0], key[1], PdfUtils.text_strip("".join(gchars), strip_text))
                )
        return grouped_chars

    @staticmethod
    def get_table_index(
            table, t, direction, split_text=False, flag_size=False, strip_text=""
    ):
        """Gets indices of the table cell where given text object lies by
        comparing their y and x-coordinates.

        Parameters
        ----------
        table : camelot.core.Table
        t : object
            PDFMiner LTTextLine object.
        direction : string
            Direction of the PDFMiner LTTextLine object.
        split_text : bool, optional (default: False)
            Whether or not to split a text line if it spans across
            multiple cells.
        flag_size : bool, optional (default: False)
            Whether or not to highlight a substring using <s></s>
            if its size is different from rest of the string. (Useful for
            super and subscripts)
        strip_text : str, optional (default: '')
            Characters that should be stripped from a string before
            assigning it to a cell.

        Returns
        -------
        indices : list
            List of tuples of the form (r_idx, c_idx, text) where r_idx
            and c_idx are row and column indices.
        error : float
            Assignment error, percentage of text area that lies outside
            a cell.
            +-------+
            |       |
            |   [Text bounding box]
            |       |
            +-------+

        """
        r_idx, c_idx = [-1] * 2
        for r in range(len(table.rows)):
            if (t.y0 + t.y1) / 2.0 < table.rows[r][0] and (t.y0 + t.y1) / 2.0 > table.rows[
                r
            ][1]:
                lt_col_overlap = []
                for c in table.cols:
                    if c[0] <= t.x1 and c[1] >= t.x0:
                        left = t.x0 if c[0] <= t.x0 else c[0]
                        right = t.x1 if c[1] >= t.x1 else c[1]
                        lt_col_overlap.append(abs(left - right) / abs(c[0] - c[1]))
                    else:
                        lt_col_overlap.append(-1)
                if len(list(filter(lambda x: x != -1, lt_col_overlap))) == 0:
                    text = t.get_text().strip("\n")
                    text_range = (t.x0, t.x1)
                    col_range = (table.cols[0][0], table.cols[-1][1])
                    warnings.warn(
                        f"{text} {text_range} does not lie in column range {col_range}"
                    )
                r_idx = r
                c_idx = lt_col_overlap.index(max(lt_col_overlap))
                break

        # error calculation
        y0_offset, y1_offset, x0_offset, x1_offset = [0] * 4
        if t.y0 > table.rows[r_idx][0]:
            y0_offset = abs(t.y0 - table.rows[r_idx][0])
        if t.y1 < table.rows[r_idx][1]:
            y1_offset = abs(t.y1 - table.rows[r_idx][1])
        if t.x0 < table.cols[c_idx][0]:
            x0_offset = abs(t.x0 - table.cols[c_idx][0])
        if t.x1 > table.cols[c_idx][1]:
            x1_offset = abs(t.x1 - table.cols[c_idx][1])
        X = 1.0 if abs(t.x0 - t.x1) == 0.0 else abs(t.x0 - t.x1)
        Y = 1.0 if abs(t.y0 - t.y1) == 0.0 else abs(t.y0 - t.y1)
        charea = X * Y
        error = ((X * (y0_offset + y1_offset)) + (Y * (x0_offset + x1_offset))) / charea

        if split_text:
            return (
                PdfUtils.split_textline(
                    table, t, direction, flag_size=flag_size, strip_text=strip_text
                ),
                error,
            )
        else:
            if flag_size:
                return (
                    [
                        (
                            r_idx,
                            c_idx,
                            PdfUtils.flag_font_size(t._objs, direction, strip_text=strip_text),
                        )
                    ],
                    error,
                )
            else:
                return [(r_idx, c_idx, PdfUtils.text_strip(t.get_text(), strip_text))], error

    @staticmethod
    def compute_accuracy(error_weights):
        """Calculates a score based on weights assigned to various
        parameters and their error percentages.

        Parameters
        ----------
        error_weights : list
            Two-dimensional list of the form [[p1, e1], [p2, e2], ...]
            where pn is the weight assigned to list of errors en.
            Sum of pn should be equal to 100.

        Returns
        -------
        score : float

        """
        SCORE_VAL = 100
        try:
            score = 0
            if sum([ew[0] for ew in error_weights]) != SCORE_VAL:
                raise ValueError("Sum of weights should be equal to 100.")
            for ew in error_weights:
                weight = ew[0] / len(ew[1])
                for error_percentage in ew[1]:
                    score += weight * (1 - error_percentage)
        except ZeroDivisionError:
            score = 0
        return score

    @staticmethod
    def compute_whitespace(d):
        """Calculates the percentage of empty strings in a
        two-dimensional list.

        Parameters
        ----------
        d : list

        Returns
        -------
        whitespace : float
            Percentage of empty cells.

        """
        whitespace = 0
        r_nempty_cells, c_nempty_cells = [], []
        for i in d:
            for j in i:
                if j.strip() == "":
                    whitespace += 1
        whitespace = 100 * (whitespace / float(len(d) * len(d[0])))
        return whitespace

    @staticmethod
    def recoverpix(doc, item):
        xref = item[0]  # xref of PDF image
        smask = item[1]  # xref of its /SMask

        # special case: /SMask or /Mask exists
        if smask > 0:
            pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
            if pix0.alpha:  # catch irregular situation
                pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
            mask = fitz.Pixmap(doc.extract_image(smask)["image"])

            try:
                pix = fitz.Pixmap(pix0, mask)
            except:  # fallback to original base image in case of problems
                pix = fitz.Pixmap(doc.extract_image(xref)["image"])

            if pix0.n > 3:
                ext = "pam"
            else:
                ext = "png"

            return {  # create dictionary expected by caller
                "ext": ext,
                "colorspace": pix.colorspace.n,
                "image": pix.tobytes(ext),
            }

        # special case: /ColorSpace definition exists
        # to be sure, we convert these cases to RGB PNG images
        if "/ColorSpace" in doc.xref_object(xref, compressed=True):
            pix = fitz.Pixmap(doc, xref)
            pix = fitz.Pixmap(fitz.csRGB, pix)
            return {  # create dictionary expected by caller
                "ext": "png",
                "colorspace": 3,
                "image": pix.tobytes("png"),
            }
        return doc.extract_image(xref)

    @staticmethod
    def extract_pdf_image(file_name, output_dir,
                          dim_limit=100, rel_size=0, abs_size=1024,
                          page_width_limit=500,
                          add_xref=False, do_rotate=False,
                          page_file_name=None, do_convert_jpg=True, ):
        """
        提取 imaged pdf file

        :param file_name:
        :param output_dir:
        :param dim_limit: each image side must be greater than this
        :param rel_size:  image size ratio must be larger than this (5%)
        :param abs_size:  absolute image size limit 2 KB: ignore if smaller
        :param page_width_limit:  页面大小限制
        :param add_xref:
        :param do_rotate: 旋转图片
        :param page_file_name:
        :param do_convert_jpg:
        :return:
        """
        t0 = time.time()

        try:
            doc = fitz.open(file_name)
            page_count = doc.page_count
            logger.info(f"开始提取图片型PDF中的图片: total page - {page_count}")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"提取图片型PDF中的图片异常：{file_name} - {e}")
            return 0

        FileUtils.check_file_exists(f"{output_dir}/demo.txt")
        xreflist = []
        imglist = []
        save_page_image_file = []
        pdf_page = []
        other_page = []
        for pno in range(page_count):
            il = doc.get_page_images(pno)
            imglist.extend([x[0] for x in il])
            for img in il:
                xref = img[0]
                if xref in xreflist:
                    continue
                width = img[2]
                height = img[3]
                if min(width, height) <= dim_limit:
                    logger.info(f"当前图片被过滤，宽度高度小于：{dim_limit}")
                    continue
                image = PdfUtils.recoverpix(doc, img)
                n = image["colorspace"]
                imgdata = image["image"]

                if len(imgdata) <= abs_size:
                    logger.info(f"当前图片被过滤，图片占用大小：{abs_size}")
                    continue
                current_rel_size = len(imgdata) / (width * height * n)
                if current_rel_size <= rel_size:
                    logger.info(f"当前图片被过滤，image size ratio：{current_rel_size} 大小：{rel_size}")
                    continue

                xref_str = f"_{xref}" if add_xref else ""

                if min(width, height) < page_width_limit:
                    image_type = "other"
                    page_number = len(other_page) + 1
                    page_number_str = f"{page_number:03}"
                    new_name = f"{FileUtils.get_file_name(file_name)}_{image_type}_{page_number_str}{xref_str}.{image['ext']}"
                else:
                    page_number = len(pdf_page) + 1
                    page_number_str = f"{page_number:03}"

                    if page_file_name is None:
                        image_type = "page"
                        new_name = f"{FileUtils.get_file_name(file_name)}_{image_type}_{page_number_str}{xref_str}.{image['ext']}"
                        if do_rotate:
                            new_name = new_name.replace(f"_page_{page_number_str}", f"_page_{page_number_str}_rotate")
                    else:
                        page_number_str = f"_{page_number:03}" if page_number > 1 else ""
                        new_name = f"{FileUtils.get_file_name(file_name)}{page_number_str}.{image['ext']}"

                image_file = os.path.join(output_dir, new_name)

                with open(image_file, "wb") as fout:
                    fout.write(imgdata)

                xreflist.append(xref)

                if do_convert_jpg:
                    image_file = PdfImageProcessor.convert_png_to_jpg(image_file, remove_src=False)

                if min(width, height) >= page_width_limit:
                    pdf_page.append(image_file)
                else:
                    other_page.append(image_file)
                save_page_image_file.append(image_file)

                if do_rotate:
                    rotated_image, angle = PdfImageProcessor.rotate_image(image_file, save_image_file=image_file)

        use_time = time.time() - t0
        imglist = list(set(imglist))

        total_image = len(xreflist)
        logger.info(f"提取图片型PDF中的图片，耗时：{use_time:.3f} s, "
                    f"总共{len(imglist)}张图片，提取：{total_image}张图片, 页面：{len(pdf_page)} 其他：{len(other_page)}。"
                    f"{file_name} - {output_dir}")

        return pdf_page, other_page

    @staticmethod
    def extract_pdf_image_page(file_name, output_dir,
                               dim_limit=100,
                               page_width_limit=500,
                               page_file_name=None):
        """
        提取 imaged pdf file

        :param file_name:
        :param output_dir:
        :param dim_limit:
        :param page_width_limit:
        :param page_file_name:
        :return:
        """
        begin_time = time.time()

        layout, dimensions, horizontal_text, vertical_text, images, filtered_images = PdfUtils.get_pdf_object(file_name)

        image_writer = ImageWriter(output_dir)

        image_infos = []
        save_page_image_file = []
        pdf_page = []
        other_page = []
        for index, image in enumerate(images):
            width, height = image.srcsize
            if PdfUtils.filter_pdf_image_page(image=image, dim_limit=dim_limit):
                continue

            save_name = image_writer.export_image(image)

            image_ext = os.path.splitext(save_name)[-1]
            raw_image_name = os.path.join(output_dir, save_name)

            if min(width, height) < page_width_limit:
                image_type = "other"
                page_number = len(other_page) + 1
                page_number_str = f"{page_number:03}"
                new_name = f"{FileUtils.get_file_name(file_name)}_{image_type}_{page_number_str}{image_ext}"
            else:
                page_number = len(pdf_page) + 1
                page_number_str = f"{page_number:03}"

                if page_file_name is None:
                    image_type = "page"
                    new_name = f"{FileUtils.get_file_name(file_name)}_{image_type}_{page_number_str}{image_ext}"
                else:
                    page_number_str = f"_{page_number:03}" if page_number > 1 else ""
                    new_name = f"{FileUtils.get_file_name(file_name)}{page_number_str}{image_ext}"

            image_file = os.path.join(output_dir, new_name)
            shutil.move(raw_image_name, image_file)
            logger.info(f"图片重命名：{raw_image_name} -> {image_file}")

            if min(width, height) >= page_width_limit:
                pdf_page.append(image_file)
            else:
                other_page.append(image_file)
            save_page_image_file.append(image_file)

            if width > height:
                src_image = cv2.imread(image_file)
                PdfImageProcessor.rotate_image_angle_v2(image=src_image,
                                                        angle=270,
                                                        save_image_file=image_file)
                logger.info(f"pdf图片宽度大于高度，逆时针旋转90度：{image_file}")

            image_info = {
                "key": PdfUtils.get_pdf_image_key(image),
                "name": image.name,
                "raw_name": save_name,
                "save_name": image_file,
                "src_save_name": raw_image_name,
                "relative_dir": f"./{new_name}",
                "bbox": image.bbox,
                "height": image.height,
                "width": image.width,
                "image_size": image.srcsize,
            }
            image_infos.append(image_info)
            logger.info(f"保存PDF图片: {index} - {image_info}")

        use_time = time.time() - begin_time

        total_image = len(images)
        logger.info(f"提取图片型PDF中的图片，耗时：{use_time:.3f} s, "
                    f"总共{total_image}张图片，提取：{total_image}张图片, 页面：{len(pdf_page)} 其他：{len(other_page)}。"
                    f"{file_name} - {output_dir}")

        return save_page_image_file

    @staticmethod
    def modify_ocr_block_line_type(ocr_cell_content: List[OcrCell]):
        """
        段落开始和结束修正

        :param ocr_cell_content:
        :return:
        """

        start_x_dict = defaultdict(list)
        end_x_dict = defaultdict(list)
        font_size_dict = defaultdict(list)

        for index, cell in enumerate(ocr_cell_content):
            if not cell.is_text():
                continue

            start_x_dict[round(cell.x1)].append(index)
            end_x_dict[round(cell.x2)].append(index)

            text_width = round(cell.text_width)
            if text_width > 0:
                font_size_dict[text_width].append(index)

        start_x_dict_sorted = CommonUtils.sorted_dict(start_x_dict, key=lambda x: len(x[1]), reverse=True)
        end_x_dict_sorted = CommonUtils.sorted_dict(end_x_dict, key=lambda x: len(x[1]), reverse=True)
        font_size_dict_sorted = CommonUtils.sorted_dict(font_size_dict, key=lambda x: len(x[1]), reverse=True)

        most_start_x = 0
        most_end_x = 0
        most_font_size = 10
        try:
            most_font_size = deepcopy(font_size_dict_sorted).popitem(last=False)[0]
            most_start_x = PdfUtils.get_pdf_line_begin_x(start_x_dict, font_size=most_font_size)
            most_end_x = deepcopy(end_x_dict_sorted).popitem(last=False)[0]
        except Exception as e:
            logger.warning(f"段落开始和结束修正,提取异常：{e}")
            pass

        for index, cell in enumerate(ocr_cell_content):
            if not cell.is_text():
                continue

            cell.parse_line_type(min_x=most_start_x, max_x=most_end_x, normal_font_size=most_font_size)

        return ocr_cell_content

    @staticmethod
    def merge_ocr_text_paragraph(ocr_cell_content: List[OcrCell]):
        """
        OCR 文本段落合并
            - txt -> 段落

        :param ocr_cell_content:
        :return:
        """
        result = []
        if ocr_cell_content is None:
            return result

        text_cells = []
        other_cells = []
        for cell in ocr_cell_content:
            if cell.is_text():
                text_cells.append(cell)
            else:
                other_cells.append(cell)

        # 开始的段落
        start_block_index = []
        for index, one_block in enumerate(text_cells):
            # 第一个默认是开始段落
            if index > 0 and one_block.line_type == PdfLineType.PARAGRAPH_START:
                start_block_index.append(index)

        merge_cells = []
        # 第一个默认是开始段落
        start_index = 0
        for index in range(0, len(start_block_index)):
            end_index = start_block_index[index]
            one_paras = text_cells[start_index:end_index]
            start_index = end_index

            new_cell = PdfUtils.merge_text_cell(cell_list=one_paras)
            if new_cell is not None:
                merge_cells.append(new_cell)

        # 添加末尾数据
        end_cells = text_cells[start_index:]
        new_cell = PdfUtils.merge_text_cell(cell_list=end_cells)
        if new_cell is not None:
            merge_cells.append(new_cell)

        merge_cells.extend(other_cells)

        merge_cells.sort(key=lambda cell: (cell.y1, cell.x1))

        return merge_cells

    @staticmethod
    def merge_text_cell(cell_list: List[OcrCell]):
        """
        合并OCR cell list

        :param cell_list:
        :return:
        """
        if len(cell_list) == 0:
            return None
        elif len(cell_list) == 1:
            return cell_list[0]

        content = []
        for block in cell_list:
            content.append(block.text)

        one_para_text = "".join(content)
        one_para_text = one_para_text.replace("\n", "")
        new_cell = OcrCell(left_top=cell_list[0].left_top,
                           right_bottom=cell_list[-1].right_bottom,
                           text=one_para_text,
                           cell_type=HtmlContentType.TXT,
                           inner_cells=cell_list)
        return new_cell

    @staticmethod
    def get_pdf_text_in_bbox(bbox: List, horizontal_text: List[LTTextLine], vertical_text: List[LTTextLine]):
        """
        筛选 在表格区域中的文本

        :param bbox:  Tuple (x1, y1, x2, y2) representing a bounding box where
            (x1, y1) -> lb and (x2, y2) -> rt in PDFMiner coordinate
            space.
        :param horizontal_text:
        :param vertical_text:
        :return:
        """

        new_horizontal_text, remain_horizontal_text = PdfUtils.text_in_bbox(bbox, horizontal_text)
        new_vertical_text, remain_vertical_text = PdfUtils.text_in_bbox(bbox, vertical_text)
        new_horizontal_text.sort(key=lambda x: (-x.y0, x.x0))
        new_vertical_text.sort(key=lambda x: (x.x0, -x.y0))

        text_bbox = {
            "horizontal": new_horizontal_text,
            "vertical": new_vertical_text,
        }

        remain_text_bbox = {
            "horizontal": remain_horizontal_text,
            "vertical": remain_vertical_text,
        }
        return text_bbox, remain_text_bbox

    @staticmethod
    def get_pdf_object(file_name, layout_kwargs=None):
        """
        获取pdf 中的文字，图片

        :param file_name:
        :param layout_kwargs:
        :return:
        """
        layout_kwargs = layout_kwargs if layout_kwargs is not None else {}
        layout, dimensions = PdfUtils.get_page_layout(file_name, **layout_kwargs)

        images = PdfUtils.get_text_objects(layout, ltype="image")
        horizontal_text = PdfUtils.get_text_objects(layout, ltype="horizontal_text")
        vertical_text = PdfUtils.get_text_objects(layout, ltype="vertical_text")

        result_images, filtered_images = PdfUtils.filter_pdf_image(images)
        return layout, dimensions, horizontal_text, vertical_text, result_images, filtered_images

    @staticmethod
    def save_pdf_image(images: List[LTImage], output_dir, image_dir, read_cache=True):
        """
        保存PDF image

        :param images:
        :param output_dir:
        :param image_dir:
        :param read_cache:
        :return:
        """
        image_infos = []
        if len(images) == 0:
            return image_infos, {}

        os.makedirs(image_dir, exist_ok=True)

        image_output_dir = os.path.join(output_dir, image_dir)
        image_info_file_name = os.path.join(image_output_dir, "image.json")
        if read_cache and FileUtils.check_file_exists(image_info_file_name):
            image_infos = FileUtils.load_json(image_info_file_name)
            logger.info(f"总共提取PDF图片【从image.json中读取】: {len(image_infos)} - {image_info_file_name}")
            pdf_image_mapping = {item["key"]: item for item in image_infos}
            return image_infos, pdf_image_mapping

        image_writer = ImageWriter(image_output_dir)
        for index, image in enumerate(images):
            w, h = image.srcsize
            area = w * h
            if area < 10:
                continue

            save_name = image_writer.export_image(image)
            relative_dir = f"{image_dir}/{save_name}"

            image_info = {
                "key": PdfUtils.get_pdf_image_key(image),
                "name": image.name,
                "raw_name": save_name,
                "save_name": os.path.join(output_dir, relative_dir),
                "relative_dir": f"./{relative_dir}",
                "bbox": image.bbox,
                "height": image.height,
                "width": image.width,
                "image_size": image.srcsize,
            }
            image_infos.append(image_info)
            logger.info(f"保存PDF图片: {index} - {image_info}")

        FileUtils.dump_json(image_info_file_name, image_infos)
        logger.info(f"总共提取PDF图片: {len(image_infos)} - {image_info_file_name}")

        pdf_image_mapping = {item["key"]: item for item in image_infos}
        return image_infos, pdf_image_mapping

    @staticmethod
    def check_is_imaged_pdf(file_name, ):
        """
        检测一个pdf 是否是图片型pdf

        :param file_name:
        :return:
        """
        begin_time = time.time()
        layout, dimensions, horizontal_text, vertical_text, images, filtered_images = PdfUtils.get_pdf_object(file_name)

        is_imaged_pdf = False
        if not horizontal_text:
            if images:
                is_imaged_pdf = True
        use_time = time.time() - begin_time
        logger.info(f"检测pdf类型，耗时：{use_time:.3f} s. "
                    f"当前pdf是 {'图片' if is_imaged_pdf else '数字'}型PDF,: {file_name} ")
        return is_imaged_pdf

    @staticmethod
    def check_is_imaged_pdf_v2(file_name, pages: List = None, new_file=None, pdf_dir=None):
        """
        检测一个pdf 是否是图片型pdf v2
                - 拆分成单页检测

        :param file_name:
        :param pages:
        :param new_file:
        :param pdf_dir:
        :return:
        """
        if pages is None:
            pages = [0, 1]

        if new_file is None:
            new_file = f"{Constants.HTML_BASE_DIR}/check_imaged_pdf/{TimeUtils.get_time()}/{FileUtils.get_raw_file_name(file_name)}"

        PdfUtils.split_pdf_by_page(pdf_file=file_name, pages=pages, new_file=new_file, pdf_dir=pdf_dir)
        result = PdfUtils.check_is_imaged_pdf(file_name=new_file)
        return result

    @staticmethod
    def get_pdf_line_begin_x(start_x_dict: Dict, font_size=10):
        """
        获取一行文本的开头

        :param start_x_dict:
        :param font_size:
        :return:
        """
        start_x_dict_sorted = CommonUtils.sorted_dict(start_x_dict, key=lambda x: len(x[1]), reverse=True)

        dict_sorted = deepcopy(start_x_dict_sorted)
        first_start_x, first_line_list = dict_sorted.popitem(last=False)

        most_start_x = first_start_x
        if len(start_x_dict_sorted) > 1:
            second_start_x, second_line_list = dict_sorted.popitem(last=False)

            # 71,92
            if second_start_x < first_start_x \
                    and abs(first_start_x - second_start_x - 2 * font_size) < font_size * 0.3:
                # and abs(len(first_line_list) - len(second_line_list)) < 4:
                most_start_x = second_start_x

        logger.info(f"most_start_x: {most_start_x} - start_x_dict_sorted: {start_x_dict_sorted}")
        return most_start_x

    @staticmethod
    def get_pdf_image_key(image: LTImage):
        """
        获取 pdf image key

        :param image:
        :return:
        """
        bbox_str = ','.join([str(int(x)) for x in image.bbox])
        key = f"{image.name}_{bbox_str}"
        return key

    @staticmethod
    def filter_pdf_image(images: List[LTImage], area_threshold=20):
        """
        过滤 PDF image

        :param images:
        :param area_threshold:
        :return:
        """
        result_images = []
        filtered_images = []
        for index, image in enumerate(images):
            w, h = image.srcsize
            area = w * h
            if area < area_threshold:
                filtered_images.append(image)
                continue

            result_images.append(image)

        return result_images, filtered_images

    @staticmethod
    def get_pdf_pages_list(file_name: str = None, password=None, pages="1", total_page=None) -> List:
        """
        Converts pages string to list of ints.

        :param file_name:
        :param password:
        :param pages:  Comma-separated page numbers.
                        Example: '1,3,4' or '1,4-end' or 'all'.
        :param total_page:
        :return:  List of int page numbers.
        """
        page_numbers = []

        if isinstance(pages, int):
            pages = str(pages)
        elif isinstance(pages, list):
            pages = ",".join([str(page) for page in pages])

        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            if total_page is None:
                total_page = PdfUtils.get_pdf_total_page(file_name=file_name, password=password)

            if pages == "all":
                page_numbers.append({"start": 1, "end": total_page})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = total_page
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})

        result_pages = []
        for p in page_numbers:
            result_pages.extend(range(p["start"], p["end"] + 1))
        sort_pages = sorted(set(result_pages))
        return sort_pages

    @staticmethod
    def filter_pdf_image_page(image: LTImage, area_threshold=10,
                              dim_limit=100, ):
        """
        过滤PDF IMAGE 页面

        :param image:
        :param area_threshold:
        :param dim_limit:
        :return:
        """
        flag = False
        width, height = image.srcsize
        area = width * height
        if area < area_threshold:
            return True
        if min(width, height) <= dim_limit:
            logger.info(f"当前图片被过滤，宽度高度小于：{dim_limit}")
            return True

        return False

    @staticmethod
    def split_pdf_by_page(pdf_file, pages: List, new_file, pdf_dir=None):
        """
        拆分PDF

        :param pdf_file:
        :param pages:
        :param new_file:
        :param pdf_dir:
        :return:
        """
        try:
            file_name = PdfUtils.download_pdf_before_check(pdf_file, pdf_dir=pdf_dir, )
            pdf_reader = PdfReader(file_name, strict=False)

            merger = PdfWriter()
            total = len(pdf_reader.pages)
            new_pages = [page for page in pages if 0 <= page < total]
            merger.append(pdf_reader, "page begin and end", pages=new_pages)

            FileUtils.check_file_exists(new_file)
            with open(new_file, "wb") as f:
                merger.write(f)

            merger.close()
            logger.info(f"保存拆分的新PDF: 页数：{len(new_pages)} 原始PDF: {total} 页,文件路径：{new_file}")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"拆分PDF出现异常：{pdf_file}")

    @staticmethod
    def extract_pdf_rect(file_name, line_max=2):
        """
        获取PDF中的矩形框

        :param file_name:
        :param line_max:
        :return:
        """
        layout, dimensions = PdfUtils.get_page_layout(file_name, )
        line_list, other_list, skip_list = PdfUtils.get_pdf_rect_from_layout(layout=layout, line_max=line_max)
        return dimensions, line_list, other_list, skip_list

    @staticmethod
    def get_pdf_rect_from_layout(layout, line_max=2):
        """
        获取PDF中的矩形框

        :param layout:
        :param line_max:
        :return:
        """
        rects: List[LTRect] = PdfUtils.get_text_objects(layout, ltype="rect")

        line_list = []
        other_list = []
        skip_list = []
        for index, rect in enumerate(rects):
            rect_h = rect.height
            rect_w = rect.width

            if rect_h < line_max and rect_w < line_max:
                skip_list.append(rect)
            elif rect_h < line_max or rect_w < line_max:
                line_list.append(rect)
            else:
                other_list.append(rect)

        logger.info(f"get pdf rect - line_list: {len(line_list)} "
                    f"-other_list: {len(other_list)} "
                    f"-skip_list: {len(skip_list)}")

        return line_list, other_list, skip_list
