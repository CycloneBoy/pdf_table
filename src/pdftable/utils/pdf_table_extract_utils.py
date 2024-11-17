#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable
# @File    ：pdf_table_extract_utils
# @Author  ：cycloneboy
# @Date    ：20xx/5/29 14:32
import os
import traceback
from typing import List, Union

from . import BaseUtil, logger, PdfUtils, FileUtils

"""
PDF表格提取工具类
"""


class PdfTableExtractUtils(BaseUtil):
    """
    PDF表格提取工具类
    """

    def init(self):
        pass

    @staticmethod
    def save_pdf_page(filepath: str, pages: Union[List, int], temp_dir: str, password=None) -> List:
        """
        Saves specified page from PDF into a temporary directory.

            :param filepath:  Filepath or URL of the PDF file.
            :param pages:  Page number.
            :param temp_dir: Tmp directory.
            :param password:

        """
        FileUtils.check_file_exists(f"{temp_dir}/temp.txt")
        total_page = 0
        # 从缓存读取
        save_file_name = []
        exist_page_list = []
        for page in pages:
            fpath = os.path.join(temp_dir, f"page-{page}.pdf")
            if FileUtils.check_file_exists(fpath):
                exist_page_list.append(page)
                save_file_name.append(fpath)

        if len(exist_page_list) == len(pages):
            return save_file_name
        else:
            save_file_name = []

        try:
            infile = PdfUtils.read_pdf(file_name=filepath, password=password)
            total_page = len(infile.pages)
            if not isinstance(pages, list):
                pages = [pages]
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"读取pdf异常：{filepath} - {e}")
            pages = []

        for page in pages:
            fpath = os.path.join(temp_dir, f"page-{page}.pdf")
            if page in exist_page_list:
                save_file_name.append(fpath)
                continue

            froot, fext = os.path.splitext(fpath)
            if page > total_page:
                msg = f"current page: {page} out off scope total page: {total_page} "
                logger.warning(msg)
                raise RuntimeError(msg)
            p = infile.pages[page - 1]
            PdfUtils.write_pdf_page(file_name=fpath, pages=p)

            layout, dim = PdfUtils.get_page_layout(fpath)
            # fix rotated PDF
            chars = PdfUtils.get_text_objects(layout, ltype="char")
            horizontal_text = PdfUtils.get_text_objects(layout, ltype="horizontal_text")
            vertical_text = PdfUtils.get_text_objects(layout, ltype="vertical_text")
            rotation = PdfUtils.get_rotation(chars, horizontal_text, vertical_text)
            if rotation != "":
                fpath_new = "".join([froot.replace("page", "p"), "_rotated", fext])
                os.rename(fpath, fpath_new)
                infile2 = PdfUtils.read_pdf(file_name=fpath_new, password=password)
                p = infile2.pages[0]
                if rotation == "anticlockwise":
                    p.rotate(90)
                elif rotation == "clockwise":
                    p.rotate(-90)
                PdfUtils.write_pdf_page(file_name=fpath, pages=p)

            save_file_name.append(fpath)

        return save_file_name
