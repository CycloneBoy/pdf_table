#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable
# @File    ：pdf_handlers
# @Author  ：cycloneboy
# @Date    ：20xx/5/29 14:24

import shutil
from tempfile import TemporaryDirectory
from typing import List

from . import TableList
from .table_extractor_lattice import TableExtractorLattice
from .table_extractor_pdf import TableExtractorPdf
from .table_extractor_stream import TableExtractorStream
from ...utils import (
    logger,
    TimeUtils,
    FileUtils,
    RequestUtils,
    PdfUtils,
    PdfTableExtractUtils
)

__all__ = [
    "PDFHandler",
]

"""
PDF 解析表格

"""


class PDFHandler(object):
    """Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.

    Parameters
    ----------
    filepath : str
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.

    """

    def __init__(self, filepath, pages="1", password=None, output_dir=None,
                 src_id=None, delete_check_success=False):
        if FileUtils.is_url(filepath):
            filepath = RequestUtils.download_pdf_from_url(filepath)
        self.filepath = filepath
        if not filepath.lower().endswith(".pdf"):
            raise NotImplementedError(f"File format not supported: {filepath}")

        self.password = password
        self.pages = self._get_pages(pages)
        self.output_dir = output_dir if output_dir is not None else \
            f"{FileUtils.get_output_dir_with_time(add_now_end=False)}/{src_id}/{TimeUtils.now_str_short()}"
        self.src_id = src_id
        self.delete_check_success = delete_check_success

    def _get_pages(self, pages) -> List:
        page_list = PdfUtils.build_pdf_pages_list(file_name=self.filepath, password=self.password, pages=pages)
        return page_list

    def _save_page(self, filepath, pages, temp):
        """Saves specified page from PDF into a temporary directory.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        pages : int
            Page number.
        temp : str
            Tmp directory.

        """
        PdfTableExtractUtils.save_pdf_page(filepath=filepath, pages=pages, temp_dir=temp, password=self.password)

    def parse(self, flavor="lattice",
              suppress_stdout=False,
              layout_kwargs={},
              temp_dir=None,
              delete_temp_dir=True,
              **kwargs):
        """Extracts tables by calling parser.get_tables on all single
        page PDFs.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use ('lattice' or 'stream').
            Lattice is used by default.
        suppress_stdout : str (default: False)
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs.
        kwargs : dict
            See camelot.read_pdf kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        tempdir = temp_dir if temp_dir is not None else TemporaryDirectory().name
        FileUtils.check_file_exists(f"{tempdir}/temp.txt")
        logger.info(f"parse pdf tempdir: {tempdir}")

        save_file_name = PdfTableExtractUtils.save_pdf_page(filepath=self.filepath, pages=self.pages,
                                                            temp_dir=tempdir, password=self.password)

        parser_class = {
            "lattice": TableExtractorLattice,
            "stream": TableExtractorStream,
            "pdf": TableExtractorPdf,
        }
        if flavor == "pdf":
            kwargs["output_dir"] = self.output_dir
            kwargs["filepath"] = self.filepath
            kwargs["src_id"] = self.src_id
            kwargs["delete_check_success"] = self.delete_check_success

        parser = parser_class[flavor](**kwargs)

        tables = []
        for p in save_file_name:
            t = parser.extract_tables(
                p, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
            )
            tables.extend(t)
        table_list = TableList(sorted(tables))

        if delete_temp_dir:
            logger.info(f"delete tempdir: {tempdir}")
            shutil.rmtree(temp_dir)

        FileUtils.delete_file_list(save_file_name)

        return table_list

    def check_imaged_pdf(self, flavor="pdf",
                         suppress_stdout=False,
                         layout_kwargs={},
                         temp_dir=None,
                         delete_temp_dir=True,
                         **kwargs):
        """
        检测是否是图片型PDF

        :param flavor:
        :param suppress_stdout:
        :param layout_kwargs:
        :param temp_dir:
        :param delete_temp_dir:
        :param kwargs:
        :return:
        """
        tempdir = temp_dir if temp_dir is not None else TemporaryDirectory().name
        FileUtils.check_file_exists(f"{tempdir}/temp.txt")
        logger.info(f"parse pdf tempdir: {tempdir}")

        save_file_name = PdfTableExtractUtils.save_pdf_page(filepath=self.filepath, pages=self.pages,
                                                            temp_dir=tempdir, password=self.password)

        if flavor == "pdf":
            kwargs["output_dir"] = self.output_dir
            kwargs["filepath"] = self.filepath
            kwargs["src_id"] = self.src_id
            kwargs["delete_check_success"] = self.delete_check_success

        parser = TableExtractorPdf(**kwargs)

        imaged_page = []
        is_imaged_pdf = True
        for page_file_name in save_file_name:
            flag = parser.check_is_imaged_pdf(
                page_file_name, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
            )
            imaged_page.append(flag)
            if not flag:
                is_imaged_pdf = False

        if delete_temp_dir:
            logger.info(f"delete tempdir: {tempdir}")
            shutil.rmtree(temp_dir)

        return is_imaged_pdf
