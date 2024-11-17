#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_extractor_base
# @Author  ：cycloneboy
# @Date    ：20xx/5/29 16:06
import os
import warnings
from abc import abstractmethod

from .pdf_image_processor import BACKENDS
from ...utils import (
    FileUtils,
    PdfUtils, CommonUtils
)

__all__ = [
    'TableExtractorBase'
]

"""
PDF 表格解析基础类
"""


class TableExtractorBase(object):

    def __init__(self, debug=True):
        self.os = None
        self.debug = debug

        self.filename = None
        self.layout = None
        self.dimensions = None
        self.images = None
        self.horizontal_text = None
        self.vertical_text = None
        self.pdf_width = None
        self.pdf_height = None
        self.rootname = None
        self.imagename = None
        self.basename = None
        self.raw_filename = None
        self.pdf_scalers = None
        self.image_scalers = None
        self.color_list = CommonUtils.get_color_list()

    def _generate_layout(self, filename, layout_kwargs):
        self.filename = filename
        self.layout_kwargs = layout_kwargs if layout_kwargs is not None else {}
        self.layout, self.dimensions = PdfUtils.get_page_layout(filename, **self.layout_kwargs)

        self.images = PdfUtils.get_text_objects(self.layout, ltype="image")
        self.horizontal_text = PdfUtils.get_text_objects(self.layout, ltype="horizontal_text")
        self.vertical_text = PdfUtils.get_text_objects(self.layout, ltype="vertical_text")
        self.pdf_width, self.pdf_height = self.dimensions
        self.rootname, __ = os.path.splitext(self.filename)
        self.imagename = "".join([self.rootname, ".png"])

        self.basename = os.path.basename(self.rootname)
        self.raw_filename = FileUtils.get_file_name(filename)

    @staticmethod
    def _get_pdf_to_image_backend(backend):
        def implements_convert():
            methods = [
                method for method in dir(backend) if method.startswith("__") is False
            ]
            return "convert" in methods

        if isinstance(backend, str):
            if backend not in BACKENDS.keys():
                raise NotImplementedError(
                    f"Unknown backend '{backend}' specified. Please use either 'poppler' or 'ghostscript'."
                )

            # if backend == "ghostscript":
            #     warnings.warn(
            #         "'ghostscript' will be replaced by 'poppler' as the default image conversion"
            #         " backend in v0.12.0. You can try out 'poppler' with backend='poppler'.",
            #         DeprecationWarning,
            #     )

            return BACKENDS[backend]()
        else:
            if not implements_convert():
                raise NotImplementedError(
                    f"'{backend}' must implement a 'convert' method"
                )

            return backend

    @abstractmethod
    def extract_tables(self, filename, **kwargs):
        pass

    @staticmethod
    def _reduce_index(t, idx, shift_text):
        """Reduces index of a text object if it lies within a spanning
        cell.

        Parameters
        ----------
        table : camelot.core.Table
        idx : list
            List of tuples of the form (r_idx, c_idx, text).
        shift_text : list
            {'l', 'r', 't', 'b'}
            Select one or more strings from above and pass them as a
            list to specify where the text in a spanning cell should
            flow.

        Returns
        -------
        indices : list
            List of tuples of the form (r_idx, c_idx, text) where
            r_idx and c_idx are new row and column indices for text.

        """
        indices = []
        for r_idx, c_idx, text in idx:
            for d in shift_text:
                if d == "l":
                    if t.cells[r_idx][c_idx].hspan:
                        while not t.cells[r_idx][c_idx].left:
                            c_idx -= 1
                if d == "r":
                    if t.cells[r_idx][c_idx].hspan:
                        while not t.cells[r_idx][c_idx].right:
                            c_idx += 1
                if d == "t":
                    if t.cells[r_idx][c_idx].vspan:
                        while not t.cells[r_idx][c_idx].top:
                            r_idx -= 1
                if d == "b":
                    if t.cells[r_idx][c_idx].vspan:
                        while not t.cells[r_idx][c_idx].bottom:
                            r_idx += 1
            indices.append((r_idx, c_idx, text))
        return indices

    @staticmethod
    def _copy_spanning_text(t, copy_text=None):
        """Copies over text in empty spanning cells.

        Parameters
        ----------
        t : camelot.core.Table
        copy_text : list, optional (default: None)
            {'h', 'v'}
            Select one or more strings from above and pass them as a list
            to specify the direction in which text should be copied over
            when a cell spans multiple rows or columns.

        Returns
        -------
        t : camelot.core.Table

        """
        for f in copy_text:
            if f == "h":
                for i in range(len(t.cells)):
                    for j in range(len(t.cells[i])):
                        if t.cells[i][j].text.strip() == "":
                            if t.cells[i][j].hspan and not t.cells[i][j].left:
                                t.cells[i][j].text = t.cells[i][j - 1].text
            elif f == "v":
                for i in range(len(t.cells)):
                    for j in range(len(t.cells[i])):
                        if t.cells[i][j].text.strip() == "":
                            if t.cells[i][j].vspan and not t.cells[i][j].top:
                                t.cells[i][j].text = t.cells[i - 1][j].text
        return t
