#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_extractor
# @Author  ：cycloneboy
# @Date    ：20xx/5/31 13:39
import re
import warnings

from .pdf_handlers import PDFHandler
from ...utils import (
    Constants,
    TimeUtils,
    PdfUtils,
    FileUtils,
    MatchUtils
)

__all__ = [
    'TableExtractor',
    'read_pdf',
    "check_imaged_pdf",
]

"""
pdf 提取工具
"""

stream_kwargs = [
    "columns",
    "edge_tol",
    "row_tol",
    "column_tol",
]
lattice_kwargs = [
    "process_background",
    "line_scale",
    "copy_text",
    "shift_text",
    "line_tol",
    "joint_tol",
    "threshold_blocksize",
    "threshold_constant",
    "iterations",
    "resolution",
]

PATTERN_PDF_FILE = re.compile(r"\.(png|pdf|xlsx|json)$", flags=re.I)


def ignore_func(filename):
    return MatchUtils.match_pattern_flag(filename, pattern=PATTERN_PDF_FILE)


class TableExtractor(object):

    @staticmethod
    def validate_input(kwargs, flavor="lattice"):

        def check_intersection(parser_kwargs, input_kwargs):
            isec = set(parser_kwargs).intersection(set(input_kwargs.keys()))
            if isec:
                raise ValueError(
                    f"{','.join(sorted(isec))} cannot be used with flavor='{flavor}'"
                )

        if flavor == "lattice":
            check_intersection(stream_kwargs, kwargs)
        else:
            check_intersection(lattice_kwargs, kwargs)

    @staticmethod
    def remove_extra(kwargs, flavor="lattice"):
        if flavor == "lattice":
            for key in kwargs.keys():
                if key in stream_kwargs:
                    kwargs.pop(key)
        else:
            for key in kwargs.keys():
                if key in lattice_kwargs:
                    kwargs.pop(key)
        return kwargs

    @staticmethod
    def read_pdf(
            filepath,
            pages="1",
            password=None,
            flavor="pdf",
            suppress_stdout=False,
            layout_kwargs={},
            temp_dir=None,
            delete_temp_dir=True,
            output_dir=None,
            src_id=None,
            delete_check_success=False,
            **kwargs
    ):
        """Read PDF and return extracted tables.

        Note: kwargs annotated with ^ can only be used with flavor='stream'
        and kwargs annotated with * can only be used with flavor='lattice'.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        pages : str, optional (default: '1')
            Comma-separated page numbers.
            Example: '1,3,4' or '1,4-end' or 'all'.
        password : str, optional (default: None)
            Password for decryption.
        flavor : str (default: 'lattice')
            The parsing method to use ('lattice' or 'stream').
            Lattice is used by default.
        suppress_stdout : bool, optional (default: True)
            Print all logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs.
        table_areas : list, optional (default: None)
            List of table area strings of the form x1,y1,x2,y2
            where (x1, y1) -> left-top and (x2, y2) -> right-bottom
            in PDF coordinate space.
        columns^ : list, optional (default: None)
            List of column x-coordinates strings where the coordinates
            are comma-separated.
        split_text : bool, optional (default: False)
            Split text that spans across multiple cells.
        flag_size : bool, optional (default: False)
            Flag text based on font size. Useful to detect
            super/subscripts. Adds <s></s> around flagged text.
        strip_text : str, optional (default: '')
            Characters that should be stripped from a string before
            assigning it to a cell.
        row_tol^ : int, optional (default: 2)
            Tolerance parameter used to combine text vertically,
            to generate rows.
        column_tol^ : int, optional (default: 0)
            Tolerance parameter used to combine text horizontally,
            to generate columns.
        process_background* : bool, optional (default: False)
            Process background lines.
        line_scale* : int, optional (default: 15)
            Line size scaling factor. The larger the value the smaller
            the detected lines. Making it very large will lead to text
            being detected as lines.
        copy_text* : list, optional (default: None)
            {'h', 'v'}
            Direction in which text in a spanning cell will be copied
            over.
        shift_text* : list, optional (default: ['l', 't'])
            {'l', 'r', 't', 'b'}
            Direction in which text in a spanning cell will flow.
        line_tol* : int, optional (default: 2)
            Tolerance parameter used to merge close vertical and horizontal
            lines.
        joint_tol* : int, optional (default: 2)
            Tolerance parameter used to decide whether the detected lines
            and points lie close to each other.
        threshold_blocksize* : int, optional (default: 15)
            Size of a pixel neighborhood that is used to calculate a
            threshold value for the pixel: 3, 5, 7, and so on.

            For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
        threshold_constant* : int, optional (default: -2)
            Constant subtracted from the mean or weighted mean.
            Normally, it is positive but may be zero or negative as well.

            For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
        iterations* : int, optional (default: 0)
            Number of times for erosion/dilation is applied.

            For more information, refer `OpenCV's dilate <https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#dilate>`_.
        resolution* : int, optional (default: 300)
            Resolution used for PDF to PNG conversion.

        Returns
        -------
        tables : camelot.core.TableList

        """
        if flavor not in ["lattice", "stream", "pdf"]:
            raise NotImplementedError(
                "Unknown flavor specified." " Use either 'lattice' or 'stream'"
            )

        with warnings.catch_warnings():
            if suppress_stdout:
                warnings.simplefilter("ignore")

            TableExtractor.validate_input(kwargs, flavor=flavor)
            p = PDFHandler(filepath, pages=pages, password=password,
                           output_dir=output_dir,
                           src_id=src_id,
                           delete_check_success=delete_check_success)
            kwargs = TableExtractor.remove_extra(kwargs, flavor=flavor)
            tables = p.parse(
                flavor=flavor,
                suppress_stdout=suppress_stdout,
                layout_kwargs=layout_kwargs,
                temp_dir=temp_dir,
                delete_temp_dir=delete_temp_dir,
                **kwargs
            )
            return tables

    @staticmethod
    def check_imaged_pdf(
            filepath,
            pages="1",
            password=None,
            flavor="pdf",
            suppress_stdout=False,
            layout_kwargs={},
            temp_dir=None,
            delete_temp_dir=True,
            output_dir=None,
            src_id=None,
            delete_check_success=False,
            **kwargs
    ):
        """
        check imaged pdf

        :param filepath:
        :param pages:
        :param password:
        :param flavor:
        :param suppress_stdout:
        :param layout_kwargs:
        :param temp_dir:
        :param delete_temp_dir:
        :param output_dir:
        :param src_id:
        :param delete_check_success:
        :param kwargs:
        :return:
        """
        with warnings.catch_warnings():
            if suppress_stdout:
                warnings.simplefilter("ignore")

            TableExtractor.validate_input(kwargs, flavor=flavor)
            handler = PDFHandler(filepath, pages=pages, password=password,
                                 output_dir=output_dir,
                                 src_id=src_id,
                                 delete_check_success=delete_check_success)
            kwargs = TableExtractor.remove_extra(kwargs, flavor=flavor)
            is_imaged_pdf = handler.check_imaged_pdf(
                flavor=flavor,
                suppress_stdout=suppress_stdout,
                layout_kwargs=layout_kwargs,
                temp_dir=temp_dir,
                delete_temp_dir=delete_temp_dir,
                **kwargs
            )
            return is_imaged_pdf


def read_pdf(
        filepath,
        pages="1",
        password=None,
        flavor="pdf",
        suppress_stdout=False,
        layout_kwargs={},
        temp_dir=None,
        delete_temp_dir=True,
        output_dir=None,
        src_id=None,
        delete_check_success=True,
        pdf_dir=None,
        debug=True,
        **kwargs
):
    base_dir = f"{Constants.PDF_CACHE_BASE}/table_file"
    # base_dir = Constants.OUTPUT_DIR
    if output_dir is None:
        src_temp = f"/{src_id}" if src_id is not None else ""
        output_dir = f"{base_dir}/inference_results/{TimeUtils.get_time()}{src_temp}/{TimeUtils.now_str_short()}"
    if pdf_dir is None:
        pdf_dir = f"{base_dir}/pdf_file"

    if temp_dir is None:
        temp_dir = output_dir

    file_name = PdfUtils.download_pdf_before_check(file_url=filepath, pdf_dir=pdf_dir)

    tables = TableExtractor.read_pdf(file_name, flavor=flavor,
                                     pages=pages,
                                     password=password,
                                     temp_dir=temp_dir,
                                     suppress_stdout=suppress_stdout,
                                     layout_kwargs=layout_kwargs,
                                     delete_temp_dir=delete_temp_dir,
                                     output_dir=output_dir,
                                     src_id=src_id,
                                     delete_check_success=delete_check_success,
                                     **kwargs
                                     )
    if debug:
        out_file = f"{output_dir}/{FileUtils.get_file_name(file_name)}"
        FileUtils.check_file_exists(f'{out_file}.html')
        tables.export(f'{out_file}.html', f='html', compress=False)
        # tables.export(f'{out_file}.json', f='json', compress=False)
        # tables.export(f'{out_file}.xlsx', f='excel', compress=False)

    # 删除pdf
    FileUtils.delete_dir_file(filepath=output_dir, ignore_func=ignore_func)

    return tables


def check_imaged_pdf(
        filepath,
        pages="1",
        password=None,
        flavor="pdf",
        suppress_stdout=False,
        layout_kwargs={},
        temp_dir=None,
        delete_temp_dir=True,
        output_dir=None,
        src_id=None,
        pdf_dir=None,
        delete_check_success=False,
        **kwargs
):
    """
    check imaged pdf

    :param filepath:
    :param pages:
    :param password:
    :param flavor:
    :param suppress_stdout:
    :param layout_kwargs:
    :param temp_dir:
    :param delete_temp_dir:
    :param output_dir:
    :param src_id:
    :param pdf_dir:
    :param delete_check_success:
    :param kwargs:
    :return:
    """
    base_dir = Constants.PDF_CACHE_BASE
    # base_dir = Constants.OUTPUT_DIR
    if output_dir is None:
        src_temp = f"/{src_id}" if src_id is not None else ""
        output_dir = f"{base_dir}/inference_results/check_imaged_pdf/{TimeUtils.get_time()}{src_temp}/{TimeUtils.now_str_short()}"
    if pdf_dir is None:
        pdf_dir = f"{base_dir}/pdf_file"

    if temp_dir is None:
        temp_dir = output_dir

    file_name = PdfUtils.download_pdf_before_check(file_url=filepath, pdf_dir=pdf_dir)

    is_imaged_pdf = TableExtractor.check_imaged_pdf(file_name, flavor=flavor,
                                                    pages=pages,
                                                    password=password,
                                                    temp_dir=temp_dir,
                                                    suppress_stdout=suppress_stdout,
                                                    layout_kwargs=layout_kwargs,
                                                    delete_temp_dir=delete_temp_dir,
                                                    output_dir=output_dir,
                                                    src_id=src_id,
                                                    delete_check_success=delete_check_success,
                                                    **kwargs
                                                    )

    return is_imaged_pdf
