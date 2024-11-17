#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：pdf_image_processor
# @Author  ：cycloneboy
# @Date    ：20xx/5/29 16:18

from ...utils.table import GhostscriptBackend
from .poppler_backend import PopplerBackend, PopplerBackendV2

__all__ = [
    "BACKENDS",
]

"""
PDF 转IMAGE 
"""
BACKENDS = {
    "poppler": PopplerBackend,
    "poppler_v2": PopplerBackendV2,
    "ghostscript": GhostscriptBackend
}


class ImageConversionBackend(object):
    def __init__(self, backend="poppler", use_fallback=True):
        if backend not in BACKENDS.keys():
            raise ValueError(f"Image conversion backend '{backend}' not supported")

        self.backend = backend
        self.use_fallback = use_fallback
        self.fallbacks = list(filter(lambda x: x != backend, BACKENDS.keys()))

    def convert(self, pdf_path, png_path):
        try:
            converter = BACKENDS[self.backend]()
            converter.convert(pdf_path, png_path)
        except Exception as e:
            import sys

            if self.use_fallback:
                for fallback in self.fallbacks:
                    try:
                        converter = BACKENDS[fallback]()
                        converter.convert(pdf_path, png_path)
                    except Exception as e:
                        raise type(e)(
                            str(e) + f" with image conversion backend '{fallback}'"
                        ).with_traceback(sys.exc_info()[2])
                        continue
                    else:
                        break
            else:
                raise type(e)(
                    str(e) + f" with image conversion backend '{self.backend}'"
                ).with_traceback(sys.exc_info()[2])
