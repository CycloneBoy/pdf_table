#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : __init__.py
# @Author   : cycloneboy
# @Date     : 20xx/8/18 - 22:44

from .configuration_ocr_document import *

from .base_infer_task import *
from .ocr_detection_task import *
# from .ocr_recognition_task import OcrRecognitionTask
from .ocr_table_structure_task import *
from .ocr_layout_task import *
from .ocr_table_cell_task import *
from .ocr_pdf_text_task import *

from .ocr_table_to_html_task import *
from .ocr_to_html_task import *

from .modeling_ocr_pdf import *
