#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：constant
# @Author  ：cycloneboy
# @Date    ：20xx/6/21 13:59
import os

from dotenv import load_dotenv

CURRENT_ABS_PATH = os.path.abspath(__file__)
PDF_TABLE_ABS_PATH = os.path.abspath(os.path.dirname(CURRENT_ABS_PATH) + os.path.sep + "..")
SRC_ABS_PATH = os.path.abspath(os.path.dirname(PDF_TABLE_ABS_PATH) + os.path.sep + "..")

TABLE_ABS_PATH = os.path.abspath(os.path.join(PDF_TABLE_ABS_PATH, "model/table"))

load_dotenv(os.getenv("PDF_TABLE_ENV", os.path.join(SRC_ABS_PATH, ".env")))


def get_user_home():
    return os.path.expanduser("~")


def get_value_from_env_or_default(default: str, env_key: str = None):
    if env_key is None:
        return default
    return default if not os.getenv(env_key) else os.getenv(env_key)


USER_HOME = get_user_home()


class Constants(object):
    """
    常量工具类
    """

    @staticmethod
    def getenv(env_key: str, default="") -> str:
        return get_value_from_env_or_default(default=default, env_key=env_key)

    USER_HOME = get_user_home()

    PDFTABLE_USE_MODELSCOPE_HUB = getenv("PDFTABLE_USE_MODELSCOPE_HUB", "0").lower() in ["true", "1"]
    PDFTABLE_BASE_OUTPUT_DIR = getenv("PDFTABLE_BASE_OUTPUT_DIR", default=f"{USER_HOME}/.cache/pdftable")

    OUTPUT_DIR = f"{PDFTABLE_BASE_OUTPUT_DIR}/outputs"
    DATA_DIR = f"{PDFTABLE_BASE_OUTPUT_DIR}/data"

    SRC_HOME_DIR = getenv("SRC_HOME_DIR", default=SRC_ABS_PATH)
    SRC_DATA_HOME_DIR = f"{SRC_HOME_DIR}/data"
    SRC_IMAGE_DIR = f"{SRC_DATA_HOME_DIR}/image"

    LOG_LEVEL = "debug"
    LOG_FILE = f"{OUTPUT_DIR}/logs/run.log"

    PDF_CACHE_BASE = f"{OUTPUT_DIR}/pdf"
    HTML_BASE_DIR = f"{PDF_CACHE_BASE}/inference_results"
    PDF_PAGE_DIR = f"{HTML_BASE_DIR}/pdf_image/pdf_page_cache"

    NUMERALS_ZH_DICT = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                        '百': 100, '千': 1000, '万': 10000, '亿': 100000000}

    ####################################################################################
    # ocr
    #
    ####################################################################################
    SCOPE_MODEL_BASE_DIR = os.path.join(getenv("MODELSCOPE_CACHE", f"{USER_HOME}/.cache/modelscope"), "hub")
    HF_HUB_BASE_DIR = getenv("HF_HUB_CACHE", f"{USER_HOME}/.cache/huggingface/hub")
    OCR_FONT_BASE_DIR = f"{SCOPE_MODEL_BASE_DIR}/cycloneboy/pdftable_config/fonts"

    FONT_CONFIG = {
        "ch": "chinese_cht.ttf",
        "chinese_cht": "chinese_cht.ttf",
        "korean": "korean.ttf",
        "japan": "japan.ttc",
        "arabic": "arabic.ttf",
        "cyrillic": "cyrillic.ttf",
        "latin": "latin.ttf",
    }
    DEFAULT_FONT_DIR = f"{OCR_FONT_BASE_DIR}/chinese_cht.ttf"

    WANDB_LOG_DIR = f"{OUTPUT_DIR}/wandb"
