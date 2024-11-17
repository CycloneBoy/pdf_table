#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：request_utils.py
# @Author  ：cycloneboy
# @Date    ：20xx/5/31 17:31
import json
import os
import random
import shutil
import string
import tempfile
from urllib.request import Request, urlopen

import requests

from .base_utils import BaseUtil

"""
request 相关工具类
"""


class RequestUtils(BaseUtil):
    """
    request 相关工具类
    """
    common_header = {'Content-Type': 'application/json;charset=utf8'}

    def init(self):
        pass

    def build_header(self):
        pass

    @staticmethod
    def post(url, data=None):
        """
        发送 post 请求
        :param url:
        :param data:
        :return:
        """
        headers = RequestUtils.common_header
        http_response = requests.post(url, data=json.dumps(data), headers=headers)
        if http_response.status_code == 200:
            response = json.loads(http_response.text)
            return response
        else:
            return None

    @staticmethod
    def random_string(length):
        ret = ""
        while length:
            ret += random.choice(
                string.digits + string.ascii_lowercase + string.ascii_uppercase
            )
            length -= 1
        return ret

    @staticmethod
    def download_pdf_from_url(url):
        """Download file from specified URL.

        Parameters
        ----------
        url : str or unicode

        Returns
        -------
        filepath : str or unicode
            Temporary filepath.

        """
        filename = f"{RequestUtils.random_string(6)}.pdf"
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            headers = {"User-Agent": "Mozilla/5.0"}
            request = Request(url, None, headers)
            obj = urlopen(request)
            content_type = obj.info().get_content_type()
            if content_type != "application/pdf":
                raise NotImplementedError("File format not supported")
            f.write(obj.read())
        filepath = os.path.join(os.path.dirname(f.name), filename)
        shutil.move(f.name, filepath)
        return filepath
