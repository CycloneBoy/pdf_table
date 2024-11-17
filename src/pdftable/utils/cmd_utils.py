#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：cmd_utils.py
# @Author  ：cycloneboy
# @Date    ：20xx/12/23 10:28
import subprocess
import traceback

from .base_utils import BaseUtil


class CmdUtils(BaseUtil):

    def init(self):
        pass

    @staticmethod
    def run_cmd(cmd):
        """

        :param cmd:
        :return:
        """
        r = None
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            r = p.stdout.read()
        except Exception as e:
            traceback.print_exc()
        return r
