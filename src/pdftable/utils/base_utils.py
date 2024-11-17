#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：base_utils.py
# @Author  ：cycloneboy
# @Date    ：20xx/6/21 13:58

from abc import abstractmethod, ABC

__all__ = [
    "BaseUtil",
]

from enum import Enum

"""
工具类基类

"""


class BaseUtil(ABC):
    """
    抽取数据基类
    """

    @abstractmethod
    def init(self):
        """
        工具类初始化
        :return:
        """
        pass


