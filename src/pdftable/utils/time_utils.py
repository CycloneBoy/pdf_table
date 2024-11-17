#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：time_utils.py
# @Author  ：cycloneboy
# @Date    ：20xx/10/29 9:44
import datetime
import time

import numpy as np

from .base_utils import BaseUtil


class Timer:
    """记录多次运行时间。"""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

    def to_show(self):
        """显示最近一次时间"""
        return TimeUtils.time_use_str(self.stop())

    def to_show_avg(self):
        """显示平均时间"""
        return TimeUtils.time_use_str(self.avg())

    def to_show_speed(self, total):
        """显示每次平均时间"""
        return TimeUtils.time_use_str(total / self.sum())


class TimeUtils(BaseUtil):
    """
    时间日期工具类

    """

    def init(self):
        pass

    @staticmethod
    def time_use_str(use_time):
        """
        使用时间转字符串
        :param use_time:
        :return:
        """
        if use_time * 1000 < 1000:
            use_time_str = "{:.3f} ms".format(use_time * 1000)
        # elif 1000 <= use_time < 1000 * 600:
        #     use_time_str = "{:.3f} s".format(use_time)
        else:
            use_time_str = "{:.3f} s".format(use_time)
            # use_time_str = TimeUtils.format_time(use_time)

        return use_time_str

    @staticmethod
    def now():
        return datetime.datetime.now()

    @staticmethod
    def now_str(time_format="%Y-%m-%d %H:%M:%S"):
        return time.strftime(time_format, time.localtime())

    @staticmethod
    def now_str_short(time_format="%Y%m%d_%H%M%S"):
        return TimeUtils.now_str(time_format)

    @staticmethod
    def format_time(use_time, time_format="%Y-%m-%d %H:%M:%S.%f"):
        """
        时间戳 格式化
        :param use_time:  timestamp = 1570774556514
        :param time_format:
        :return:
        """
        if isinstance(use_time, datetime.datetime):
            d_time = use_time
        else:

            if len(str(use_time)) > 10:
                use_time = use_time / 1000
            d_time = datetime.datetime.fromtimestamp(use_time)
        show_time = d_time.strftime(time_format)
        return show_time

    @staticmethod
    def get_time():
        return time.strftime("%Y-%m-%d", time.localtime())

    @staticmethod
    def calc_diff_time(begin_time, end_time, time_format="%Y-%m-%d %H:%M:%S"):
        """
        计算两个时间字符串的时间差
        :param begin_time:
        :param end_time:
        :param time_format:
        :return:
        """
        t1 = datetime.datetime.strptime(begin_time, time_format)
        t2 = datetime.datetime.strptime(end_time, time_format)

        diff_time = t2.timestamp() - t1.timestamp()

        return diff_time

    @staticmethod
    def get_time_dif(start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return datetime.timedelta(seconds=int(round(time_dif)))

    @staticmethod
    def to_timestamp(time_str, time_format="%Y-%m-%d"):
        """
        时间转换

        :param time_str:
        :param time_format:
        :return:
        """
        t1 = 0
        if not time_str:
            return t1

        if isinstance(time_str, datetime.datetime):
            t1 = int(time_str.timestamp())
        elif isinstance(time_str, str):
            if str(time_str).find('-') > -1:
                if len(str(time_str)) >= 19:
                    time_format = "%Y-%m-%d %H:%M:%S"
                    time_str = time_str[:19]
                else:
                    time_str = time_str[:10]
                raw_time = datetime.datetime.strptime(time_str, time_format)
                t1 = int(raw_time.timestamp())
            else:
                t1 = int(time_str)

        return t1
