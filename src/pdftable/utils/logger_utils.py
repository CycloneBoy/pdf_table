#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : logger_utils.py
# @Author  ：cycloneboy
# @Date  : 20xx/9/17 - 下午10:14

import logging
import os
import time
from logging import handlers
from pathlib import Path

from . import Constants


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s %(levelname)s %(pathname)s[%(lineno)d]: %(message)s'):
        filename_str = filename
        self.filename = filename
        self.logger = logging.getLogger(filename_str)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename_str, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


def get_time():
    return time.strftime("%Y-%m-%d", time.localtime())


def get_log_file_name(file_name):
    # file_name = str(f"{LOG_DIR}/{file_name}")
    index = str(file_name).rindex(".")
    pre_name = str(file_name)[:index]
    last_name = str(file_name)[index:]
    log_file = "{}_{}{}".format(pre_name, get_time(), last_name)
    print(log_file)
    dir_name = os.path.dirname(log_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return log_file


def get_log(filename, level='info'):
    my_file_name = "{}/data/log/{}".format(Constants.SRC_HOME_DIR, filename)
    my_logger = Logger(my_file_name, level=level)
    return my_logger.logger


def init_logger1(filename='test'):
    log = Logger('log/{}_{}.log'.format(filename, get_time()), level='debug')
    return log.logger


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)

    log_format = logging.Formatter(fmt='%(asctime)s %(levelname)s %(pathname)s[%(lineno)d]: %(message)s',
                                   datefmt='%Y-%d-%m %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


# logger = init_logger("test.log")
logger = Logger(filename=get_log_file_name(Constants.LOG_FILE), level=Constants.LOG_LEVEL).logger

if __name__ == '__main__':
    # get_log("info2.log").info("test")
    file_name = "../info2.log"
    index = str(file_name).rindex(".")
    print(file_name[:index])
    logger.info('test')
