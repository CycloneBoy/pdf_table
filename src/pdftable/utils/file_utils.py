#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：file_utils
# @Author  ：cycloneboy
# @Date    ：20xx/6/21 14:24
import collections
import glob
import hashlib
import json
import os
import pickle
import shutil
from functools import partial
from typing import List, Union
from urllib.parse import urlparse as parse_url
from urllib.parse import uses_relative, uses_netloc, uses_params

import math

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")

from . import BaseUtil, logger, TimeUtils, Constants

'''
文件处理的工具类

'''


class FileUtils(BaseUtil):
    """
    文件工具类
    """

    def init(self):
        pass

    @staticmethod
    def get_content(path, encoding='gbk'):
        """
        读取文本内容
        :param path:
        :param encoding:
        :return:
        """
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            content = ''
            for l in f:
                l = l.strip()
                content += l
            return content

    @staticmethod
    def save_to_text(filename, content, mode='w'):
        """
        保存为文本
        :param filename:
        :param content:
        :return:
        """
        FileUtils.check_file_exists(filename)
        with open(filename, mode, encoding='utf-8') as f:
            f.writelines(content)

    @staticmethod
    def append_to_text(file_name, content: Union[str, List]):
        if isinstance(content, list):
            content = "\n".join(content)

        content = f"{content}\n"
        FileUtils.save_to_text(file_name, content, mode='a')

    @staticmethod
    def save_bert_vocab(file_path, vocab_dict):
        """
        保存vocab_dict

        :param file_path:
        :param vocab_dict:
        :return:
        """
        bert_vocab = [x for x, y in vocab_dict.items()]
        FileUtils.save_to_text(file_path, "\n".join(bert_vocab))

    @staticmethod
    def save_to_json(filename, content):
        """
        保存map 数据
        :param filename:
        :param maps:
        :return:
        """
        FileUtils.check_file_exists(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False)

    @staticmethod
    def load_json(filename) -> json:
        if not os.path.exists(filename):
            return dict()

        with open(filename, 'r', encoding='utf8') as f:
            return json.load(f)

    @staticmethod
    def load_json_line(filename, encoding='utf8') -> List:
        """
        读取json line 格式的数据
        :param filename:
        :param encoding:
        :return:
        """
        result = []
        if not os.path.exists(filename):
            return result

        with open(filename, 'r', encoding=encoding) as f:
            for line in f:
                result.append(json.loads(line))
        return result

    @staticmethod
    def dump_json(fp, obj, sort_keys=False, indent=4, show_info=False):
        try:
            fp = os.path.abspath(fp)
            if not os.path.exists(os.path.dirname(fp)):
                os.makedirs(os.path.dirname(fp))
            with open(fp, 'w', encoding='utf8') as f:
                json.dump(obj, f, ensure_ascii=False, sort_keys=sort_keys, indent=indent, separators=(',', ':'))
            if show_info:
                logger.info(f'json 文件保存成功，{fp}')
            return True
        except Exception as e:
            logger.info(f'json 文件 {obj} 保存失败, {e}')
            return False

    @staticmethod
    def dump_json_string(obj, sort_keys=False, ):
        """
        序列化 json string
        :param obj:
        :param sort_keys:
        :return:
        """
        json_str = json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, separators=(',', ':'))
        return json_str

    @staticmethod
    def dump_json_line(file_name, data, encoding='utf8', show_info=True):
        """
        保存json line 格式
        :param file_name:
        :param data:
        :param encoding:
        :param show_info:
        :return:
        """
        try:
            file_name = os.path.abspath(file_name)
            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))
            with open(file_name, 'w', encoding=encoding) as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                # json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ':'))
            if show_info:
                logger.info(f'json line 文件保存成功，{file_name}')
            return True
        except Exception as e:
            logger.info(f'json line 文件 {data} 保存失败, {e}')
            return False

    @staticmethod
    def get_file_name_list(path, type="*.txt"):
        """获取指定路径下的指定类型的所有文件"""
        files = glob.glob(os.path.join(path, type))
        return files

    @staticmethod
    def check_file_exists(filename, delete=False):
        """检查文件是否存在"""
        if filename is None:
            return False
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print("文件夹不存在,创建目录:{}".format(dir_name))
        return os.path.exists(filename)

    @staticmethod
    def read_to_text(path, encoding='utf-8'):
        """读取txt 文件"""
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
            return content

    @staticmethod
    def read_to_text_list(path, encoding='utf-8'):
        """
        读取txt文件,默认utf8格式,
        :param path:
        :param encoding:
        :return:
        """
        list_line = []
        if not os.path.exists(path):
            return list_line
        with open(path, 'r', encoding=encoding) as f:
            list_line = f.readlines()
            list_line = [row.rstrip("\n") for row in list_line]
            return list_line

    @staticmethod
    def list_file(file_dir, endswith="", add_dir=False):
        """
        获取目录下的 指定后缀的文件列表
        :param file_dir:
        :param endswith:
        :param add_dir:
        :return:
        """
        file_list = []
        if not os.path.exists(file_dir):
            return file_list

        for file_name in os.listdir(file_dir):
            if file_name.endswith(endswith):
                if add_dir:
                    file_list.append(os.path.join(file_dir, file_name))
                else:
                    file_list.append(file_name)

        return file_list

    @staticmethod
    def get_dir_sub_dir(file_dir, add_dir=False):
        """
        获取当前目录下的 所有子目录
        :param file_dir:
        :param add_dir:
        :return:
        """
        file_list = []
        if not os.path.exists(file_dir):
            return file_list

        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            file_path = os.path.abspath(file_path)
            if os.path.isdir(file_path):
                if add_dir:
                    file_list.append(file_path)
                else:
                    file_list.append(file_name)

        return file_list

    @staticmethod
    def delete_file_all(path):
        """
        删除一个目录下的所有文件
        :param path:
        :return:
        """

        for i in os.listdir(path):
            path_children = os.path.join(path, i)
            if os.path.isfile(path_children):
                os.remove(path_children)
            else:  # 递归, 删除目录下的所有文件
                FileUtils.delete_file(path_children)

    @staticmethod
    def delete_file_list(file_list, show_log=True):
        """
        删除一个列表

        :param file_list:
        :param show_log:
        :return:
        """
        for index, item in enumerate(file_list):
            FileUtils.delete_file(item, show_log=show_log)

    @staticmethod
    def list_dir_or_file(file_dir, add_parent=False, sort=False, start_with=None, is_dir=True,
                         end_with=None, ):
        """
        读取文件夹下的所有子文件夹
        :param file_dir:
        :param add_parent:
        :param sort:
        :param start_with:
        :param is_dir:
        :param end_with:
        :return:
        """
        dir_list = []
        if not os.path.exists(file_dir):
            return dir_list
        for name in os.listdir(file_dir):
            run_dir = os.path.join(file_dir, name)
            flag = os.path.isdir(run_dir) if is_dir else os.path.isfile(run_dir)
            if flag:
                if start_with is not None and not str(name).startswith(start_with):
                    continue
                if end_with is not None and not str(name).endswith(end_with):
                    continue
                if add_parent:
                    run_dir = os.path.join(file_dir, name)
                else:
                    run_dir = name
                dir_list.append(run_dir)

        if sort:
            dir_list.sort(key=lambda k: str(k), reverse=False)

        return dir_list

    @staticmethod
    def list_dir(file_dir, add_parent=False, sort=False, start_with=None):
        """
        读取文件夹下的所有子文件夹
        :param file_dir:
        :param add_parent:
        :param sort:
        :param start_with:
        :param is_dir:
        :return:
        """
        return FileUtils.list_dir_or_file(file_dir=file_dir, add_parent=add_parent, sort=sort, start_with=start_with,
                                          is_dir=True)

    @staticmethod
    def list_file_prefix(file_dir, add_parent=False, sort=False, start_with=None, end_with=None, ):
        """
        读取文件夹下的所有文件
        :param file_dir:
        :param add_parent:
        :param sort:
        :param start_with: 文件前缀
        :param end_with: 文件前缀
        :return:
        """
        return FileUtils.list_dir_or_file(file_dir=file_dir, add_parent=add_parent, sort=sort,
                                          start_with=start_with, end_with=end_with,
                                          is_dir=False)

    @staticmethod
    def dir_tree(filepath, ignore_dir_names=None, ignore_file_names=None, ignore_func=None):
        """
        读取 文件夹下的所有文件

        :param filepath:
        :param ignore_dir_names:
        :param ignore_file_names:
        :param ignore_func:
        :return:
        """
        if ignore_dir_names is None:
            ignore_dir_names = []
        if ignore_file_names is None:
            ignore_file_names = []
        if ignore_func is None:
            ignore_func = os.path.exists

        ret_list = []

        if isinstance(filepath, str):
            basename = os.path.basename(filepath)
            if not os.path.exists(filepath):
                print(f"路径不存在:{filepath}")
                return None, None
            elif os.path.isfile(filepath) and basename not in ignore_file_names and ignore_func(filepath):
                return [filepath], [basename]
            elif os.path.isdir(filepath) and basename not in ignore_dir_names:
                for file in os.listdir(filepath):
                    full_filepath = os.path.join(filepath, file)
                    full_basename = os.path.basename(full_filepath)
                    if os.path.isfile(full_filepath) \
                            and full_basename not in ignore_file_names \
                            and ignore_func(full_filepath):
                        ret_list.append(full_filepath)
                    if os.path.isdir(full_filepath) and full_basename not in ignore_dir_names:
                        ret_list.extend(FileUtils.dir_tree(full_filepath,
                                                           ignore_dir_names,
                                                           ignore_file_names,
                                                           ignore_func=ignore_func)[0])

        return ret_list, [os.path.basename(p) for p in ret_list]

    @staticmethod
    def delete_file(path, show_log=True):
        """
        删除一个文件
        :param path:
        :param show_log:
        :return:
        """
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
                if show_log:
                    logger.info(f"删除文件：{path}")

    @staticmethod
    def get_path_dir(file_dir, add_parent=False, sort=False, start_with=None):
        """
        读取文件夹下的所有子文件夹
        :param file_dir:
        :param add_parent:
        :param sort:
        :param start_with:
        :return:
        """
        dir_list = []
        # files_list = []
        for root, dirs, files in os.walk(file_dir):
            for run_dir in dirs:
                if start_with is not None and not str(run_dir).startswith(start_with):
                    continue
                if add_parent:
                    run_dir = os.path.join(file_dir, run_dir)
                dir_list.append(run_dir)
            # dir_list.extend(dirs)
            # files_list.extend(files)
        if sort:
            dir_list.sort(key=lambda k: str(k), reverse=False)

        return dir_list

    @staticmethod
    def save_to_pickle(model, file_name):
        """
        保存模型
        :param model:
        :param file_name:
        :return:
        """
        FileUtils.check_file_exists(file_name)
        pickle.dump(model, open(file_name, "wb"))

    @staticmethod
    def load_to_model(file_name):
        """
         使用pickle加载模型文件
        :param file_name:
        :return:
        """
        loaded_model = pickle.load(open(file_name, "rb"))
        return loaded_model

    @staticmethod
    def get_file_name(file_name, add_end=False):
        """
        获取文件名称
        :param file_name:
        :param add_end:
        :return:
        """
        if file_name is None or len(file_name) < 1:
            return ""

        begin_index = str(file_name).rfind("/")
        if add_end:
            return file_name[(begin_index + 1):]
        end_index = str(file_name).rfind(".")
        return file_name[(begin_index + 1):end_index] if len(file_name) > 1 else file_name

    @staticmethod
    def get_raw_file_name(file_name):
        """
        获取文件名称
        :param file_name:
        :return:
        """
        begin_index = str(file_name).rfind("/")
        return file_name[(begin_index + 1):]

    @staticmethod
    def remove_file_name_end(file_name):
        """
        移除文件后缀
        :param file_name:
        :return:
        """
        end_index = str(file_name).rfind(".")
        return file_name[0:end_index]

    @staticmethod
    def get_file_size(file_name):
        """
        获取文件大小
        :param file_name:
        :return:
        """
        file_size = os.path.getsize(file_name)
        return file_size

    @staticmethod
    def get_file_line(file_name):
        """
        获取文件 行数
        :param file_name:
        :return:
        """
        count = 0
        with open(file_name, encoding="utf-8") as f:
            for line in f:
                count += 1
        return count

    @staticmethod
    def get_dir_file_name(file_name):
        """
        获取文件目录名称
        :param file_name:
        :return:
        """
        return os.path.dirname(file_name)

    @staticmethod
    def read_file(file_name, block_size=1024 * 8):
        """
        读取文件
        :param block_size:
        :param file_name:
        :return:
        """

        count = 0
        with open(file_name) as fp:
            for chunk in FileUtils.chunked_file_reader(fp, block_size):
                count += 1
        return count

    @staticmethod
    def chunked_file_reader(fp, block_size=1024 * 8):
        """生成器函数：分块读取文件内容，使用 iter 函数
        """
        # 首先使用 partial(fp.read, block_size) 构造一个新的无需参数的函数
        # 循环将不断返回 fp.read(block_size) 调用结果，直到其为 '' 时终止
        for chunk in iter(partial(fp.read, block_size), ''):
            yield chunk

    @staticmethod
    def read_lines(file_name, encoding='utf-8'):
        """
        给定一个文件路径，读取文件并返回一个迭代器，这个迭代器将顺序返回文件的每一行
        """
        with open(file_name, 'r', encoding=encoding) as f:
            for line in f:
                yield line

    @staticmethod
    def get_parent_dir_name(file_name):
        """
        获取文件父目录名称
        :param file_name:
        :return:
        """
        dir_name = FileUtils.get_dir_file_name(file_name)
        begin_index = str(dir_name).rfind("/")
        return dir_name[(begin_index + 1):]

    @staticmethod
    def copy_file(src, dst, cp_metadata=False):
        """
        拷贝文件
        :param src:
        :return:
        """
        file_name = os.path.basename(src)
        dst_file_name = os.path.join(dst, file_name)

        if src is None or len(src) == 0 or not os.path.exists(src):
            return dst_file_name
        FileUtils.check_file_exists(os.path.join(dst, "tmp.txt"))

        if cp_metadata:
            # Copy files, but preserve metadata (cp -p src dst)
            shutil.copy2(src, dst)
        else:
            #  Copy src to dst. (cp src dst)
            shutil.copy(src, dst)

        file_name = os.path.basename(src)
        dst_file_name = os.path.join(dst, file_name)
        return dst_file_name

    @staticmethod
    def copy_file_rename(src, dst):
        """
        拷贝文件
        :param src:
        :return:
        """
        if src is None or len(src) == 0 or not os.path.exists(src) or os.path.exists(dst):
            return dst
        FileUtils.check_file_exists(dst)

        shutil.copyfile(src, dst)

        file_name = os.path.basename(src)
        dst_file_name = os.path.join(dst, file_name)
        return dst_file_name

    @staticmethod
    def copy_dir(src, dst, symlinks=True):
        """
        拷贝目录
        :param src:
        :return:
        """
        FileUtils.check_file_exists(os.path.join(dst, "tmp.txt"))
        # Copy directory tree (cp -R src dst)
        shutil.copytree(src, dst, symlinks)

    @staticmethod
    def move_file(src, dst):
        """
        移动文件
        :param src:
        :return:
        """
        FileUtils.check_file_exists(os.path.join(dst, "tmp.txt"))
        shutil.move(src, dst)

    @staticmethod
    def add_time_sub_dir(save_dir):
        """
        添加时间子目录： %Y-%m-%d
        :param save_dir:
        :return:
        """
        file_path = os.path.join(save_dir, TimeUtils.get_time())
        return file_path

    @staticmethod
    def load_set_file(path):
        words = set()
        with open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w)
        return words

    @staticmethod
    def load_tag_from_list(tag_list):
        """
            读取tag 映射
        :param tag_list:
        :return:
        """
        label2id = {}
        id2label = {}
        for index, item in enumerate(tag_list):
            label2id[item] = index
            id2label[index] = item
        return label2id, id2label

    @staticmethod
    def get_checkpoint_num(file_path):
        """
        获取 checkpoint_num
        :param file_path:
        :return:
        """
        dir_name = FileUtils.get_parent_dir_name(f"{file_path}/test.txt")
        check_num = str(dir_name).split("-")[1]
        return int(check_num)

    @staticmethod
    def get_step_num(file_name, index=2):
        """
        获取 epoch_step
        :param file_name: eval_0_7000_metric.json
        :param index:
        :return:
        """
        raw_file_name = FileUtils.get_file_name(file_name)
        check_num = str(raw_file_name).split("_")[index]
        res = math.floor(float(check_num))
        return res

    @staticmethod
    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    @staticmethod
    def read_table_file(file_name, output_indexes, sep='\t', need_tokenize=False, skip_first=0):
        """
        读取 csv table 文件
        :param file_name:
        :param output_indexes:
        :param sep:
        :param need_tokenize:
        :param skip_first:
        :return:
        """
        outputs = []

        if not FileUtils.check_file_exists(file_name):
            return outputs

        require_len = max(output_indexes) + 1
        with open(file_name, encoding='utf-8') as f:
            for line in f:
                items = line.split(sep)
                if len(items) < require_len:
                    continue
                output = []
                for index in output_indexes:
                    line_src = items[index].strip()
                    if need_tokenize:
                        line_src = line_src.split()
                    output.append(line_src)
                outputs.append(output)

        return outputs[skip_first:]

    @staticmethod
    def get_label_mapping(labels):
        """
            label2id, id2label
        :param labels:
        :return:
        """
        label2id = {}
        id2label = {}
        for index, item in enumerate(labels):
            label2id[item] = index
            id2label[index] = item
        return label2id, id2label

    @staticmethod
    def md5file(fname):
        """
        文件内容md5
        :param fname:
        :return:
        """
        hash_md5 = hashlib.md5()
        f = open(fname, "rb")
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
        f.close()
        return hash_md5.hexdigest()

    @staticmethod
    def is_url(url):
        """Check to see if a URL has a valid protocol.
            # https://github.com/pandas-dev/pandas/blob/master/pandas/io/common.py

        Parameters
        ----------
        url : str or unicode

        Returns
        -------
        isurl : bool
            If url has a valid protocol return True otherwise False.

        """
        try:
            return parse_url(url).scheme in _VALID_URLS
        except Exception:
            return False

    @staticmethod
    def delete_dir_file(filepath, ignore_func):
        """
        删除目录 下的指定文件

        :param filepath:
        :param ignore_func:
        :return:
        """
        file_list, name_list = FileUtils.dir_tree(filepath=filepath, ignore_func=ignore_func)

        logger.info(f"delete_dir_file begin file_list: {filepath} - {len(file_list)}")

        for file_name in file_list:
            FileUtils.delete_file(file_name)

        logger.info(f"delete_dir_file finish file_list: {filepath} - {len(file_list)}")

    @staticmethod
    def get_pdf_to_image_file_name(file_name):
        """
        获取pdf to image file name

        :param file_name:
        :return:
        """
        if file_name is None:
            return None

        image_name = file_name
        if FileUtils.is_pdf_file(file_name):
            image_name = (str(file_name).replace(".pdf", ".png")
                          .replace(".PDF", ".png"))
        return image_name

    @staticmethod
    def is_pdf_file(file_name):
        """
        判断是否是PDF
        :param file_name:
        :return:
        """
        if file_name is None:
            return False
        is_pdf = str(file_name).lower().endswith(".pdf")
        return is_pdf

    @staticmethod
    def delete_ocr_result_file(output_dir, raw_filename, file_name_dict=None):
        """
        删除OCR 中间文件
        :param output_dir:
        :param raw_filename:
        :param file_name_dict:
        :return:
        """
        if file_name_dict is None:
            file_name_dict = [
                {"start_with": raw_filename, "end_with": ".jpg"},
                {"start_with": raw_filename, "end_with": ".png"},
                {"start_with": f"ocr_{raw_filename}", "end_with": ".jpg"},
            ]

        file_list_all = []
        for config in file_name_dict:
            file_list = FileUtils.list_file_prefix(file_dir=output_dir,
                                                   add_parent=True,
                                                   start_with=config["start_with"],
                                                   end_with=config["end_with"])

            file_list_all.extend(file_list)

        # FileUtils.delete_file_list(file_list_all, show_log=False)
        logger.info(f"删除中间文件:{len(file_list_all)} 个文件。{file_list_all}")

    @staticmethod
    def read_wandb_username():
        user_name = ""
        return user_name

    @staticmethod
    def load_table_label_txt(txt_path):
        """
        加载 table label

        :param txt_path:
        :return:
        """
        pred_html_dict = {}
        if not os.path.exists(txt_path):
            return pred_html_dict
        with open(txt_path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                img_name, pred_html = line
                pred_html_dict[img_name] = pred_html
        return pred_html_dict

    @staticmethod
    def check_file_name_end_model_weight(file_name):
        path, ext = os.path.splitext(file_name)
        flag = ext in [".pdparams", ".pth", ".bin", ".pt", ".safetensors"]
        return flag

    @staticmethod
    def get_output_dir_with_time(add_now_end=True):
        output_dir = f"{Constants.HTML_BASE_DIR}/pdf_debug/{TimeUtils.get_time()}"
        if add_now_end:
            output_dir = os.path.join(output_dir, TimeUtils.now_str_short())
        return output_dir
