#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：common_utils
# @Author  ：cycloneboy
# @Date    ：20xx/6/21 15:04
import json
import os
import re
from collections import defaultdict, OrderedDict
from typing import Union, List, Tuple

import numpy as np
from transformers import TrainingArguments, HfArgumentParser

from pdftable.entity import HtmlTableCompareType
from pdftable.entity.common_entity import ModelArguments, DataTrainingArguments
from pdftable.utils import BaseUtil, FileUtils, logger, Constants
from pdftable.utils.match_utils import MatchUtils

"""
通用工具类
"""


class CommonUtils(BaseUtil):
    """
    通用工具类
    """

    def init(self):
        pass

    @staticmethod
    def sort_dict(label_dict):
        """
        对一个字典的 val列表 排序，并转换成 字典
        :param label_dict:
        :return:
        """
        label_key_dict = {}
        for key, val in label_dict.items():
            val_dict = defaultdict(int)
            for char_i in val:
                val_dict[char_i] += 1
            label_key_dict[key] = CommonUtils.sorted_dict(val_dict)

        # 对错别字排序
        result_dict = CommonUtils.sorted_dict(label_key_dict, key=lambda x: sum(x[1].values()))
        return result_dict

    @staticmethod
    def sorted_dict(label_dict, key=lambda x: x[1], reverse=True):
        """
        对词典进行排序
        :param label_dict:
        :param key:
        :param reverse:
        :return:
        """
        sort_list = sorted(label_dict.items(), key=key, reverse=reverse)
        sort_dict = OrderedDict()
        for row in sort_list:
            sort_dict[row[0]] = row[1]
        return sort_dict

    @staticmethod
    def clean_sentence(sentence):
        "clean one line from pdf"
        # result = re.sub("\u3000*\n*", "", sentence).replace("\x82", "")
        result = re.sub(r"\x82", "", sentence)
        result = re.sub(r"\u200b", "", result)
        result = re.sub(r"\ue004", "", result)
        result = re.sub(r"\ue008", "", result)
        result = re.sub(r"\ue009", "", result)
        result = re.sub(r"\u3000", " ", result)
        result = re.sub(r" ", " ", result)
        result = re.sub(r" ", " ", result)
        result = re.sub(r" ", " ", result)
        result = re.sub(r"&nbsp", " ", result)
        result = re.sub(r"&NBSP", " ", result)
        result = re.sub(r"﻿", " ", result)
        result = result.strip("\r")
        result = result.strip("\n")
        result = result.strip("\r")

        # 移除 页码
        result = MatchUtils.PATTERN_REPLACE_LAW_PAGE.sub("", result)
        # 移除 打印/关闭窗口
        result = MatchUtils.PATTERN_VIOLATION_FILTER_LINE_2.sub("", result)

        # 同一行中的换行符用空格替换
        # one_line_sep = str(result).count("\n")
        # two_line_sep = str(result).count("\n\n")
        # if two_line_sep < 3:
        #     result = str(result).replace('\n', ' ')

        return result

    @staticmethod
    def clean_sentence_remove_space(sentence):
        """
        清理句子

        :param sentence:
        :return:
        """
        sentence_clean = re.sub(r"\s", "", CommonUtils.clean_sentence(sentence))
        return sentence_clean

    @staticmethod
    def get_torch_device(use_gpu=True):
        import torch
        device = torch.device("cuda:0") if torch.cuda.is_available() and use_gpu else torch.device("cpu")

        return device

    @staticmethod
    def print_model_param(model, show_model=True, use_numpy=False,
                          show_info=True, save_dir=None,
                          paddle_model=False):
        """
         打印出每一层的参数的大小
        :param model:
        :param show_model:
        :param use_numpy:
        :param show_info:
        :param save_dir:
        :param paddle_model:
        :return:
        """
        if show_info and show_model:
            print(model)

        model_net = str(model)
        model_params = []

        end_type = "torch"
        if paddle_model:
            use_numpy = True
            end_type = "paddle"

        params_dict = model.named_parameters()
        for name, parameters in params_dict:
            parm_size = parameters.size() if not use_numpy else parameters.detach().cpu().numpy().shape
            msg = f"{name} : {parm_size}"
            if show_info:
                print(msg)
            model_params.append(msg)

        if save_dir is not None:
            file_name_torch_net = f"{save_dir}_{end_type}_net.txt"
            file_name_torch_param = f"{save_dir}_{end_type}_param.txt"
            FileUtils.save_to_text(file_name_torch_net, model_net)
            FileUtils.save_to_text(file_name_torch_param, "\n".join(model_params))

        return model_net, model_params

    @staticmethod
    def get_result_http_server(output_dir):
        result_http_server = "http://localhost:9100"

        prefix = f"{Constants.PDF_CACHE_BASE}/inference_results/"
        if output_dir is not None and str(output_dir).startswith(prefix):
            end = os.path.dirname(output_dir)[len(prefix):]
            result_http_server = f"{result_http_server}/{end}"

        return result_http_server

    @staticmethod
    def list_to_str(content: Union[str, List]):
        if isinstance(content, list):
            content = "\n".join(content)

        content = f"{content}\n"
        return content

    @staticmethod
    def get_color_list():
        color1 = (255, 255, 0)
        color2 = (255, 0, 255)
        color3 = (0, 0, 255)
        color4 = (0, 255, 255)
        color5 = (0, 255, 0)
        color6 = (255, 0, 0)

        color_list = [color1, color2, color3, color4, color5, color6]
        return color_list

    @staticmethod
    def get_html_color_list():
        color_list = [
            "Red", "Yellow", "Lime", "Aqua", "Blue",
            "Fuchsia", "Maroon", "Olive", "Green", "Teal",
            "Navy", "Purple"
        ]
        return color_list

    @staticmethod
    def render_html_template(template_name, demo_list):
        """
        採用 jinja2 生成模板

        :param template_name:
        :param demo_list:
        :return:
        """
        from jinja2 import FileSystemLoader, Environment
        load = FileSystemLoader('templates')
        env = Environment(loader=load)
        template = env.get_template(template_name)
        result = template.render(demo_list)
        return result

    @staticmethod
    def calc_pair_sentences_diff(input_text1, input_text2):
        """
        比对两个字符串 的差异

        :param input_text1:
        :param input_text2:
        :return:
        """
        text1_dict = defaultdict(int)
        text2_dict = defaultdict(int)

        for item in input_text1:
            text1_dict[item] += 1
        for item in input_text2:
            text2_dict[item] += 1

        diff_dict = {}
        for key, val in text1_dict.items():
            val2 = text2_dict.get(key, 0)
            diff_total = val - val2
            if diff_total != 0:
                diff_dict[key] = diff_total

        for key, val in text2_dict.items():
            val2 = text1_dict.get(key, 0)
            diff_total = val2 - val
            if diff_total != 0:
                diff_dict[key] = diff_total

        sorted_diff_dict = CommonUtils.sorted_dict(diff_dict)
        return sorted_diff_dict

    @staticmethod
    def calc_pair_structure_diff(input_cell1, input_cell2):
        """
        比对 table cell 的差异
            - <tdcolspan="2"></td>
            - <td></td>
        :param input_cell1:
        :param input_cell2:
        :return:
        """
        colspan1, rowspan1 = CommonUtils.extract_cell_span(input_cell1)
        colspan2, rowspan2 = CommonUtils.extract_cell_span(input_cell2)

        diff_col = colspan1 - colspan2
        diff_row = rowspan1 - rowspan1
        diff_dict = {
            "same_col": True if diff_col == 0 else False,
            "same_row": True if diff_row == 0 else False,
            "diff_col": diff_col,
            "diff_row": diff_row,
            "colspan1": colspan1,
            "rowspan1": rowspan1,
            "colspan2": colspan2,
            "rowspan2": rowspan2,
        }
        return diff_dict

    @staticmethod
    def extract_cell_span_by_name(cell, span_str='colspan="'):
        cell = cell.replace("'", '"')
        colspan = 1

        if span_str in cell:
            col_begin = cell.index(span_str)
            span_remain = cell[col_begin + len(span_str):]
            col_end = span_remain.index('"')
            find_colspan = span_remain[:col_end]
            colspan = int(find_colspan.replace('"', ""))

        return colspan

    @staticmethod
    def extract_cell_span(cell):
        colspan = CommonUtils.extract_cell_span_by_name(cell, span_str='colspan="')
        rowspan = CommonUtils.extract_cell_span_by_name(cell, span_str='rowspan="')
        return colspan, rowspan

    @staticmethod
    def results_to_thread_size(results: List, thread_num=1):
        """
        划分任务列表

        :param results:
        :param thread_num:
        :return:
        """
        total = len(results)
        need_run_ids = []
        batch_size = int(total / thread_num)
        logger.info(f"total: {total} - {thread_num} - {batch_size}")

        for index in range(0, thread_num):
            begin_index = index * batch_size
            end_index = (index + 1) * batch_size
            if index == thread_num - 1:
                end_index = total - 1
            run_total = end_index - begin_index
            need_run_ids.append([results[begin_index], results[end_index], run_total])

        return need_run_ids

    @staticmethod
    def parse_model_and_data_args(show_info=True) -> Tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
        """
        解析命令行参数 model_args, data_args
        :return:
        """
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        if show_info:
            logger.info(model_args)
            logger.info(data_args)
        # logger.info(training_args)

        return model_args, data_args, training_args

    @staticmethod
    def remove_space(sentence):
        """
        清理句子

        :param sentence:
        :return:
        """
        sentence_clean = re.sub(r"\s", "", sentence)
        return sentence_clean

    @staticmethod
    def is_ocr_number(text):
        flag = True
        for item in text:
            if not MatchUtils.match_pattern_flag(item, MatchUtils.PATTERN_OCR_TEXT_ZH_NUMBER):
                flag = False
                break

        dot_count = str(text).count(".")
        if flag and dot_count > 1:
            return True

        return False

    @staticmethod
    def clean_table_span(span):
        span = span.replace("<", "《").replace(">", "》")
        return span

    @staticmethod
    def calc_diff_structure_cell(pred_structure_cells, label_structure_cells):
        """
        计算预测和标注不同的表格结构

        :param pred_structure_cells:
        :param label_structure_cells:
        :return:
        """
        pred_row_total = len(pred_structure_cells)
        label_row_total = len(label_structure_cells)

        pred_cell_total = sum([len(item) for item in pred_structure_cells])
        label_cell_total = sum([len(item) for item in label_structure_cells])

        if pred_row_total != label_row_total:
            compare_type = HtmlTableCompareType.DIFF_CELL_DIFF_ROW
        elif pred_cell_total != label_cell_total:
            compare_type = HtmlTableCompareType.DIFF_CELL_ROW_COL_SPAN
        else:
            compare_type = HtmlTableCompareType.DIFF_CELL_SPAN_SAME

        diff_item = {
            "compare_type": compare_type.desc,
            "pred_row_total": pred_row_total,
            "label_row_total": label_row_total,
            "diff_row_total": pred_row_total - label_row_total,
            "pred_cell_total": pred_cell_total,
            "label_cell_total": label_cell_total,
            "diff_cell_total": pred_cell_total - label_cell_total,
        }

        return diff_item

    @staticmethod
    def get_parameter_number(model):
        """
        获取模型的参数量
        :param model:
        :return:
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data = {"Total": total_num, "Trainable": trainable_num}
        return data

    @staticmethod
    def calc_model_parameter_number(model, model_name: str = ""):
        """
        计算模型参数

        :param model:
        :param model_name:
        :return:
        """
        parameter_number = CommonUtils.get_parameter_number(model)
        total_parameter = parameter_number["Total"]
        trainable_parameter = parameter_number["Trainable"]

        logger.info(f"----------------------------------模型[{model_name}]参数汇总:--------------------------------")
        logger.info(f"Total params: {format(total_parameter, ',')}")
        logger.info(f"Total Trainable params: {format(trainable_parameter, ',')}")
        logger.info(f"Total Trainable params percent: {trainable_parameter / total_parameter:.4f}")

        return total_parameter, trainable_parameter

    @staticmethod
    def clean_table(table: str):
        replace_mapping = {
            "\r": "",
            "\n": "",
            "<thead>": "",
            "</thead>": "",
            "<tbody>": "",
            "</tbody": "",
            "<th width": "<td width",
            "<th>": "<td>",
            "</th>": "</td>"
        }

        new_table = table
        for k, v in replace_mapping.items():
            new_table = new_table.replace(k, v)
        return new_table

    @staticmethod
    def box_list_to_json_str(bbox_list: Union[List, np.ndarray]) -> str:
        """
        box to str
        :param bbox_list:
        :return:
        """
        if isinstance(bbox_list, list):
            bbox_list_str = json.dumps([item.tolist() if isinstance(item, np.ndarray) else item for item in bbox_list])
        else:
            bbox_list_str = json.dumps(bbox_list.tolist())
        return bbox_list_str

    @staticmethod
    def download_model_from_hub(model_name_or_path: str,
                                revision=None,
                                cache_dir=None,
                                force_use_ms=False):
        use_ms = Constants.PDFTABLE_USE_MODELSCOPE_HUB
        model_dir = model_name_or_path
        if force_use_ms or use_ms:
            try:
                from modelscope import snapshot_download
                if cache_dir is None:
                    cache_dir = Constants.SCOPE_MODEL_BASE_DIR
                model_dir = snapshot_download(model_name_or_path,
                                              revision=revision,
                                              cache_dir=cache_dir)
            except ImportError:
                raise ImportError("Please install modelscope via `pip install modelscope -U`")
        else:
            try:
                from huggingface_hub import snapshot_download
                if cache_dir is None:
                    cache_dir = Constants.HF_HUB_BASE_DIR
                model_dir = snapshot_download(repo_id=model_name_or_path,
                                              repo_type="model",
                                              revision=revision,
                                              cache_dir=cache_dir)

            except ImportError:
                raise ImportError("Please install huggingface_hub via `pip install huggingface_hub -U`")

            logger.info(f'Loading the model using model_dir: {model_dir}')

        model_dir = os.path.expanduser(model_dir)
        return model_dir
