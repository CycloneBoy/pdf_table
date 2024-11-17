#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：deploy_utils.py
# @Author  ：cycloneboy
# @Date    ：20xx/6/24 10:49
import os
from itertools import chain
from pathlib import Path

import numpy as np
import re

import onnx
import torch.nn as nn
from transformers.onnx import OnnxConfig

from .logger_utils import logger

from .base_utils import BaseUtil
from .file_utils import FileUtils

"""
部署相关工具类
"""


class DeployUtils(BaseUtil):
    """
    部署相关工具类
    """

    def init(self):
        pass

    @staticmethod
    def onnx_to_fp16(float_onnx_file, fp16_onnx_file):
        """
            onnx model to fp16

        :param float_onnx_file:
        :param fp16_onnx_file:
        :return:
        """
        from onnxconverter_common import float16
        if not FileUtils.check_file_exists(fp16_onnx_file):
            onnx_model = onnx.load_model(float_onnx_file)
            trans_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
            onnx.save_model(trans_model, fp16_onnx_file)
            logger.info(f"转换ONNX 模型 到 FP16")
        else:
            logger.info(f"ONNX FP16 模型已经存在：{fp16_onnx_file} ")

    @staticmethod
    def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
        """
        Get idx of the last dimension in probability arrays, which is greater than a limitation.

        Args:
            probs (List[List[float]]): The input probability arrays.
            limit (float): The limitation for probability.
            return_prob (bool): Whether to return the probability
        Returns:
            List[List[int]]: The index of the last dimension meet the conditions.
        """
        probs = np.array(probs)
        dim_len = len(probs.shape)
        if dim_len > 1:
            result = []
            for p in probs:
                result.append(DeployUtils.get_bool_ids_greater_than(p, limit, return_prob))
            return result
        else:
            result = []
            for i, p in enumerate(probs):
                if p > limit:
                    if return_prob:
                        result.append((i, p))
                    else:
                        result.append(i)
            return result

    @staticmethod
    def get_span(start_ids, end_ids, with_prob=False):
        """
        Get span set from position start and end list.

        Args:
            start_ids (List[int]/List[tuple]): The start index list.
            end_ids (List[int]/List[tuple]): The end index list.
            with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
        Returns:
            set: The span set without overlapping, every id can only be used once .
        """
        if with_prob:
            start_ids = sorted(start_ids, key=lambda x: x[0])
            end_ids = sorted(end_ids, key=lambda x: x[0])
        else:
            start_ids = sorted(start_ids)
            end_ids = sorted(end_ids)

        start_pointer = 0
        end_pointer = 0
        len_start = len(start_ids)
        len_end = len(end_ids)
        couple_dict = {}
        while start_pointer < len_start and end_pointer < len_end:
            if with_prob:
                start_id = start_ids[start_pointer][0]
                end_id = end_ids[end_pointer][0]
            else:
                start_id = start_ids[start_pointer]
                end_id = end_ids[end_pointer]

            if start_id == end_id:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_id < end_id:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_id > end_id:
                end_pointer += 1
                continue
        result = [(couple_dict[end], end) for end in couple_dict]
        result = set(result)
        return result

    @staticmethod
    def get_id_and_prob(span_set, offset_mapping):
        """
        Return text id and probability of predicted spans

        Args:
            span_set (set): set of predicted spans.
            offset_mapping (list[int]): list of pair preserving the
                    index of start and end char in original text pair (prompt + text) for each token.
        Returns:
            sentence_id (list[tuple]): index of start and end char in original text.
            prob (list[float]): probabilities of predicted spans.
        """
        prompt_end_token_id = offset_mapping[1:].index([0, 0])
        bias = offset_mapping[prompt_end_token_id][1] + 1
        for index in range(1, prompt_end_token_id + 1):
            offset_mapping[index][0] -= bias
            offset_mapping[index][1] -= bias

        sentence_id = []
        prob = []
        for start, end in span_set:
            prob.append(start[1] * end[1])
            start_id = offset_mapping[start[0]][0]
            end_id = offset_mapping[end[0]][1]
            sentence_id.append((start_id, end_id))
        return sentence_id, prob

    @staticmethod
    def dbc2sbc(s):
        rs = ""
        for char in s:
            code = ord(char)
            if code == 0x3000:
                code = 0x0020
            else:
                code -= 0xfee0
            if not (0x0021 <= code and code <= 0x7e):
                rs += char
                continue
            rs += chr(code)
        return rs

    @staticmethod
    def map_offset(ori_offset, offset_mapping):
        """
        map ori offset to token offset
        """
        for index, span in enumerate(offset_mapping):
            if span[0] <= ori_offset < span[1]:
                return index
        return -1

    @staticmethod
    def unify_prompt_name(prompt):
        # The classification labels are shuffled during finetuneing, so they need
        # to be unified during evaluation.
        if re.search(r'\[.*?\]$', prompt):
            prompt_prefix = prompt[:prompt.find("[", 1)]
            cls_options = re.search(r'\[.*?\]$', prompt).group()[1:-1].split(",")
            cls_options = sorted(list(set(cls_options)))
            cls_options = ",".join(cls_options)
            prompt = prompt_prefix + "[" + cls_options + "]"
            return prompt
        return prompt

    @staticmethod
    def export_model(model, model_inputs, onnx_path, onnx_config: OnnxConfig):
        """
        Exports a model into ONNX format.

        :param model:
        :param model_inputs:
        :param onnx_path:
        :param onnx_config:
        :return:
        """
        from torch.onnx import export
        onnx_outputs = list(onnx_config.outputs.keys())
        dynamic_axes = {name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())}
        if isinstance(onnx_path, Path):
            onnx_path = onnx_path.as_posix()
        FileUtils.check_file_exists(onnx_path)
        export(
            model,
            (model_inputs,),
            f=onnx_path,
            input_names=list(onnx_config.inputs.keys()),
            output_names=onnx_outputs,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=True,
            opset_version=onnx_config.default_onnx_opset,
        )

    @staticmethod
    def model_eval(model: nn.Module, device="cuda", fp16_full_eval=True):
        """
        mode eval
        :param model:
        :param device:
        :param fp16_full_eval:
        :return:
        """
        model.to(device)
        if fp16_full_eval:
            model = model.half()
            logger.info(f"使用FP16量化模型推理: device: {device} - {model.__class__.__name__}")
        model.eval()
        return model

    @staticmethod
    def prepare_onnx_model(onnx_dir, device_id=0, num_threads=4):
        try:
            import onnx
            import onnxruntime as ort
            from onnxconverter_common import float16
        except ImportError:
            logger.warning(
                "The inference precision is change to 'fp32', please install the dependencies that required for 'fp16' inference, pip install onnxruntime-gpu onnx onnxconverter-common"
            )

        if onnx_dir.endswith(".onnx"):
            fp16_model_file = onnx_dir
        else:
            os.makedirs(onnx_dir, exist_ok=True)
            float_onnx_file = os.path.join(onnx_dir, "model.onnx")
            fp16_model_file = os.path.join(onnx_dir, "fp16_model.onnx")
            if not os.path.exists(fp16_model_file):
                onnx_model = onnx.load_model(float_onnx_file)
                trans_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
                onnx.save_model(trans_model, fp16_model_file)

                logger.info(f"转换ONNX 模型 到 FP16: {fp16_model_file}")

        providers = [("CUDAExecutionProvider", {"device_id": device_id})]
        sess_options = ort.SessionOptions()
        # sess_options.enable_profiling = True
        # sess_options.intra_op_num_threads = num_threads
        # sess_options.inter_op_num_threads = num_threads
        predictor = ort.InferenceSession(fp16_model_file, sess_options=sess_options, providers=providers)
        assert "CUDAExecutionProvider" in predictor.get_providers(), (
            "The environment for GPU inference is not set properly. "
            "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. "
            "Please run the following commands to reinstall: \n "
            "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"
        )
        logger.info(f"采用ONNX FP16 推理【device_id={device_id}】：{fp16_model_file}")

        return predictor
