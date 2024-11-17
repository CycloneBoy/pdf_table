#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : base_infer_task.py
# @Author   : cycloneboy
# @Date     : 20xx/7/16 - 18:04
import math
import os
import time
from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count
from typing import Dict

import numpy as np
import torch
from transformers.onnx import OnnxConfig

from pdftable.model.ocr_pdf.ocr_table_model_config import TABLE_MODEL_DICT
from pdftable.utils import (
    Constants,
    TimeUtils, CommonUtils,
)
from pdftable.utils.deploy_utils import DeployUtils

__all__ = [
    "BaseInferTask"
]


class BaseInferTask(metaclass=ABCMeta):

    def __init__(self, model, task, priority_path=None, **kwargs):

        self.model = model
        self.is_static_model = kwargs.get("is_static_model", False)
        self.model_provider = kwargs.get("model_provider", "model_scope")
        self.task = task
        self.kwargs = kwargs
        self._priority_path = priority_path
        self._usage = ""
        # The dygraph model instance
        self._model = None
        self._tokenizer = None
        self._onnx_config: OnnxConfig = None

        self._pre_processor = None
        self._post_processor = None
        # The static model instance
        self._input_spec = None
        self._config = None
        self._init_class = None
        self._custom_model = False
        self._param_updated = False

        self._num_threads = self.kwargs["num_threads"] if "num_threads" in self.kwargs else math.ceil(cpu_count() / 2)
        self._infer_precision = self.kwargs["precision"] if "precision" in self.kwargs else "fp16"
        self.eval_fp16 = True if self._infer_precision == "fp16" else False
        # Default to use pytorch Inference
        self._predictor_type = kwargs.get("predictor_type", "pytorch")
        # The root directory for storing Taskflow related files, default to ~/.paddlenlp.
        self._home_path = self.kwargs["home_path"] if "home_path" in self.kwargs else Constants.OUTPUT_DIR
        self._task_flag = self.kwargs["task_flag"] if "task_flag" in self.kwargs else self.model
        self.from_hf_hub = kwargs.pop("from_hf_hub", False)

        self._lazy_load = kwargs.get("lazy_load", False)
        self._num_workers = kwargs.get("num_workers", 0)
        self._use_fast = kwargs.get("use_fast", True)
        self.do_transform = kwargs.get("do_transform", False)
        self.device = kwargs.get("device", "cuda:0")
        self.output_dir = kwargs.get("output_dir", None)
        self.debug = kwargs.get("debug", True)

        self.run_time = kwargs.get("run_time", TimeUtils.now_str_short())
        self.lang = kwargs.get("lang", "en")
        self.task_type = kwargs.get("task_type", "wtw")
        self.server_model = kwargs.get("server_model", False)
        self.use_modelscope_hub = kwargs.get("use_modelscope_hub", Constants.PDFTABLE_USE_MODELSCOPE_HUB)

        self._task_path_default = os.path.join(self._home_path, "taskflow", self.task, self.model)

        if "task_path" in self.kwargs:
            self._task_path = self.kwargs["task_path"]
            self._custom_model = True
        elif self._priority_path:
            self._task_path = os.path.join(self._home_path, "taskflow", self._priority_path)
        else:
            self._task_path = os.path.join(self._home_path, "taskflow", self.task, self.model)

        self.model_dict = TABLE_MODEL_DICT

        # if not self.from_hf_hub:
        #     pass
        #     # TaskFlowCommonUtils.download_check(self._task_flag)

    @abstractmethod
    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

    @abstractmethod
    def _build_processor(self, ):
        """
        Construct the processor.
        """

    @abstractmethod
    def _preprocess(self, inputs, **kwargs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """

    @abstractmethod
    def _run_model(self, inputs, **kwargs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """

    @abstractmethod
    def _postprocess(self, inputs, **kwargs):
        """
        The model output is the logits and pros, this function will convert the model output to raw text.
        """

    def _prepare_pytorch_mode(self):
        """
        Construct the input data and predictor in the pytorch mode.
        """
        self._model = DeployUtils.model_eval(model=self._model, device=self.device, fp16_full_eval=self.eval_fp16)

        self.predictor = self._model
        # logger.info(f"采用pytorch推理,use_fp16: {self.eval_fp16}")

    def _prepare_onnx_mode(self):
        model_name_or_path = self.get_model_name_or_path()

        predictor = DeployUtils.prepare_onnx_model(onnx_dir=model_name_or_path)

        self.predictor = predictor

    def _prepare_trt_mode(self):
        raise RuntimeError("TensorRt infer not supported!!!")

    def _get_inference_model(self):
        if self.is_static_model:
            self.inference_model_path = self._task_path
        else:
            _base_path = (
                self._task_path
                if not self.from_hf_hub
                else os.path.join(self._home_path, "taskflow", self.task, self.model)
            )
            self.inference_model_path = os.path.join(_base_path, "pytorch_model.bin")

        if self._predictor_type in ["pytorch", "other"]:
            self._construct_model(self.model)

        self._build_processor()

        if self._predictor_type == "pytorch":
            self._prepare_pytorch_mode()
        elif self._predictor_type == "onnx":
            self._prepare_onnx_mode()
        elif self._predictor_type == "trt":
            self._prepare_trt_mode()
        else:
            self.predictor = self._model

    def get_model_name_or_path(self):
        model_name_or_path = self._task_path
        pretrain_weight_pytorch = f"{model_name_or_path}/pytorch_model.bin"

        if os.path.exists(pretrain_weight_pytorch):
            return model_name_or_path

        if self._priority_path is not None:
            if str(self._priority_path).find("/") > -1:
                model_name_or_path = self._priority_path

            catch_path = os.path.join(Constants.SCOPE_MODEL_BASE_DIR, self._priority_path)
            if os.path.exists(catch_path):
                model_name_or_path = catch_path

            # catch_path = os.path.join(MODEL_HOME, self._priority_path)
            # if os.path.exists(catch_path):
            #     model_name_or_path = catch_path

        if self.task in [
            "ocr_detection",
            "ocr_recognition",
            "ocr_table_structure",
            "cls_image",
            "ocr_layout"
        ]:
            if self.model_provider == "model_scope":
                model_name_or_path = self.get_model_path_from_model_scope()
            elif self.model_provider == "PaddleOCR":
                model_name_or_path = self.get_model_path_from_paddleocr()
            elif self.model_provider == "Other":
                model_name_or_path = self.get_model_path_from_other()

        # model_name_or_path = os.path.join(Constants.SCOPE_MODEL_BASE_DIR, model_config)
        if model_name_or_path in [
            'cycloneboy/line_cell',
            'cycloneboy/line_cell_pdf'
        ]:
            model_dir = model_name_or_path
        else:
            model_dir = CommonUtils.download_model_from_hub(model_name_or_path)

        return model_dir

    def get_model_id(self, config: Dict):
        if config is None:
            return None
        if self.server_model and not self.use_modelscope_hub and "hf_server_model" in config:
            model_id = config["hf_server_model"]
        elif self.server_model and "server_model" in config:
            model_id = config["server_model"]
        elif not self.use_modelscope_hub and "hf_model" in config:
            model_id = config["hf_model"]
        else:
            model_id = config["model"]
        return model_id

    def get_model_path_from_model_scope(self, ):
        """
        model_scope model path

        :return:
        """
        model_dict = self.model_dict[self.model_provider]
        if self.task == "ocr_detection":
            model_config = model_dict["detection"][self._config.backbone]["general"]
        elif self.task == "ocr_recognition":
            model_config = model_dict["recognition"][self._config.recognizer][self._config.task_type]
        elif self.task == "ocr_table_structure":
            model_config = model_dict["table_structure"][self._config.model_name][self._config.task_type]
        elif self.task == "ocr_layout":
            model_config = model_dict["layout"][self._config.model_name][self._config.task_type]
        else:
            model_config = None

        model_id = self.get_model_id(model_config)
        return model_id

    def get_model_path_from_paddleocr(self, ):
        """
        paddleocr model path

        :return:
        """
        model_dict = self.model_dict[self.model_provider]
        backbone = self._config.backbone
        lang = self.lang
        if self.task == "ocr_detection":
            config_name = "detection"
        elif self.task == "ocr_recognition":
            config_name = "recognition"
            if backbone in ["PP-OCRv4"] and lang not in ["ch", "en"]:
                backbone = "PP-OCRv3"
        elif self.task == "ocr_table_structure":
            config_name = "table_structure"
        elif self.task == "cls_image":
            config_name = "cls_image"
        elif self.task == "ocr_layout":
            config_name = "layout"
        raw_config = model_dict[config_name][backbone]

        if config_name == "detection" and lang not in ["ch", "en", "ml"]:
            lang = "ml"
        elif config_name == "recognition" and lang not in ["ch", "en", "chinese_cht",
                                                           "korean", "japan"]:
            lang = "en"
        elif config_name in ["cls_image", "layout"]:
            lang = self.task_type

        config = raw_config.get(lang, "en")
        model_id = self.get_model_id(config)
        return model_id

    def get_model_path_from_other(self, ):
        """
        other 中获取模型地址

        :return:
        """
        model_dict = self.model_dict[self.model_provider]
        if self.task == "ocr_detection":
            model_config = model_dict["detection"][self._config.backbone]["general"]
        elif self.task == "ocr_recognition":
            model_config = model_dict["recognition"][self._config.recognizer][self._config.task_type]
        elif self.task == "ocr_table_structure":
            model_config = model_dict["table_structure"][self._config.model_name][self._config.task_type]
        elif self.task == "ocr_layout":
            model_config = model_dict["layout"][self._config.model_name][self._config.task_type]
        else:
            model_config = None

        model_id = self.get_model_id(model_config)
        return model_id

    def help(self):
        """
        Return the usage message of the current task.
        """
        print("Examples:\n{}".format(self._usage))

    def __call__(self, *args, **kwargs):
        inputs = self._preprocess(*args, **kwargs)
        outputs = self._run_model(inputs, **kwargs)
        results = self._postprocess(outputs, **kwargs)
        return results

    def infer_pytorch(self, batch: dict, generate=False):
        self.build_torch_infer_batch(batch)
        with torch.no_grad():
            if not generate:
                result = self.predictor(**batch)
            else:
                result = self.predictor.generate(**batch)

        return result

    def build_torch_infer_batch(self, batch):
        """
        构造推理数据

        :param batch:
        :return:
        """
        for k, v in batch.items():
            if isinstance(v, dict):
                batch[k] = v
            elif isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)

                batch[k] = v.to(self.device)
                if self.eval_fp16:
                    batch[k] = batch[k].half()
            else:
                batch[k] = v

        return batch

    def build_onnx_infer_batch(self, batch):
        """
        构造推理数据

        :param batch:
        :return:
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()

            batch[k] = v
            if k in ["image", "pixel_values"] and self.eval_fp16:
                batch[k] = v.astype(np.half)

        return batch

    def infer(self, input_dict: dict, generate=False):
        start_time = time.time()
        if self._predictor_type == "onnx":
            self.build_onnx_infer_batch(input_dict)
            result = self.predictor.run(None, input_dict)
        elif self._predictor_type == "pytorch":
            result = self.infer_pytorch(batch=input_dict, generate=generate)
        elif self._predictor_type == "trt":
            self.build_torch_infer_batch(input_dict)
            result = self.predictor(input_dict)
        else:
            result = self.predictor(**input_dict)

        elapse = time.time() - start_time
        # logger.info(f"result:{result}")
        return result, elapse

    def get_input_name(self, input_idx=0):
        if self._predictor_type != "onnx":
            return None
        res = self.predictor.get_inputs()[input_idx].name
        return res

    def get_output_name(self, output_idx=0):
        if self._predictor_type != "onnx":
            return None
        return self.predictor.get_outputs()[output_idx].name

    def get_onnx_output_dict(self, outputs):
        if self._predictor_type != "onnx":
            return None
        result = {}
        total = len(self.predictor.get_outputs())
        for index in range(total):
            name = self.get_output_name(index)
            result[name] = outputs[index]

        return result

    def save_predict(self, image, predict, **kwargs):
        pass
