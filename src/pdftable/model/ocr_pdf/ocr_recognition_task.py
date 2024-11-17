#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_recognition_task
# @Author  ：cycloneboy
# @Date    ：20xx/7/17 14:30


import time
from copy import deepcopy

import torch

from .base_infer_task import BaseInferTask
from ..ocr_rec_pp import PPOcrRecognitionConfig, PPOcrRecPreProcessor, PPOcrRecPostProcessor
from ..ocr_recognition import OCRRecognitionConfig, OCRRecognitionPreprocessor, OCRRecognitionPostProcessor, \
    OCRRecognition

"""
OcrRecognitionTask
"""

__all__ = [
    'OcrRecognitionTask'
]


class OcrRecognitionTask(BaseInferTask):

    def __init__(self, task="ocr_recognition", model="ConvNextViT", task_type="document", **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        if model in ["ConvNextViT", "CRNN", "LightweightEdge"]:
            self._config = OCRRecognitionConfig(recognizer=model, task_type=task_type)
            self.model_provider = "model_scope"
        elif model in ["PP-OCRv4", "PP-OCRv3", "PP-Table"]:
            rec_image_shape = "3, 48, 320"
            if model == "PP-Table":
                rec_image_shape = "3, 32, 320"
            self._config = PPOcrRecognitionConfig(backbone=model,
                                                  rec_image_shape=rec_image_shape,
                                                  lang=self.lang)

            self.model_provider = "PaddleOCR"
            self._predictor_type = "onnx"
        else:
            raise RuntimeError(f"current model is not supported: {model}")

        model_name_or_path = self.get_model_name_or_path()
        self._config.model_path = model_name_or_path
        self._get_inference_model()

    def _construct_model(self, model):
        if self.model_provider == "model_scope":
            self._model = OCRRecognition(self._config)

    def _build_processor(self):
        if self.model_provider == "model_scope":
            self._pre_processor = OCRRecognitionPreprocessor(self._config)
            self._post_processor = OCRRecognitionPostProcessor(self._config)
        elif self.model_provider == "PaddleOCR":
            self._pre_processor = PPOcrRecPreProcessor(self._config)
            self._post_processor = PPOcrRecPostProcessor(self._config)

    def _preprocess(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if self.model_provider == "model_scope":
            batch = []
            for item in inputs:
                one_item = self._pre_processor(item)
                batch.append(one_item)
        elif self.model_provider == "PaddleOCR":
            batch = self._pre_processor(inputs)

        results = {"inputs": batch}

        return results

    def _run_model(self, inputs, **kwargs):
        infer_data_loader = inputs["inputs"]

        results = []
        begin = time.time()
        for index, batch in enumerate(infer_data_loader):
            image = batch["image"]

            run_batch = {}
            if self._predictor_type == "onnx":
                input_name = self.get_input_name()
            else:
                input_name = "inputs"
            run_batch[input_name] = image

            pred_res, elapse = self.infer(run_batch)

            if self._predictor_type == "onnx":
                pred_res = torch.tensor(pred_res[0]).to(self.device)

            # logger.info(f'pred_res: {pred_res.shape}')

            infer_result = {
                "results": pred_res,
                "elapse": elapse,
            }
            raw_batch = deepcopy(batch)
            raw_batch.pop("image")
            infer_result.update(raw_batch)

            results.append(infer_result)

        use_time = time.time() - begin
        inputs["results"] = results
        inputs["use_time"] = use_time
        return inputs

    def _postprocess(self, inputs, **kwargs):
        results = []

        use_time = inputs["use_time"]

        if self.model_provider == "model_scope":
            predict_result = inputs["results"]
            for index, batch in enumerate(predict_result):
                predict = self._post_processor(batch["results"])

                results.append(predict['preds'][0])
        elif self.model_provider == "PaddleOCR":
            predict = self._post_processor(inputs)
            for index, pred in enumerate(predict):
                results.append(pred['preds'])

        # if self.debug:
        #     logger.info(f"推理结束：{len(predict_result)} - 耗时：{use_time:.3f} s / {use_time / 60:.3f} min.")
        return results
