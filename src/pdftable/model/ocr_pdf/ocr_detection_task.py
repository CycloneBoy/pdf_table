#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : ocr_detection_task.py
# @Author   : cycloneboy
# @Date     : 20xx/7/16 - 18:27
import os
import time
from copy import deepcopy

import cv2
import torch
from pdftable.utils.ocr import OcrInferUtils

from .base_infer_task import BaseInferTask
from ..db_net import DbNetConfig, OCRDetectionPreprocessor, OCRDetectionPostProcessor, OCRDetectionDbNet
from ..db_pp import DbPPConfig, PPOcrDetectionPreprocessor, PPOcrDetectionPostProcessor

from ...utils import logger, FileUtils

"""
OcrDetectionTask
"""

__all__ = [
    'OcrDetectionTask'
]


class OcrDetectionTask(BaseInferTask):

    def __init__(self, task="ocr_detection", model="db", backbone: str = "resnet18", thresh: float = 0.2,
                 **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        if model == "db":
            self._config = DbNetConfig(backbone=backbone, thresh=thresh)
            self.model_provider = "model_scope"
            self._predictor_type = "pytorch"
        elif model == "db_pp":
            det_limit_side_len = 960
            det_limit_type = "max"
            if model == "PP-Table":
                det_limit_side_len = 736
                det_limit_type = "min"

            lang = self.lang
            if lang not in ["ch","en"]:
                lang = "ml"
            self._config = DbPPConfig(backbone=backbone,
                                      det_limit_side_len=det_limit_side_len,
                                      det_limit_type=det_limit_type,
                                      thresh=thresh,
                                      lang=lang)
            self.model_provider = "PaddleOCR"
            self._predictor_type = "onnx"
        else:
            raise RuntimeError(f"current model is not supported: {model}")

        model_name_or_path = self.get_model_name_or_path()
        self._config.model_path = model_name_or_path

        self._get_inference_model()

    def _construct_model(self, model):
        if model == "db":
            self._model = OCRDetectionDbNet(self._config)

    def _build_processor(self):
        if self.model == "db":
            self._pre_processor = OCRDetectionPreprocessor(self._config)
            self._post_processor = OCRDetectionPostProcessor(self._config)
        elif self.model == "db_pp":
            self._pre_processor = PPOcrDetectionPreprocessor(self._config)
            self._post_processor = PPOcrDetectionPostProcessor(self._config)

    def _preprocess(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = []
        for item in inputs:
            one_item = self._pre_processor(item)
            batch.append(one_item)

        results = {"inputs": batch}
        return results

    def _run_model(self, inputs, **kwargs):
        infer_data_loader = inputs["inputs"]

        results = []
        begin = time.time()
        for index, batch in enumerate(infer_data_loader):
            image = batch["image"].unsqueeze(0)

            run_batch = {}
            if self._predictor_type == "onnx":
                input_name = self.get_input_name()
            else:
                input_name = "input"
            run_batch[input_name] = image

            pred_res, elapse = self.infer(run_batch)

            if self._predictor_type == "onnx":
                pred_res = torch.tensor(pred_res[0]).to(self.device)

            logger.info(f'pred_res: {pred_res.shape}')

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

        predict_result = inputs["results"]
        for index, batch in enumerate(predict_result):
            predict = self._post_processor(batch)

            det_results = predict['det_polygons']
            if self.output_dir is not None and self.debug and "inputs" in predict:
                self.show_results(predict)
            results.append(det_results)

        logger.info(f"推理结束：{len(predict_result)} - 耗时：{use_time:.3f} s / {use_time / 60:.3f} min.")
        return results

    def show_results(self, predict):
        det_results = predict['det_polygons']
        image_file = predict['inputs']
        src_im = OcrInferUtils.draw_text_det_res(det_results, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(self.output_dir, "det_res_{}".format(img_name_pure))
        FileUtils.check_file_exists(img_path)
        cv2.imwrite(img_path, src_im)
        logger.info(f"The visualized image saved in {img_path}")
