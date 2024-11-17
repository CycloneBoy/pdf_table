#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : cls_image_pulc_task.py
# @Author   : cycloneboy
# @Date     : 20xx/10/15 - 13:39
import time
from copy import deepcopy

from pdftable.model.table.lgpma.checkpoint import load_checkpoint
from .base_infer_task import BaseInferTask
from ..cls import ClsPulcConfig, PPLCNet, PPLCNetImageProcessor, PPLCNetImagePostProcessor

"""
图像分类
"""

__all__ = [
    'ClsImagePulcTask'
]


class ClsImagePulcTask(BaseInferTask):

    def __init__(self, task="cls_image", model="PPLCNet", task_type="text_image_orientation", **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self.task_type = task_type
        self._config = ClsPulcConfig(recognizer=model, task_type=task_type)
        self.model_provider = "PaddleOCR"
        self._predictor_type = "pytorch"

        model_name_or_path = self.get_model_name_or_path()
        self._config.model_path = model_name_or_path
        self._get_inference_model()

    def _construct_model(self, model):
        config_params = self._config.get_model_params()
        self._model = PPLCNet(**config_params)

        map_loc = 'cpu' if self.device == 'cpu' else None
        checkpoint = load_checkpoint(self._model, self._config.model_path, map_location=map_loc)

    def _build_processor(self):
        self._pre_processor = PPLCNetImageProcessor(task=self.task_type)
        self._post_processor = PPLCNetImagePostProcessor(task=self.task_type)

    def _preprocess(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = []
        for item in inputs:
            one_item = self._pre_processor(item)
            item = {"image": one_item["pixel_values"]}
            batch.append(item)

        results = {"inputs": batch}
        return results

    def _run_model(self, inputs, **kwargs):
        infer_data_loader = inputs["inputs"]

        results = []
        begin = time.time()
        for index, batch in enumerate(infer_data_loader):
            pred_res, elapse = self.infer(batch)
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

        predict_result = inputs["results"]
        for index, batch in enumerate(predict_result):
            predict = self._post_processor(batch)

            raw_result = predict[0]
            results.append(raw_result)

        if len(results) == 1:
            results = results[0]
        # logger.info(f"推理结束【{self.task_type}】：{len(predict_result)} - 耗时：{use_time:.3f} s / {use_time / 60:.3f} min.")
        return results
