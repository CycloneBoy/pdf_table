#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_layout_task
# @Author  ：cycloneboy
# @Date    ：20xx/7/18 13:56
import os
import time

import cv2

from .base_infer_task import BaseInferTask
from ..docx_layout import DocXLayoutConfig, DocXLayoutModel, DocXLayoutPreProcessor, DocXLayoutImagePostProcessor
from ..picodet import OCRPicodetPreProcessor, OCRPicodetPostProcessor, PicodetConfig
from ...utils import logger, FileUtils
from ...utils.ocr import OcrInferUtils

"""
Ocr Layout Task
"""

__all__ = [
    'OcrLayoutTask'
]


class OcrLayoutTask(BaseInferTask):

    def __init__(self, task="ocr_layout", model="picodet",
                 task_type: str = "en",
                 score_threshold: float = 0.5, nms_threshold: float = 0.5,
                 **kwargs):
        super().__init__(task=task, model=model, task_type=task_type, **kwargs)

        if model == "picodet":
            if task_type not in ["ch","en","table"]:
                task_type = "en"
            self._config = PicodetConfig(task_type=task_type,
                                         score_threshold=score_threshold,
                                         nms_threshold=nms_threshold,
                                         )
            self._predictor_type = "onnx"
            self.model_provider = "PaddleOCR"
            self.eval_fp16 = False
        elif model == "DocXLayout":
            self._config = DocXLayoutConfig()
            self._predictor_type = "pytorch"
            self.model_provider = "model_scope"

        model_name_or_path = self.get_model_name_or_path()
        self._config.model_path = model_name_or_path

        self._get_inference_model()

    def _construct_model(self, model):
        if self.model == "picodet":
            logger.warning(f"pytorch model not support yet!")
        elif self.model == "DocXLayout":
            run_model = DocXLayoutModel(self._config)
            self._model = run_model

    def _build_processor(self):
        if self.model == "picodet":
            self._pre_processor = OCRPicodetPreProcessor(self._config)
            self._post_processor = OCRPicodetPostProcessor(self._config)
        elif self.model == "DocXLayout":
            self._pre_processor = DocXLayoutPreProcessor()
            self._post_processor = DocXLayoutImagePostProcessor(self._config)

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
            image = batch["image"]

            run_batch = {}
            if self._predictor_type == "onnx":
                input_name = self.get_input_name()
                image = image.unsqueeze(0)
            else:
                input_name = "pixel_values"
            run_batch[input_name] = image

            pred_res, elapse = self.infer(run_batch)

            if self._predictor_type == "onnx":
                pred_res = self.get_onnx_output_dict(pred_res)

            if self.model == "picodet":
                add_keys = ['image_file', 'org_shape', 'scale_factor', 'target_shape']
            else:
                add_keys = ['image_file', "meta"]
            for key in add_keys:
                if key in batch:
                    pred_res[key] = batch[key]

            logger.info(f'pred_res: {pred_res.keys()}')

            infer_result = {
                "results": pred_res,
                "elapse": elapse,
            }

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
            predict = self._post_processor(batch["results"])

            pred = predict["bboxs"]
            results.append(pred)

            if self.output_dir is not None:
                image_file = inputs["inputs"][index]['image_file']
                image_file = image_file if len(image_file) > 0 else kwargs.get("image_file", None)
                if image_file is not None and len(image_file) > 0:
                    raw_file_name = FileUtils.get_file_name(image_file)
                    if self.debug:
                        image = cv2.imread(image_file)
                        self.save_predict(image=image, predict=pred,
                                          img_name=f"layout_{raw_file_name}.jpg")

                    file_name = f"{self.output_dir}/predict_{raw_file_name}.json"
                    new_result = []
                    for item in pred:
                        bbox = item["bbox"].tolist()
                        item["bbox"] = bbox
                        new_result.append(item)
                    FileUtils.save_to_json(file_name, new_result)

        logger.info(f"推理结束：{len(predict_result)} - 耗时：{use_time:.3f} s / {use_time / 60:.3f} min.")

        return results

    def get_onnx_output_dict(self, outputs):
        if self._predictor_type != "onnx":
            return None
        total = len(self.predictor.get_outputs())

        np_score_list, np_boxes_list = [], []
        # output_names = self.predictor.get_outputs()
        # input_names = self.predictor.get_inputs()
        # logger.info(f"input_names:{input_names}")
        # logger.info(f"output_names:{output_names}")
        num_outs = int(len(outputs) / 2)
        for out_idx in range(num_outs):
            np_score_list.append(outputs[out_idx])
            np_boxes_list.append((outputs[out_idx + num_outs]))
        result = dict(boxes=np_score_list, boxes_num=np_boxes_list)

        return result

    def save_predict(self, image, predict, **kwargs):
        img_name = kwargs.get('img_name', "demo.png")
        img_path = os.path.join(self.output_dir, img_name)
        OcrInferUtils.draw_text_layout_res(image, layout_res=predict, save_path=img_path)
