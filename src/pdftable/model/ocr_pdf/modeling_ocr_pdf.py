#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：modeling_ocr_pdf
# @Author  ：cycloneboy
# @Date    ：20xx/7/14 15:08
import os
import time
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from .configuration_ocr_document import OCRDocumentConfig
from ..db_net.configuration_dbnet import DbNetConfig
from ..db_net.modeling_db_net import OCRDetectionDbNet
from ..db_net.processor_ocr_dbnet import OCRDetectionPreprocessor, OCRDetectionPostProcessor
from ..ocr_recognition import OCRRecognitionConfig, OCRRecognitionPreprocessor, OCRRecognition, \
    OCRRecognitionPostProcessor
from ...utils import logger, FileUtils, TimeUtils, Constants, DeployUtils
from ...utils.ocr import OcrCommonUtils

"""
OCR document

"""

__all__ = [
    'OcrDocument'
]


class OcrDocument(object):

    def __init__(self, config: OCRDocumentConfig, debug=False, output_dir=None, **kwargs):
        self.config = config
        self.debug = debug
        self.output_dir = output_dir

        self.device = "cuda"
        self.fp16_full_eval = True
        self._predictor_type = kwargs.get("predictor_type", "onnx")

        self.model_dict = {
            "detection": {
                "resnet18": {
                    "general": {
                        "backbone": "resnet18",
                        "model": 'damo/cv_resnet18_ocr-detection-db-line-level_damo',
                        "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg'
                    },
                },
                "proxylessnas": {
                    "general": {
                        "backbone": "proxylessnas",
                        "model": "damo/cv_proxylessnas_ocr-detection-db-line-level_damo",
                        "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg'
                    },
                },
            },
            "recognition": {
                "CRNN": {
                    "general": {
                        "recognizer": "CRNN",
                        "model": 'damo/cv_crnn_ocr-recognition-general_damo',
                        "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
                    },
                },
                "LightweightEdge": {
                    "general": {
                        "recognizer": "LightweightEdge",
                        "model": 'damo/cv_LightweightEdge_ocr-recognitoin-general_damo',
                        "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
                    },
                },
                "ConvNextViT": {
                    "general": {
                        "recognizer": "ConvNextViT",
                        "model": 'damo/cv_convnextTiny_ocr-recognition-general_damo',
                        "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
                    },
                    "handwritten": {
                        "recognizer": "ConvNextViT",
                        "model": 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo',
                        "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_handwritten.jpg'
                    },
                    "document": {
                        "recognizer": "ConvNextViT",
                        "model": 'damo/cv_convnextTiny_ocr-recognition-document_damo',
                        "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_document.png'
                    },
                    "licenseplate": {
                        "recognizer": "ConvNextViT",
                        "model": 'damo/cv_convnextTiny_ocr-recognition-licenseplate_damo',
                        "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_licenseplate//ocr_recognition_licenseplate.jpg'
                    },
                    "scene": {
                        "recognizer": "ConvNextViT",
                        "model": 'damo/cv_convnextTiny_ocr-recognition-scene_damo',
                        "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
                    },
                },
            }
        }

        self.ocr_detection_model: OCRDetectionDbNet = None
        self.ocr_detection_preprocessor: OCRDetectionPreprocessor = None
        self.ocr_detection_post_processor: OCRDetectionPostProcessor = None

        self.ocr_recognition_model: OCRRecognition = None
        self.ocr_recognition_preprocessor: OCRRecognitionPreprocessor = None
        self.ocr_recognition_post_processor: OCRRecognitionPostProcessor = None

        self.build_model()

    def build_detection_processor(self, model_name_or_path):
        config = DbNetConfig(backbone=self.config.detector,
                             model_path=model_name_or_path)

        self.ocr_detection_preprocessor = OCRDetectionPreprocessor(config)
        self.ocr_detection_post_processor = OCRDetectionPostProcessor(config)

        return config

    def build_detection_model(self, model_name_or_path):
        """
        加载detection模型

        :return:
        """
        config = self.build_detection_processor(model_name_or_path=model_name_or_path)

        model = OCRDetectionDbNet(config)
        model = DeployUtils.model_eval(model=model, device=self.device, fp16_full_eval=self.fp16_full_eval)

        self.ocr_detection_model = model

    def build_recognition_processor(self, model_name_or_path):
        config = OCRRecognitionConfig.from_pretrained(model_name_or_path)
        config.model_path = model_name_or_path

        self.ocr_recognition_preprocessor = OCRRecognitionPreprocessor(config)
        self.ocr_recognition_post_processor = OCRRecognitionPostProcessor(config)

        return config

    def build_recognition_model(self, model_name_or_path):
        """
        加载recognition模型

        :return:
        """
        config = self.build_recognition_processor(model_name_or_path=model_name_or_path)

        model = OCRRecognition(config)
        model = DeployUtils.model_eval(model=model, device=self.device, fp16_full_eval=self.fp16_full_eval)

        self.ocr_recognition_model = model

    def build_model(self):
        """
        加载模型：

        :return:
        """
        detection_model_name_or_path = self.config.model_path_detector
        recognition_model_name_or_path = self.config.model_path_recognizer
        if self.config.model_path is not None and self.config.model_path != "":
            detection_model_name_or_path = os.path.join(self.config.model_path, "detection")
            recognition_model_name_or_path = os.path.join(self.config.model_path, "recognition")

        if detection_model_name_or_path == "":
            detection_model_config = self.model_dict["detection"][self.config.detector]["general"]["model"]
            detection_model_name_or_path = os.path.join(Constants.SCOPE_MODEL_BASE_DIR, detection_model_config)

        if recognition_model_name_or_path == "":
            recognition_model_config = self.model_dict["recognition"][self.config.recognizer][self.config.task_type][
                "model"]
            recognition_model_name_or_path = os.path.join(Constants.SCOPE_MODEL_BASE_DIR, recognition_model_config)

        self.build_detection_model(model_name_or_path=detection_model_name_or_path)
        self.build_recognition_model(model_name_or_path=recognition_model_name_or_path)

        logger.info(f"加载OCR模型：{self.config.detector} - {self.config.recognizer} "
                    f"-task_type: {self.config.task_type}")

    def text_detection(self, image):
        """
        检测一张图片

        :param image:
        :return:
        """
        batch = self.ocr_detection_preprocessor(image)

        for k, v in batch.items():
            if k in ["image"]:
                v = v.to(self.device).unsqueeze(0)
                if self.fp16_full_eval:
                    v = v.half()
            batch[k] = v

        logger.info(f"detection image: {batch['image'].shape}")

        start = time.time()
        with torch.no_grad():
            outputs = self.ocr_detection_model(batch)

        pred = outputs['results']
        logger.info(f"检测 pred: {pred.shape}")

        outputs['results'] = outputs['results'].float()
        outputs = self.ocr_detection_model.postprocess(outputs)
        det_result = {"polygons": outputs['det_polygons']}

        det_result = det_result['polygons']

        use_time = time.time() - start
        logger.info(f"检测耗时：{use_time:3f} s. det_result: {det_result}")

        # sort detection result with coord
        det_result_list = det_result.tolist()
        det_result_list = sorted(det_result_list, key=lambda x: 0.01 * sum(x[::2]) / 4 + sum(x[1::2]) / 4)
        result = np.array(det_result_list)
        metric = {
            "use_time": use_time
        }
        return result, metric

    def _text_recognition(self, image, index=0):
        """
        识别一张图片

        :param image:
        :param index:
        :return:
        """
        raw_batch = self.ocr_recognition_preprocessor(image)
        batch = raw_batch['image'].to(self.device)
        if self.fp16_full_eval:
            batch = batch.half()

        # logger.info(f"recognition image: {batch.shape}")

        start = time.time()
        with torch.no_grad():
            outputs = self.ocr_recognition_model(batch)

        # logger.info(f"识别 outputs:{outputs.shape}")

        out_preds = self.ocr_recognition_model.postprocess(outputs)

        preds = {"text": out_preds['preds']}
        # logger.info(f"识别 preds: {preds}")

        use_time = time.time() - start
        logger.info(f"识别[{index}]耗时：{use_time:3f} s. result:{preds}")

        metric = {
            "use_time": use_time
        }

        return preds, metric

    def text_recognition(self, det_result, image_full):
        """
        文字识别

        :param det_result:
        :param image_full:
        :return:
        """

        use_times = []
        output = []
        for i in range(det_result.shape[0]):
            pts = OcrCommonUtils.order_point(det_result[i])
            image_crop = OcrCommonUtils.crop_image(image_full, pts)
            result, metric = self._text_recognition(image_crop, index=i)
            one_use_time = metric['use_time']
            use_times.append(one_use_time)

            output.append({
                "index": i + 1,
                "text": result['text'],
                "bbox": pts,
            })

        total_use_time = sum(use_times)
        avg_use_time = total_use_time / len(use_times)
        metric = {
            "use_time": total_use_time,
            "avg_use_time": avg_use_time,
            "total": len(use_times),
        }
        logger.info(f"识别结束：{metric}")

        return output, metric

    def show_ocr_result(self, ocr_result: List):
        output = []
        for item in ocr_result:
            output.append([item['index'], item['text'], ','.join([str(e) for e in list(item['bbox'].reshape(-1))])])

        ocr_result_pd = pd.DataFrame(output, columns=['检测框序号', '行识别结果', '检测框坐标'])

        return ocr_result_pd

    def __call__(self, inputs, save_result=True):
        image_full = OcrCommonUtils.read_image(inputs, return_nparray=True)
        det_result, det_metric = self.text_detection(image_full, )

        ocr_result, rec_metric = self.text_recognition(det_result=det_result, image_full=image_full)

        metric = {
            "detection": det_metric,
            "recognition": rec_metric,
        }

        logger.info(f"OCR识别结束：{metric}")

        if self.output_dir is not None and save_result:

            ocr_result_show = self.show_ocr_result(ocr_result)
            logger.info(f"det_result: {det_result}")
            logger.info(f"ocr_result: {ocr_result}")
            logger.info(f"ocr_result_show: {ocr_result_show}")
            logger.info(f"metric: {metric}")

            save_file = f"{self.output_dir}/ocr_{FileUtils.get_file_name(inputs)}_{TimeUtils.now_str_short()}.png"
            FileUtils.check_file_exists(save_file)

            image_draw = OcrCommonUtils.draw_boxes(image_full=image_full, det_result=det_result)
            image_draw = Image.fromarray(image_draw)
            image_draw.save(save_file)
            logger.info(f"save_file: {save_file}")

            ocr_result_file = save_file.replace(".png", ".txt")
            ocr_result_show.to_csv(ocr_result_file, header=True, index=False, sep="\t", )

            ocr_result_to_save = []
            for item in ocr_result:
                new_item = deepcopy(item)
                new_item['bbox'] = item['bbox'].tolist()
                ocr_result_to_save.append(new_item)

            metric["result"] = ocr_result_to_save

            ocr_result_json_file = save_file.replace(".png", ".json")
            FileUtils.dump_json(ocr_result_json_file, metric)

        return det_result, ocr_result, metric
