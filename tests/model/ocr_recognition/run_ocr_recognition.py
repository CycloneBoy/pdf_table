#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_ocr_recognition
# @Author  ：cycloneboy
# @Date    ：20xx/7/14 13:16

import time

import numpy as np
import torch

from pdftable.model.ocr_recognition import OCRRecognitionPreprocessor, OCRRecognitionPostProcessor, \
    OCRRecognitionOnnxConfig, OCRRecognitionConfig, OCRRecognition
from pdftable.utils import logger, DeployUtils, Constants, CommonUtils

"""
run  ocr recognition
"""


class RunOcrRecognition(object):

    def __init__(self):
        self.scope_model_base_dir = Constants.SCOPE_MODEL_BASE_DIR
        self.model_name = "damo/cv_resnet18_ocr-detection-db-line-level_damo"
        self.model_name_or_path = f"{self.scope_model_base_dir}/{self.model_name}"
        self.fp16_full_eval = True
        # self.fp16_full_eval = False
        self.do_transform = True
        # self.show_model = True
        self.show_model = False
        self._init_class = "ConvNextViT"
        self.device = "cuda"
        self.image_dir = f'f"{Constants.DATA_DIR}/pdf/table_file/temp_file/image_output'
        self.img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'

        # self.img_url = f"{self.image_dir}/img00045.png"
        self.preprocessor: OCRRecognitionPreprocessor = None
        self.post_processor: OCRRecognitionPostProcessor = None

        # self.model_backbone = "CRNN"
        # self.model_backbone = "LightweightEdge"
        self.model_backbone = "ConvNextViT"

        # self.task_type = "general"
        self.task_type = "document"

        self.model_dict = {
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

        self.get_model_name_or_path()
        self.build_processor()

    def get_model_name_or_path(self):
        if self.model_backbone in ["CRNN", "LightweightEdge"]:
            self.task_type = "general"
            logger.info(f"{self.model_backbone} only support task type: {self.task_type}")

        model_config = self.model_dict[self.model_backbone][self.task_type]
        self.model_name = model_config["model"]
        self.img_url = model_config["image_url"]

        self.model_name_or_path = f"{self.scope_model_base_dir}/{self.model_name}"
        logger.info(f"当前加载模型：{self.model_backbone} - {self.task_type}")

    def build_processor(self):
        config = OCRRecognitionConfig.from_pretrained(self.model_name_or_path)
        # config = OCRRecognitionConfig(recognizer=self.model_backbone,
        #                               model_path=self.model_name_or_path)
        config.model_path = self.model_name_or_path

        self.preprocessor = OCRRecognitionPreprocessor(config)
        self.post_processor = OCRRecognitionPostProcessor(config)

        return config

    def load_model(self, ):
        config = self.build_processor()
        logger.info(f"config: {config}")

        model = OCRRecognition(config)

        save_dir2 = f"{Constants.DATA_DIR}/txt/model_scope/ocr_recognition/ocr_rec_{config.recognizer}_{self.task_type}"
        if self.show_model:
            model_net2, model_params2 = CommonUtils.print_model_param(model, save_dir=save_dir2, use_numpy=True)

        return model

    def run(self, img_url=None):
        model = self.load_model()

        model = DeployUtils.model_eval(model=model, device=self.device, fp16_full_eval=self.fp16_full_eval)

        if img_url is None:
            img_url = self.img_url

        batch = self.preprocessor(img_url)
        # batch = batch.to(self.device)
        # if self.fp16_full_eval:
        #     batch.half()

        # print(f"{batch.shape}")

        # batch['img'] = batch['img'].to(self.device).unsqueeze(0)

        for k, v in batch.items():
            if k in ["image"]:
                v = v.to(self.device)
                if self.fp16_full_eval:
                    v = v.half()
            batch[k] = v

        image = batch['image']
        logger.info(f"image: {image.shape}")

        start = time.time()
        with torch.no_grad():
            outputs = model(image)

        print(f"outputs:{outputs.shape}")

        out_preds = model.postprocess(outputs)

        preds = out_preds['preds']
        print(f"preds:{preds}")

        use_time = time.time() - start
        print(f"耗时：{use_time:3f} s. result:{preds}")

        return preds

    def load_config(self):
        config = OCRRecognitionConfig.from_pretrained(self.model_name_or_path)
        logger.info(f"config: {config}")

    def run_all(self):
        for backbone, v in self.model_dict.items():
            if backbone not in ["ConvNextViT"]:
                continue
            for task, model_config in v.items():
                self.model_backbone = backbone
                self.task_type = task
                logger.info(f"开始执行：{self.model_backbone} - {self.task_type}")
                self.get_model_name_or_path()
                self.run()

                self.export_model_onnx()
                self.compare_model()

    def export_model_onnx(self):
        self.get_model_name_or_path()
        model = self.load_model()

        model = DeployUtils.model_eval(model=model, device=self.device, fp16_full_eval=self.fp16_full_eval)

        onnx_config = OCRRecognitionOnnxConfig(model.config)
        logger.info(f"onnx_config: {onnx_config.outputs}")

        if self.model_backbone in ["CRNN", "LightweightEdge"]:
            model_inputs = torch.rand(1, 3, 32, 640)
        elif self.model_backbone == "ConvNextViT":
            model_inputs = torch.rand(3, 3, 32, 300)

        model_inputs = model_inputs.to(self.device).half()
        onnx_path = f"{self.model_name_or_path}/fp16_model.onnx"

        logger.info(f"model_inputs: {model_inputs.shape} - {model_inputs}")

        export_model = model.recognizer
        DeployUtils.export_model(model=export_model, model_inputs=model_inputs, onnx_path=onnx_path,
                                 onnx_config=onnx_config)

    def compare_model(self, ):
        predictor = DeployUtils.prepare_onnx_model(onnx_dir=self.model_name_or_path)

        batch = self.preprocessor(self.img_url)

        img = batch['image'].to(self.device)
        if self.fp16_full_eval:
            img = img.half()

        input_dict = {
            "pixel_values": img
        }
        input_dict = {key: value.detach().cpu().numpy() if not isinstance(value, np.ndarray) else value for key, value
                      in
                      input_dict.items()}
        results = predictor.run(None, input_dict)

        logger.info(f'results: {results[0].shape}')
        outputs = torch.tensor(results[0]).to(self.device).float()

        logger.info(f'outputs: {outputs}')
        predict = self.post_processor(outputs)
        predict = predict["preds"]
        logger.info(f'predict: {predict}')

    def test_ms_convnext(self):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        import cv2

        ocr_recognition = pipeline(Tasks.ocr_recognition,
                                   model='damo/cv_crnn_ocr-recognition-general_damo')

        ### 使用url
        img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
        result = ocr_recognition(img_url)
        print(result)



def main():
    runner = RunOcrRecognition()
    # runner.run()
    # runner.load_config()
    # runner.run_all()
    # runner.export_model_onnx()
    # runner.compare_model()
    runner.test_ms_convnext()


if __name__ == '__main__':
    main()
