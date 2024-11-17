#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_ocr_db_net
# @Author  ：cycloneboy
# @Date    ：20xx/7/13 18:05
import time

import numpy as np
import torch

from pdftable.model import OCRDetectionPreprocessor, OCRDetectionPostProcessor, DbNetConfig, OCRDetectionDbNet, \
    DbNetOnnxConfig
from pdftable.utils import Constants, CommonUtils, DeployUtils, logger


class RunOcrDbNet(object):

    def __init__(self):
        self.scope_model_base_dir = Constants.SCOPE_MODEL_BASE_DIR
        self.model_name = "damo/cv_resnet18_ocr-detection-db-line-level_damo"
        self.model_name_or_path = f"{self.scope_model_base_dir}/{self.model_name}"
        self.fp16_full_eval = True
        # self.fp16_full_eval = False
        self.do_transform = True
        self._init_class = "ConvNextViT"
        self.device = "cuda"
        # self.image_dir = f'{Constants.OUTPUT_DIR}/image_output'
        self.img_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg'
        # self.img_url = f"{self.image_dir}/img00045.png"

        self.preprocessor: OCRDetectionPreprocessor = None
        self.post_processor: OCRDetectionPostProcessor = None

        self.model_backbone = "resnet18"
        # self.model_backbone = "proxylessnas"

        self.get_model_name_or_path()
        self.build_processor()

    def get_model_name_or_path(self):
        if self.model_backbone == "resnet18":
            self.model_name = "damo/cv_resnet18_ocr-detection-db-line-level_damo"
        elif self.model_backbone == "proxylessnas":
            self.model_name = "damo/cv_proxylessnas_ocr-detection-db-line-level_damo"

        self.model_name_or_path = f"{self.scope_model_base_dir}/{self.model_name}"

    def build_processor(self):
        config = DbNetConfig(backbone=self.model_backbone,
                             model_path=self.model_name_or_path)

        self.preprocessor = OCRDetectionPreprocessor(config)
        self.post_processor = OCRDetectionPostProcessor(config)

        return config

    def load_model(self, show_model=False):
        config = self.build_processor()
        model = OCRDetectionDbNet(config)

        save_dir2 = f"{Constants.DATA_DIR}/txt/model_scope/ocr_dbnet/{config.backbone}_ocr_db"
        if show_model:
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
                v = v.to(self.device).unsqueeze(0)
                if self.fp16_full_eval:
                    v = v.half()
            batch[k] = v

        logger.info(f"image: {batch['image'].shape}")

        start = time.time()
        with torch.no_grad():
            outputs = model(batch)

            pred = outputs['results']
            logger.info(f"pred: {pred.shape}")

            outputs['results'] = outputs['results'].float()
            logger.info(f"outputs: {outputs['results'].shape}")
            logger.info(f"outputs: {outputs}")
            outputs = model.postprocess(outputs)

        result = {"polygons": outputs['det_polygons']}

        use_time = time.time() - start
        print(f"耗时：{use_time:3f} s. result:{result}")

        return result

    def export_model_onnx(self):
        model = self.load_model()

        model = DeployUtils.model_eval(model=model, device=self.device, fp16_full_eval=self.fp16_full_eval)

        onnx_config = DbNetOnnxConfig(model.config)
        logger.info(f"onnx_config: {onnx_config.outputs}")

        model_inputs = torch.rand(1, 3, 1056, 736).to(self.device).half()
        onnx_path = f"{self.model_name_or_path}/fp16_model.onnx"

        logger.info(f"model_inputs: {model_inputs.shape} - {model_inputs}")

        export_model = model.detector
        DeployUtils.export_model(model=export_model, model_inputs=model_inputs, onnx_path=onnx_path,
                                 onnx_config=onnx_config)

    def compare_model(self, ):
        predictor = DeployUtils.prepare_onnx_model(onnx_dir=self.model_name_or_path)

        batch = self.preprocessor(self.img_url)

        image = batch['image'].to(self.device).unsqueeze(0)
        if self.fp16_full_eval:
            image = image.half()

        org_shape = batch['org_shape']

        input_dict = {
            "pixel_values": image
        }
        input_dict = {key: value.detach().cpu().numpy() if not isinstance(value, np.ndarray) else value for key, value
                      in
                      input_dict.items()}
        results = predictor.run(None, input_dict)

        logger.info(f'results: {results[0].shape}')
        outputs = {
            "results": torch.tensor(results[0]).to(self.device).float(),
            "org_shape": org_shape,
        }
        logger.info(f'outputs: {outputs}')
        predict = self.post_processor(outputs)
        logger.info(f'predict: {predict}')


def main():
    runner = RunOcrDbNet()
    runner.run()
    # runner.export_model_onnx()
    # runner.compare_model()


if __name__ == '__main__':
    main()
