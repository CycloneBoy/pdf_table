#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_table_centernet
# @Author  ：cycloneboy
# @Date    ：20xx/5/25 18:18
from typing import Dict, Any

import cv2
import numpy as np
import torch
from pdftable.model import OCRTableCenterNetPreProcessor, TableRecModel, OCRTableCenterNetPostProcessor, \
    TableCenterNetConfig, TableStructureRec, TableCenterNetOnnxConfig

from pdftable.utils import CommonUtils, Constants, DeployUtils, logger


class RunTableCenternet(object):

    def __init__(self):
        self.scope_model_base_dir = Constants.SCOPE_MODEL_BASE_DIR
        self.model_name = f"iic/cv_dla34_table-structure-recognition_cycle-centernet"
        self.model_name_or_path = f"{self.scope_model_base_dir}/{self.model_name}"

        self.image_dir = f'{Constants.OUTPUT_DIR}/image_output'

        self.fp16_full_eval = True
        # self.fp16_full_eval = False
        self.device = CommonUtils.get_torch_device()

        # self.show_model = True
        self.show_model = False
        self.infer_model = None

        self.preprocessor = OCRTableCenterNetPreProcessor()
        self.post_processor = OCRTableCenterNetPostProcessor()

        self.img_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/table_recognition.jpg'
        # self.img_url = f"{self.image_dir}/img00045.png"

    def load_model(self, ):
        # model = TableRecModel()
        #
        # model_path = f"{self.model_name_or_path}/pytorch_model.pt"
        # checkpoint = torch.load(model_path, map_location="cpu")
        # params_pretrained = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        # model.load_state_dict(params_pretrained)
        config = TableCenterNetConfig(model_path=self.model_name_or_path)
        model = TableStructureRec(config)

        save_dir = f"{Constants.DATA_DIR}/txt/model_scope/dla34_table-structure-recognition_cycle/dla34_"
        if self.show_model:
            model_net2, model_params2 = CommonUtils.print_model_param(model, save_dir=save_dir, use_numpy=True)

        return model

    def run(self):
        model = self.load_model()

    def our_run_model(self, model_name_or_path=None, img_url=None):
        if model_name_or_path is None:
            model_name_or_path = self.model_name_or_path
        model = self.load_model()

        model = DeployUtils.model_eval(model=model, device=self.device, fp16_full_eval=self.fp16_full_eval)

        if img_url is None:
            img_url = self.img_url

        raw_batch = self.preprocessor(img_url)
        one_batch = raw_batch[0]
        batch = one_batch["image"]
        batch = batch.to(self.device)
        if self.fp16_full_eval:
            batch = batch.half()

        print(f"{batch.shape}")

        with torch.no_grad():
            outputs = model(batch)

        print(f"outputs:{outputs}")
        # for k, v in outputs[0].items():
        #     logger.info(f"{k} - {v.shape}")

        results = {'results': outputs, 'meta': one_batch['meta']}

        out_preds = self.post_processor(results)

        preds = out_preds
        print(f"preds:{preds}")

        return preds

    def export_model_onnx(self):
        model = self.load_model()

        model = DeployUtils.model_eval(model=model, device=self.device, fp16_full_eval=self.fp16_full_eval)

        onnx_config = TableCenterNetOnnxConfig(model.config)
        logger.info(f"onnx_config: {onnx_config.outputs}")

        model_inputs = torch.rand(1, 3, 1024, 1024)

        model_inputs = model_inputs.to(self.device).half()
        onnx_path = f"{self.model_name_or_path}/fp16_model.onnx"

        logger.info(f"model_inputs: {model_inputs.shape} - {model_inputs}")

        export_model = model
        DeployUtils.export_model(model=export_model, model_inputs=model_inputs, onnx_path=onnx_path,
                                 onnx_config=onnx_config)

    def compare_model(self, ):
        predictor = DeployUtils.prepare_onnx_model(onnx_dir=self.model_name_or_path)

        batch = self.preprocessor(self.img_url)
        one_batch = batch[0]

        img = one_batch['image'].to(self.device)
        if self.fp16_full_eval:
            img = img.half()

        input_dict = {
            "pixel_values": img
        }
        input_dict = {key: value.detach().cpu().numpy() if not isinstance(value, np.ndarray) else value for key, value
                      in
                      input_dict.items()}
        raw_outputs = predictor.run(None, input_dict)
        outputs = raw_outputs[0]

        logger.info(f'outputs: {len(raw_outputs)}')
        logger.info(f'predictor input: {predictor.get_inputs()}')
        logger.info(f'predictor output: {predictor.get_outputs()}')
        logger.info(f'predictor output: {predictor.get_modelmeta()}')
        logger.info(f'outputs: {outputs.shape}')
        logger.info(f'outputs: {outputs}')

        for index, v in enumerate(raw_outputs):
            logger.info(f"{index} - {v.shape}")

        predict = {
            "hm": raw_outputs[0],
            "v2c": raw_outputs[1],
            "c2v": raw_outputs[2],
            "reg": raw_outputs[3],
        }

        for k, v in predict.items():
            v = torch.tensor(v).to(self.device)
            if self.fp16_full_eval:
                v = v.half()
            predict[k] = v

        results = {'results': predict, 'meta': one_batch['meta']}

        out_preds = self.post_processor(results)

        preds = out_preds

        logger.info(f'predict: {preds}')



def main():
    runner = RunTableCenternet()
    # runner.run()
    runner.our_run_model()


if __name__ == '__main__':
    main()
