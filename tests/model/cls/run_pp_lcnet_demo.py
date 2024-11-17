#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_pp_lcnet_demo
# @Author  ：cycloneboy
# @Date    ：20xx/9/28 17:10
import os

import torch

from pdftable.model.cls.cls_pp_lcnet import PPLCNet
from pdftable.model.cls.image_processing_pplcnet import PPLCNetImageProcessor
from pdftable.model.ocr_pdf.cls_image_pulc_task import ClsImagePulcTask
from pdftable.model.ocr_pdf.ocr_table_preprocess_task import OcrTablePreprocessTask
from pdftable.model.table.lgpma.checkpoint import load_checkpoint
from pdftable.utils import Constants, logger, FileUtils, CommonUtils


class RunPpLcnetDemo(object):

    def __init__(self):
        self.model_name = "PPLcnet"
        # self.task_name = "table_attribute"
        # self.task_name = "text_image_orientation"
        # self.task_name = "textline_orientation"
        self.task_name = "language_classification"

        self.task_config = {
            "table_attribute": {
                "use_ssld": True,
                "class_num": 6,
            },
            "text_image_orientation": {
                "use_ssld": True,
                "class_num": 4,
            },
            "textline_orientation": {
                "use_ssld": True,
                "class_num": 2,
                "stride_list": [2, [2, 1], [2, 1], [2, 1], [2, 1]]
            },
            "language_classification": {
                "use_ssld": True,
                "class_num": 10,
                "stride_list": [2, [2, 1], [2, 1], [2, 1], [2, 1]]
            }
        }

        self.class_num = 6
        self.device = "cuda:0"
        self.do_visualize = True
        self.output_dir = FileUtils.get_output_dir_with_time()
        self.model_base_dir = f"{Constants.SCOPE_MODEL_BASE_DIR}/cycloneboy"
        self.image_base_dir = f'{Constants.DATA_DIR}/pulc_demo_imgs'

    def get_config(self):
        model_params = self.task_config[self.task_name]

        self.class_num = model_params["class_num"]
        return model_params

    def run(self):
        model, image_processor = self.build_model()

        # image = f"{self.image_base_dir}/{self.task_name}/val_3253.jpg"
        # image = f"{self.image_base_dir}/{self.task_name}/val_9.jpg"
        # image = f"{self.image_base_dir}/{self.task_name}/img_rot180_demo.jpg"
        # image = f"{self.image_base_dir}/{self.task_name}/textline_orientation_test_0_0.png"
        image = f"{self.image_base_dir}/{self.task_name}/word_20.png"
        data = image_processor(image, return_tensors="pt")

        logger.info(f"data: {data}")
        pixel_values = data["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = model(pixel_values)

        logger.info(f"outputs: {outputs}")
        results = image_processor.post_process(outputs)
        logger.info(f"results: {results}")

    def build_model(self):
        model = PPLCNet(**self.get_config())
        os.makedirs(self.output_dir, exist_ok=True)
        save_dir = f"{self.output_dir}/{self.model_name}_{self.task_name}".lower()
        CommonUtils.print_model_param(model, save_dir=save_dir)

        model.to(self.device)
        model.eval()

        map_loc = 'cpu' if self.device == 'cpu' else None
        pretrain_weight = f"{self.model_base_dir}/{self.task_name}/pytorch_model.bin"
        checkpoint = load_checkpoint(model, pretrain_weight, map_location=map_loc)

        image_processor = PPLCNetImageProcessor(task=self.task_name)
        return model, image_processor

    def run_batch(self):
        model, image_processor = self.build_model()

        image_batch = {
            "text_image_orientation": []
        }
        # image_list = image_batch[self.task_name]
        image_list = FileUtils.list_file_prefix(file_dir=f"{self.image_base_dir}/{self.task_name}", add_parent=True)

        for image in image_list:
            image_name = FileUtils.get_file_name(image)
            data = image_processor(image, return_tensors="pt")

            logger.info(f"{image_name} - data: {data}")
            pixel_values = data["pixel_values"].to(self.device)

            with torch.no_grad():
                outputs = model(pixel_values)

            logger.info(f"outputs: {outputs}")
            results = image_processor.post_process(outputs)
            logger.info(f"eval result: {image_name} results: {results}")

    def run_cls_task(self):
        task = ClsImagePulcTask(task_type=self.task_name)
        # task = OcrTablePreprocessTask()
        image_list = FileUtils.list_file_prefix(file_dir=f"{self.image_base_dir}/{self.task_name}", add_parent=True)

        for image in image_list:
            image_name = FileUtils.get_file_name(image)

            outputs = task(image)
            logger.info(f"{image_name} - outputs: {outputs}")


def main():
    runner = RunPpLcnetDemo()
    # runner.run()
    # runner.run_batch()
    runner.run_cls_task()


if __name__ == '__main__':
    main()
