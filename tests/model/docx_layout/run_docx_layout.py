#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_docx_layout
# @Author  ：cycloneboy
# @Date    ：20xx/9/27 18:47
import os

import cv2
import torch

from pdftable.model.docx_layout import DocXLayoutConfig, DocXLayoutModel, DocXLayoutPreProcessor, \
    DocXLayoutImagePostProcessor
from pdftable.utils import Constants, FileUtils
from pdftable.utils.ocr import OcrInferUtils


class RunDocxLayout(object):

    def __init__(self):
        self.model_name = "DocXLayout"
        self.device = "cuda:0"
        self.do_visualize = True
        self.output_dir = FileUtils.get_output_dir_with_time()
        self.checkpoint = f'{Constants.SCOPE_MODEL_BASE_DIR}/cycloneboy/cv_dla34_layout-analysis_docxlayout_general/DocXLayout_231012.pth'

    def run(self):
        config = DocXLayoutConfig()
        config.model_path = self.checkpoint

        model = DocXLayoutModel(config)

        pre_processor = DocXLayoutPreProcessor()
        post_processor = DocXLayoutImagePostProcessor(config)

        os.makedirs(self.output_dir, exist_ok=True)
        save_dir = f"{self.output_dir}/{self.model_name.lower()}"
        # CommonUtils.print_model_param(model, save_dir=save_dir)

        model.to(self.device)
        model.eval()

        image_file = f"{Constants.SRC_IMAGE_DIR}/page01.png"
        data = pre_processor(image_file, return_tensors="pt")

        # logger.info(f"data: {data}")
        pixel_values = data["image"].to(self.device)

        with torch.no_grad():
            outputs = model(pixel_values)

        results = post_processor(outputs, meta=data["meta"])

        layout_dets = results["bboxs"]
        # subfield_dets = results["subfield_dets"]
        # layout_res = results["layout_res"]
        # logger.info(f"layout_dets: {layout_dets}")
        # logger.info(f"subfield_dets: {subfield_dets}")
        # logger.info(f"layout_res: {layout_res}")

        image = cv2.imread(image_file)
        img_path = os.path.join(self.output_dir, FileUtils.get_file_name(image_file, add_end=True))
        OcrInferUtils.draw_text_layout_res(image, layout_res=layout_dets, save_path=img_path)


def main():
    runner = RunDocxLayout()
    runner.run()


if __name__ == '__main__':
    main()
