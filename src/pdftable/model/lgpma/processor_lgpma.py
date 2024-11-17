#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : processor_lgpma.py
# @Author   : cycloneboy
# @Date     : 20xx/12/10 - 18:05
from typing import Dict, Any

import cv2

from pdftable.model.lgpma.configuration_lgpma import LgpmaConfig
from pdftable.model.table.lgpma.lgpma_preprocess import Compose
from pdftable.utils import FileUtils, logger, TimeUtils
from pdftable.utils.ocr import OcrCommonUtils

__all__ = [
    "LgpmaPreProcessor",
    "LgpmaPostProcessor"
]


class LgpmaPreProcessor(object):

    def __init__(self, config: LgpmaConfig):
        super().__init__()

        self.config = config

        self.test_pipeline = Compose(self.config.get_test_pipeline_config())

    def process(self, image_file):
        data = dict(img=image_file)
        data = self.test_pipeline(data)

        image_data = data["img"][0].data
        run_data = {
            "img_metas": [[data["img_metas"][0].data]],
            "img": [image_data.unsqueeze(0)]
        }
        return run_data

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        data_batch = []
        for item in inputs:
            data = self.process(item)

            data["inputs"] = item
            data_batch.append(data)
        return data_batch


class LgpmaPostProcessor(object):

    def __init__(self, config: LgpmaConfig, output_dir=None, show_info=True):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.show_info = show_info

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pred = inputs['results'][0]

        bbox_list = pred['content_ann']['bboxes']
        structure_str_list = pred['html']

        bbox_list = OcrCommonUtils.box_list_two_point_to_four_point(bbox_list)

        result = {
            'polygons': bbox_list,
            'structure_str_list': structure_str_list,
            "inputs": inputs["inputs"]
        }

        # if self.output_dir is not None:
        #     self.save_result(preds=result, image_name=inputs["inputs"])

        return result

    def save_result(self, preds: Dict, image_name):
        boxes = preds["polygons"]

        if self.output_dir is not None:
            image_file = f"{self.output_dir}/{FileUtils.get_file_name(image_name)}_{TimeUtils.now_str_short()}.jpg"
            img = cv2.imread(image_name)
            bboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in boxes]
            for box in bboxes:
                for j in range(0, len(box), 2):
                    cv2.line(img, (box[j], box[j + 1]), (box[(j + 2) % len(box)], box[(j + 3) % len(box)]), (0, 0, 255),
                             1)
            cv2.imwrite(image_file, img)

            box_result = []
            for bbox in boxes:
                box_list = []
                for i in range(0, 4):
                    box_list.append(f"{bbox[2 * i]:.5f},{bbox[2 * i + 1]:.5f}")
                box_result.append(";".join(box_list))

            boxe_file = image_file.replace(".jpg", "_box.txt")
            FileUtils.save_to_text(boxe_file, "\n".join(box_result) + "\n")

            if self.show_info:
                logger.info(f"保存Lgpma识别结果：cell总量：{len(boxes)} - {image_name} ")
