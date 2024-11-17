#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : processor_mtl_tabnet.py
# @Author   : cycloneboy
# @Date     : 20xx/12/15 - 22:37
import copy
from typing import Dict, Any

import cv2
import numpy as np

from pdftable.model.mtl_tabnet.configuration_mtl_tabnet import MtlTabnetConfig
from pdftable.model.table.lgpma.base_utils import imread
from pdftable.model.table.lgpma.lgpma_preprocess import Compose
from pdftable.model.table.mtl_tabnet.master_post_processor import MasterPostProcessor
from pdftable.utils import FileUtils, logger, TimeUtils
from pdftable.utils.ocr import OcrCommonUtils

__all__ = [
    "MtlTabNetPreProcessor",
    "MtlTabNetPostProcessor"
]


class MtlTabNetPreProcessor(object):

    def __init__(self, config: MtlTabnetConfig):
        super().__init__()

        self.config = config

        self.test_pipeline = self.build_test_pipeline()

    def build_test_pipeline(self):
        test_pipeline = Compose(self.config.get_test_pipeline_config())
        return test_pipeline

    def replace_image_to_tensor(self, pipelines):
        """Replace the ImageToTensor transform in a data pipeline to
        DefaultFormatBundle, which is normally useful in batch inference.

        Args:
            pipelines (list[dict]): Data pipeline configs.

        Returns:
            list: The new pipeline list with all ImageToTensor replaced by
                DefaultFormatBundle.

        Examples:
        """
        pipelines = copy.deepcopy(pipelines)
        for i, pipeline in enumerate(pipelines):
            if pipeline['type'] == 'ImageToTensor':
                logger.warn(
                    '"ImageToTensor" pipeline is replaced by '
                    '"DefaultFormatBundle" for batch inference. It is '
                    'recommended to manually replace it in the test '
                    'data pipeline in your config file.', UserWarning)
                pipelines[i] = {'type': 'DefaultFormatBundle'}
        return pipelines

    def process(self, image_file):
        img = imread(image_file)
        is_ndarray = isinstance(img, np.ndarray)
        cfg = self.config.config.copy()
        if is_ndarray:
            cfg.test_pipeline[0].type = 'LoadImageFromNdarray'

        cfg.test_pipeline = self.replace_image_to_tensor(cfg.test_pipeline)

        if is_ndarray:
            data = dict(img=img)
        else:
            data = dict(img_info=dict(filename=img), img_prefix=None)
        test_pipeline = Compose(cfg.test_pipeline)

        data = test_pipeline(data)

        image_data = data["img"].data
        run_data = {
            "img_metas": [[data["img_metas"].data]],
            "img": image_data.unsqueeze(0)
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


class MtlTabNetPostProcessor(object):

    def __init__(self, config: MtlTabnetConfig, output_dir=None, show_info=True):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.show_info = show_info
        self.postprocess = MasterPostProcessor(output_dir=self.output_dir)

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = inputs['results'][0]
        img_path = inputs["inputs"]

        pred = self.postprocess(result, file_name=img_path)

        pred_text = result['text']

        bbox_list = pred["new_bbox"]
        bbox_list = OcrCommonUtils.box_list_two_point_to_four_point(bbox_list)

        structure_str = pred['structure_str']
        structure_str_list = pred['structure_str_list']
        html_context = pred['html_context']

        result = {
            'polygons': bbox_list,
            'structure_str_list': structure_str_list,
            'structure_str': structure_str,
            'html_context': html_context,
            "inputs": inputs["inputs"]
        }

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
                logger.info(f"保存MtlTabNet识别结果：cell总量：{len(boxes)} - {image_name} ")
