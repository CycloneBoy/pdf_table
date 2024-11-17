#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：processer_lore
# @Author  ：cycloneboy
# @Date    ：20xx/5/26 14:22
import io
import os
from typing import Union, Dict, Any

import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps
import requests
import torch

from .configuration_lore import LoreConfig
from .lineless_table_process import get_affine_transform_upper_left, \
    process_logic_output, get_affine_transform
from ...utils import CommonUtils, FileUtils, logger, TimeUtils
from ...utils.ocr import OcrCommonUtils

"""
Table Lore Preprocessor
"""

__all__ = [
    'TableLorePreProcessor',
    'TableLorePostProcessor',
]


class TableLorePreProcessor(object):

    def __init__(self, config: LoreConfig):
        self.config = config
        self.device = CommonUtils.get_torch_device()
        self.resolution = config.resolution

    def read_image(self, image_path_or_url):
        if image_path_or_url.startswith("http"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
        image = image.convert("RGB")

        image = np.array(image)
        img = image
        return img

    def convert_to_ndarray(self, input) -> np.ndarray:
        if isinstance(input, str):
            img = np.array(self.read_image(input))
        elif isinstance(input, PIL.Image.Image):
            img = np.array(input.convert('RGB'))
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 2:
                input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
            img = input[:, :, ::-1]
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')
        return img

    def process(self, img):
        mean = np.array([0.408, 0.447, 0.470],
                        dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.289, 0.274, 0.278],
                       dtype=np.float32).reshape(1, 1, 3)
        height, width = img.shape[0:2]

        inp_height, inp_width = self.resolution
        if self.config.upper_left:
            c = np.array([0, 0], dtype=np.float32)
            s = max(height, width) * 1.0

            trans_input = get_affine_transform_upper_left(c, s, 0,
                                                          [inp_width, inp_height])
        else:
            c = np.array([width / 2., height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
            trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])

        resized_image = cv2.resize(img, (width, height))
        inp_image = cv2.warpAffine(
            resized_image,
            trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height,
                                                      inp_width)
        images = torch.from_numpy(images)
        c = torch.from_numpy(c)
        meta = {
            'c': c,
            's': s,
            'input_height': inp_height,
            'input_width': inp_width,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4
        }

        result = {'pixel_values': images, 'meta': meta}

        result = TableLorePreProcessor.update_meta(result, add_batch=True)

        return result

    @staticmethod
    def update_meta(one_item, add_batch=False):
        if "meta" not in one_item:
            return one_item
        meta = one_item["meta"]
        new_meta = []
        new_meta.extend(meta['c'])
        new_meta.append(meta['s'])
        new_meta.append(meta['input_height'])
        new_meta.append(meta['input_width'])
        new_meta.append(meta['out_height'])
        new_meta.append(meta['out_width'])
        if "img_id" in meta:
            new_meta.append(meta['img_id'])
        new_meta = torch.from_numpy(np.array(new_meta)).long()
        if add_batch:
            new_meta = new_meta.unsqueeze(0)
        one_item["meta"] = new_meta

        return one_item

    def __call__(self, inputs):
        """process the raw input data
        Args:
            inputs:
                - A string containing an HTTP link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL or opencv directly
        Returns:
            outputs: the preprocessed image
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        data_batch = []
        for item in inputs:
            img = self.convert_to_ndarray(item)[:, :, ::-1]

            data = self.process(img)

            data["inputs"] = item
            # data["meta"]["inputs"] = item
            data_batch.append(data)
        return data_batch


class TableLorePostProcessor(object):

    def __init__(self, config: LoreConfig, output_dir=None, show_info=True):
        self.config = config
        self.output_dir = output_dir
        self.show_info = show_info

    def __call__(self, inputs: Dict[str, Any], image_name=None) -> Dict[str, Any]:
        if isinstance(inputs, Dict):
            raw_input = inputs.get("inputs", None)
            if 'results' in inputs:
                inputs = inputs['results']
            slct_dets = inputs['pred_boxes']
            logi = inputs['logits']
        else:
            raw_input = None
            slct_dets = inputs[0]
            logi = inputs[1]
        slct_logi = process_logic_output(logi)
        result = {
            "polygons": slct_dets.detach().cpu().numpy()[0],
            # "boxes": slct_dets,
            "logi": np.array(slct_logi[0].cpu().numpy()),

        }
        if raw_input is not None:
            result["inputs"] = raw_input

        if self.output_dir is not None:
            image_name = raw_input if image_name is None else image_name[-1]
            self.save_result(preds=result, image_name=image_name)

        return result

    def save_result(self, preds: Dict, image_name):
        boxes = preds["polygons"]
        logits = preds["logi"]

        if self.output_dir is not None:

            if self.config.eval:
                base_dir = f"{self.output_dir}/{self.config.task_type}"
                base_name = FileUtils.get_file_name(image_name, add_end=True)
                image_file = f"{base_dir}/image/{base_name}"
                boxe_file = f"{base_dir}/center/{base_name}.txt"
                logit_file = f"{base_dir}/logi/{base_name}.txt"
            else:
                image_file = f"{self.output_dir}/{FileUtils.get_file_name(image_name)}_{TimeUtils.now_str_short()}.jpg"
                boxe_file = image_file.replace(".jpg", "_box.txt")
                logit_file = image_file.replace(".jpg", "_logit.txt")

            FileUtils.check_file_exists(image_file)
            FileUtils.check_file_exists(boxe_file)
            FileUtils.check_file_exists(logit_file)

            OcrCommonUtils.draw_lore_bboxes(image_name=image_name,
                                            boxes=boxes,
                                            logits=logits,
                                            save_name=image_file)

            box_result = []
            for bbox in boxes:
                box_list = []
                for i in range(0, 4):
                    box_list.append(f"{bbox[2 * i]:.5f},{bbox[2 * i + 1]:.5f}")
                box_result.append(";".join(box_list))

            FileUtils.save_to_text(boxe_file, "\n".join(box_result) + "\n")

            logit_result = []
            for logit in logits:
                logit_result.append(",".join([str(int(item)) for item in logit]))
            FileUtils.save_to_text(logit_file, "\n".join(logit_result) + "\n")

            if self.show_info:
                logger.info(f"保存LORE识别结果：cell总量：{len(boxes)} - {image_name} ")
