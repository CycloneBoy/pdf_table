#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：processor_slanet
# @Author  ：cycloneboy
# @Date    ：20xx/9/4 15:56
from typing import Dict, Any

import PIL
import numpy as np
import requests
import torch
from PIL import Image

from .configuration_slanet import SLANetConfig
from .table_postprocess import TableLabelDecode
from ..db_pp.processor_ocr_db_pp import create_operators, transform

__all__ = [
    "SLANetPreprocessor",
    "SLANetPostProcessor",
]


class SLANetPreprocessor(object):

    def __init__(self, config: SLANetConfig):
        super().__init__()

        self.config = config

        self.pre_process_list = [
            {'ResizeTableImage': {
                'max_len': config.table_max_len, }
            }, {
                'NormalizeImage': {
                    'std': [0.229, 0.224, 0.225],
                    'mean': [0.485, 0.456, 0.406],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }, {
                'PaddingTableImage': {
                    'size': [config.table_max_len, config.table_max_len]
                }
            },
            {
                'ToCHWImage': None
            }, {
                'KeepKeys': {
                    'keep_keys': ['image', 'shape']
                }
            }]

        self.preprocess_op = create_operators(self.pre_process_list)

    def read_image(self, image_path_or_url):
        image_content = image_path_or_url
        if image_path_or_url.startswith("http"):
            image_content = requests.get(image_path_or_url, stream=True).raw
        image = Image.open(image_content)
        image = image.convert("RGB")

        return image

    def __call__(self, inputs):
        """process the raw input data
        Args:
            inputs:
                - A string containing an HTTP link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL(PIL.Image.Image) or opencv(np.ndarray) directly, 3 channels RGB
        Returns:
            outputs: the preprocessed image
        """
        if isinstance(inputs, str):
            img = np.array(self.read_image(inputs))
        elif isinstance(inputs, PIL.Image.Image):
            img = np.array(inputs)
        elif isinstance(inputs, np.ndarray):
            img = inputs
        else:
            raise TypeError(
                f'inputs should be either str, PIL.Image, np.array, but got {type(inputs)}'
            )

        img = img[:, :, ::-1]
        height, width, _ = img.shape

        ori_im = img.copy()

        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0

        resized_img = torch.from_numpy(img).float().unsqueeze(0)

        result = {
            'image': resized_img,
            'org_shape': ori_im.shape,
            'shape_list': shape_list,
            "inputs": inputs
        }

        # logger.info(f"image: {height} x {width}  -> {shape_list}")
        return result


class SLANetPostProcessor(object):

    def __init__(self, config: SLANetConfig):
        super().__init__()

        self.config = config
        self.lang = config.lang
        self.vocab_file = config.get_vocab_file()

        self.postprocess_op = TableLabelDecode(character_dict_path=self.vocab_file,
                                               merge_no_span_structure=config.merge_no_span_structure, )

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pred = inputs['results']
        shape_list = np.expand_dims(inputs['shape_list'], axis=0)

        post_result = self.postprocess_op(pred, [shape_list])

        structure_str_list = post_result['structure_batch_list'][0]
        bbox_list = post_result['bbox_batch_list'][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = [
                                 '<html>', '<body>', '<table>'
                             ] + structure_str_list + ['</table>', '</body>', '</html>']

        if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
            new_bboxes = []
            for (x1, y1, x2, y2) in bbox_list:
                box = [x1, y1, x2, y1, x2, y2, x1, y2]
                new_bboxes.append(box)
            bbox_list = np.array(new_bboxes)

        result = {
            'polygons': bbox_list,
            'structure_str_list': structure_str_list,
            "inputs": inputs["inputs"]
        }
        return result
