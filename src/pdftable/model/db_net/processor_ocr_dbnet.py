#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：processor_ocr_dbnet
# @Author  ：cycloneboy
# @Date    ：20xx/7/13 18:37
from typing import Dict, Any

import PIL
import cv2
import math
import numpy as np
import requests
import torch
from PIL import Image

from .configuration_dbnet import DbNetConfig
from .ocr_detection_utils import boxes_from_bitmap, polygons_from_bitmap
from ...utils import logger

"""
ocr dbnet Preprocessor
"""

__all__ = [
    "OCRDetectionPreprocessor",
    "OCRDetectionPostProcessor",
]


class OCRDetectionPreprocessor(object):

    def __init__(self, config: DbNetConfig):
        """The base constructor for all ocr recognition preprocessors.

        """
        super().__init__()

        self.image_short_side = config.img_width

    def read_image(self, image_path_or_url):
        image_content = image_path_or_url
        if image_path_or_url.startswith("http"):
            image_content = requests.get(image_path_or_url, stream=True).raw
        image = Image.open(image_content)
        image = image.convert("RGB")

        return image

    def resize(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.image_short_side
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.image_short_side
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))

        return resized_img

    def normalize(self, img):
        img = img - np.array([123.68, 116.78, 103.94], dtype=np.float32)
        img /= 255.
        return img

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

        resized_img = self.resize(img)
        resized_img = self.normalize(resized_img)
        # convert hwc image to chw image
        resized_img = torch.from_numpy(resized_img).permute(2, 0, 1).float()

        result = {'image': resized_img, 'org_shape': [height, width]}

        _, new_height, new_width = resized_img.shape
        logger.info(f"image: {height} x {width}  -> {new_height} x {new_width} ")
        return result


class OCRDetectionPostProcessor(object):

    def __init__(self, config: DbNetConfig):
        """The base constructor for all ocr recognition preprocessors.

        """
        super().__init__()

        self.config = config
        self.thresh = config.thresh
        self.return_polygon = config.return_polygon

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pred = inputs['results'][0] if len(inputs['results'].shape) == 4 else inputs['results']
        pred = pred.float()
        height, width = inputs['org_shape']
        segmentation = pred > self.thresh
        if self.return_polygon:
            boxes, scores = polygons_from_bitmap(pred, segmentation, width,
                                                 height)
        else:
            boxes, scores = boxes_from_bitmap(pred, segmentation, width,
                                              height)
        result = {'det_polygons': np.array(boxes)}
        return result
