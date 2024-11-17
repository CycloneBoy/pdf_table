#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author  ：cycloneboy
# @Date  : 2023/4/4 - 23:03
import io
import os

import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps
import requests
import torch

__all__ = [
    "OCRConvNextViTPreprocessor"
]


class OCRConvNextViTPreprocessor(object):

    def __init__(self, ):
        """The base constructor for all ocr recognition preprocessors.

        Args:
            model_dir (str): model directory to initialize some resource
            mode: The mode for the preprocessor.
        """

        self.do_chunking = True
        self.target_height = 32
        self.target_width = 804
        if self.do_chunking:
            self.target_width = 804

    def keepratio_resize(self, img):
        cur_ratio = img.shape[1] / float(img.shape[0])
        mask_height = self.target_height
        mask_width = self.target_width
        if cur_ratio > float(self.target_width) / self.target_height:
            cur_target_height = self.target_height
            cur_target_width = self.target_width
        else:
            cur_target_height = self.target_height
            cur_target_width = int(self.target_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
        mask[:img.shape[0], :img.shape[1]] = img
        img = mask
        return img

    def read_image(self, image_path_or_url):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        image = image.convert("RGB")

        return image

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
            if isinstance(item, str):

                img = np.array(self.read_image(item).convert('L'))
            elif isinstance(item, PIL.Image.Image):
                img = np.array(item.convert('L'))
            elif isinstance(item, np.ndarray):
                if len(item.shape) == 3:
                    img = cv2.cvtColor(item, cv2.COLOR_RGB2GRAY)
            else:
                raise TypeError(
                    f'inputs should be either (a list of) str, PIL.Image, np.array, but got {type(item)}'
                )

            img = self.keepratio_resize(img)
            img = torch.FloatTensor(img)
            if self.do_chunking:
                chunk_img = []
                for i in range(3):
                    left = (300 - 48) * i
                    chunk_img.append(img[:, left:left + 300])
                merge_img = torch.cat(chunk_img, 0)
                data = merge_img.view(3, 1, self.target_height, 300) / 255.
            else:
                data = img.view(1, 1, self.target_height,
                                self.target_width) / 255.
            data_batch.append(data)
        data_batch = torch.cat(data_batch, 0)
        return data_batch
