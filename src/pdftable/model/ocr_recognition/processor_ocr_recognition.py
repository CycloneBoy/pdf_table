#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：processor_ocr_recognition
# @Author  ：cycloneboy
# @Date    ：20xx/7/14 13:22


import io
import os
from typing import Dict, Any

import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps
import requests
import torch

import torch.nn.functional as F

from .configuration_ocr_recognition import OCRRecognitionConfig

__all__ = [
    "OCRRecognitionPreprocessor",
    "OCRRecognitionPostProcessor",
]


class OCRRecognitionPreprocessor(object):

    def __init__(self, config: OCRRecognitionConfig, ):
        """The base constructor for all ocr recognition preprocessors.

        Args:
            model_dir (str): model directory to initialize some resource
            mode: The mode for the preprocessor.
        """

        self.do_chunking = config.do_chunking
        self.target_height = config.img_height
        self.target_width = config.img_width

    def keepratio_resize(self, img):
        height, width, _ = img.shape

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
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[:img.shape[0], :img.shape[1], :] = img
        img = mask

        # logger.info(f"img: {height} x {width}  -> {mask_height} x {mask_width} ")
        return img

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
                - An image loaded in PIL or opencv directly
        Returns:
            outputs: the preprocessed image
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        data_batch = []
        for item in inputs:
            if isinstance(item, str):
                img = np.array(self.read_image(item))
            elif isinstance(item, PIL.Image.Image):
                img = np.array(item.convert('RGB'))
            elif isinstance(item, np.ndarray):
                if len(item.shape) == 2:
                    img = cv2.cvtColor(item, cv2.COLOR_GRAY2RGB)
                else:
                    img = item
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
                data = merge_img.view(3, self.target_height, 300, 3) / 255.
            else:
                data = img.view(1, self.target_height, self.target_width, 3) / 255.
            data = data.permute(0, 3, 1, 2)
            data_batch.append(data)
        data_batch = torch.cat(data_batch, 0)
        return {'image': data_batch}


class OCRRecognitionPostProcessor(object):

    def __init__(self, config: OCRRecognitionConfig):
        """The base constructor for all ocr recognition preprocessors.

        """
        super().__init__()

        self.config = config
        self.vocab_file = config.vocab_file
        self.label_mapping = dict()
        self.char_mapping = dict()
        self.load_vocab()

    def load_vocab(self):
        dict_path = os.path.join(self.config.model_path, self.vocab_file)

        with open(dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            cnt = 1
            # ConvNextViT model start from index=2
            if self.config.do_chunking:
                cnt += 1
            for line in lines:
                line = line.strip('\n')
                self.label_mapping[cnt] = line
                self.char_mapping[line] = cnt
                cnt += 1

    def __call__(self, inputs) -> Dict[str, Any]:
        outprobs = inputs
        outprobs = F.softmax(outprobs, dim=-1)
        preds = torch.argmax(outprobs, -1)
        batchSize, length = preds.shape
        final_str_list = []
        for i in range(batchSize):
            pred_idx = preds[i].cpu().data.tolist()
            last_p = 0
            str_pred = []
            for p in pred_idx:
                if p != last_p and p != 0:
                    str_pred.append(self.label_mapping[p])
                last_p = p
            final_str = ''.join(str_pred)
            final_str_list.append(final_str)
        outputs = {'preds': final_str_list, 'probs': inputs}
        return outputs
