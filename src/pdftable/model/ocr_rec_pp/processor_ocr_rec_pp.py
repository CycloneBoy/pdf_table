#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：processor_ocr_rec_pp
# @Author  ：cycloneboy
# @Date    ：20xx/8/31 17:07
from typing import Dict, Any

import cv2
import math
import numpy as np
import PIL
from PIL import Image
import requests

from .rec_postprocess import CTCLabelDecode
from .configuration_ocr_recognition_pp import PPOcrRecognitionConfig

__all__ = [
    "PPOcrRecPreProcessor",
    "PPOcrRecPostProcessor",
]


class PPOcrRecPreProcessor(object):

    def __init__(self, config: PPOcrRecognitionConfig, ):
        self.config = config
        self.rec_image_shape = config.rec_image_shape
        self.rec_batch_num = config.rec_batch_num
        self.limited_max_width = config.limited_max_width
        self.limited_min_width = config.limited_min_width

    def read_image(self, image_path_or_url):
        image_content = image_path_or_url
        if image_path_or_url.startswith("http"):
            image_content = requests.get(image_path_or_url, stream=True).raw
        image = Image.open(image_content)
        image = image.convert("RGB")

        return image

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int((imgH * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)

        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(ratio_imgH)

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

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

        img_num = len(inputs)

        # Calculate the aspect ratio of all text bars
        width_list = []
        img_list = []
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

            width_list.append(img.shape[1] / float(img.shape[0]))
            img_list.append(img)

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        batch_num = self.rec_batch_num
        data_batch = []
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                current_img = img_list[indices[ino]]
                h, w = current_img.shape[0:2]

                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)

            for ino in range(beg_img_no, end_img_no):
                current_img = img_list[indices[ino]]
                norm_img = self.resize_norm_img(current_img, max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)

            batch = {
                'image': norm_img_batch,
                "indices": indices,
                "batch_beg_img_no": beg_img_no
            }
            data_batch.append(batch)

        return data_batch


class PPOcrRecPostProcessor(object):

    def __init__(self, config: PPOcrRecognitionConfig):
        super().__init__()

        self.config = config
        self.vocab_file = config.get_vocab_file()

        self.postprocess_op = CTCLabelDecode(character_dict_path=self.vocab_file,
                                             use_space_char=True)

    def __call__(self, inputs) -> Dict[str, Any]:
        pred = inputs['results']

        for rno in range(len(pred)):
            current_pred = pred[rno]
            prob_out = current_pred["results"]
            rec_result = self.postprocess_op(prob_out)
            indices = current_pred['indices']
            batch_beg_img_no = current_pred['batch_beg_img_no']
            rec_res = [['', 0.0]] * len(indices)
            beg_img_no = batch_beg_img_no
            rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        final_str_list = []
        probs_list = []
        outputs = []
        for item in rec_res:
            final_str_list.append(item[0])
            probs_list.append(item[1])
            outputs.append({'preds': item[0], 'probs': item[1]})

        # outputs = {'preds': final_str_list, 'probs': probs_list}
        return outputs
