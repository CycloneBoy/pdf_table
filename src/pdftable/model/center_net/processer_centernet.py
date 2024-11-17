#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable
# @File    ：processer_centernet
# @Author  ：cycloneboy
# @Date    ：20xx/5/25 18:42
import io
import os
from typing import Union, Dict, Any

import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps
import requests
import torch

from .table_process import bbox_decode, gbox_decode, nms, bbox_post_process, \
    gbox_post_process, group_bbox_by_gbox

__all__ = [
    "OCRTableCenterNetPreProcessor",
    "OCRTableCenterNetPostProcessor"
]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


class OCRTableCenterNetPreProcessor(object):

    def __init__(self, config=None):
        self.config = config

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

    def get_affine_transform(self,
                             center,
                             scale,
                             rot,
                             output_size,
                             shift=np.array([0, 0], dtype=np.float32),
                             inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def process(self, img):
        mean = np.array([0.408, 0.447, 0.470],
                        dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.289, 0.274, 0.278],
                       dtype=np.float32).reshape(1, 1, 3)
        height, width = img.shape[0:2]
        inp_height, inp_width = 1024, 1024
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = self.get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(img, (width, height))
        inp_image = cv2.warpAffine(
            resized_image,
            trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height,
                                                      inp_width)
        images = torch.from_numpy(images)
        meta = {
            'c': c,
            's': s,
            'input_height': inp_height,
            'input_width': inp_width,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4
        }

        result = {'image': images, 'meta': meta}
        return result

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
            data_batch.append(data)
        return data_batch


class OCRTableCenterNetPostProcessor(object):

    def __init__(self, config=None):
        self.config = config
        self.K = 1000
        self.MK = 4000

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output = inputs['results'][0] if isinstance(inputs['results'], list) else inputs['results']
        meta = inputs['meta']
        hm = output['hm'].sigmoid_()
        v2c = output['v2c']
        c2v = output['c2v']
        reg = output['reg']
        bbox, _ = bbox_decode(hm[:, 0:1, :, :], c2v, reg=reg, K=self.K)
        gbox, _ = gbox_decode(hm[:, 1:2, :, :], v2c, reg=reg, K=self.MK)

        bbox = bbox.detach().cpu().numpy()
        gbox = gbox.detach().cpu().numpy()
        bbox = nms(bbox, 0.3)
        bbox = bbox_post_process(bbox.copy(), [meta['c']],
                                 [meta['s']], meta['out_height'],
                                 meta['out_width'])
        gbox = gbox_post_process(gbox.copy(), [meta['c']],
                                 [meta['s']], meta['out_height'],
                                 meta['out_width'])
        bbox = group_bbox_by_gbox(bbox[0], gbox[0])

        res = []
        for box in bbox:
            if box[8] > 0.3:
                res.append(box[0:8])

        # sort detection result with coord
        box_list = sorted(res,key=lambda x: 0.01 * sum(x[::2]) / 4 + sum(x[1::2]) / 4)

        result = {
            "polygons": np.array(box_list),
        }
        result.update(inputs)

        return result
