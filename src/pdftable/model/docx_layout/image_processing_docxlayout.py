#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：image_processing_docxlayout
# @Author  ：cycloneboy
# @Date    ：20xx/9/27 15:50
import os
import time
import math
from functools import cmp_to_key
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from torch import TensorType

from transformers.image_processing_utils import BaseImageProcessor, logger
from transformers.image_utils import PILImageResampling, ImageInput

from pdftable.model.docx_layout import DocXLayoutConfig
from .processor_utils import get_affine_transform, ctdet_4ps_decode, ctdet_cls_decode, pnms, ctdet_4ps_post_process, \
    sort_pts, pts_intersection_rate

"""
layout image processing
"""

__all__ = [
    "DocXLayoutPreProcessor",
    "DocXLayoutImagePostProcessor"
]


class DocXLayoutPreProcessor(BaseImageProcessor):
    r"""
    Constructs a Detr image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's (height, width) dimensions after resizing. Can be overridden by the `size` parameter in
            the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image to the largest image in a batch and create a pixel mask. Can be
            overridden by the `do_pad` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
            self,
            do_resize: bool = True,
            size: Dict[str, int] = None,
            resample: PILImageResampling = PILImageResampling.BILINEAR,
            do_rescale: bool = True,
            rescale_factor: Union[int, float] = 1 / 255,
            do_normalize: bool = True,
            image_mean: Union[float, List[float]] = None,
            image_std: Union[float, List[float]] = None,
            do_pad: bool = True,
            **kwargs,
    ) -> None:
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None if size is None else 1333

        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        # size = get_size_dict(size, max_size=max_size, default_to_square=False)

        super().__init__(**kwargs)
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.40789655, 0.44719303, 0.47026116]
        self.image_std = image_std if image_std is not None else [0.2886383, 0.27408165, 0.27809834]
        self.do_pad = do_pad

        self.fix_res = True
        self.flip_test = False
        self.input_h = 768
        self.input_w = 768
        self.pad = 0
        self.down_ratio = 4
        self.image_mean = np.array(self.image_mean, dtype=np.float32).reshape(1, 1, 3)
        self.image_std = np.array(self.image_std, dtype=np.float32).reshape(1, 1, 3)
        self.image_name = None

    def do_preprocess(self, image, scale=1.0) -> Dict:
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.fix_res:
            inp_height, inp_width = self.input_h, self.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.pad)  # + 1
            inp_width = (new_width | self.pad)  # + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        vis_image = inp_image

        inp_image = ((inp_image / 255. - self.image_mean) / self.image_std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'input_height': inp_height,
                'input_width': inp_width,
                'vis_image': vis_image,
                'out_height': inp_height // self.down_ratio,
                'out_width': inp_width // self.down_ratio}

        data = {
            'image_file': self.image_name if self.image_name is not None else "",
            "image": images,
            "meta": meta
        }

        return data

    def preprocess(self,
                   images: Union[ImageInput, str, List[str]],
                   return_tensors: Optional[Union[TensorType, str]] = None,
                   **kwargs):
        if isinstance(images, str):
            self.image_name = images
            images = cv2.imread(images)
        elif isinstance(images, list) and isinstance(images[0], str):
            self.image_name = images
            images = [cv2.imread(image) for image in images]

        is_list = True
        if not isinstance(images, list):
            images = [images]
            is_list = False

        data = [self.do_preprocess(image) for image in images]

        # pixel_values = {"pixel_values": [item["pixel_values"] for item in data]}
        # meta = {"meta": }
        # encoded_inputs = BatchFeature(data=pixel_values, tensor_type=return_tensors)
        # encoded_inputs["meta"] = [item["meta"] for item in data]
        if not is_list:
            return data[0]
        return data


class DocXLayoutImagePostProcessor(object):

    def __init__(self, config: DocXLayoutConfig):
        """The base constructor for all ocr layout preprocessors.

        """
        super().__init__()
        self.config = config

        self.top_k = config.top_k
        self.scores_thresh = config.scores_thresh
        self.num_classes = config.num_classes
        self.nms = config.use_nms
        self.max_per_image = self.top_k
        self.debug = True

    def __call__(self, outputs, **kwargs):
        """Postprocess an image or a batch of images."""
        if "meta" not in kwargs:
            meta = outputs.pop("meta")
        else:
            meta = kwargs["meta"]
        results = self.post_process(outputs, meta)

        results_dict = {
            "bboxs": results,
            "boxes_num": len(results)
        }
        return results_dict

    def post_process(self, outputs: Dict, meta: Dict):
        """
        后处理操作

        :param outputs:
        :param meta:
        :return:
        """
        start_time = time.time()
        outputs, dets, dets_sub, corner, forward_time = self.post_process_model(outputs=outputs)
        dets, dets_sub = self.post_process_bbox(dets, dets_sub, corner, meta)

        results = self.merge_outputs([dets])
        layout_detection_info, subfield_detection_info = self.convert_eval_format(results)
        tot_time = time.time() - start_time

        res = {
            'results': results,
            'tot': tot_time,
            'output': outputs,
            "layout_dets": layout_detection_info,
            "subfield_dets": subfield_detection_info,
        }

        layout_res = self.wrap_result(res, category_map=self.config.id2label)
        res["layout_res"] = layout_res

        # 统一输出格式
        new_layout_res = []
        for item in layout_res['layouts']:
            point = item['pts']
            bbox = [point[0], point[1], point[4], point[5]]
            new_item = {
                "bbox": np.array(bbox),
                "label": item['category'],
                "score": item['confidence'],
                "category_id": self.config.label2id[item['category']],
            }
            new_layout_res.append(new_item)

        return new_layout_res

    def post_process_model(self, outputs: Dict, ):
        """
        模型推理后处理

        :param outputs:
        :return:
        """
        hm = outputs['hm'].sigmoid_()
        cls = outputs['cls'].sigmoid_()
        ftype = outputs['ftype'].sigmoid_()

        # add sub
        hm_sub = outputs['hm_sub'].sigmoid_()

        wh = outputs['wh']
        reg = outputs['reg']

        wh_sub = outputs['wh_sub']
        reg_sub = outputs['reg_sub']

        forward_time = time.time()

        # return dets [bboxes, scores, clses]
        dets, inds = ctdet_4ps_decode(hm, wh, reg=reg, K=self.top_k)

        # add sub
        dets_sub, inds_sub = ctdet_4ps_decode(hm_sub, wh_sub, reg=reg_sub, K=self.top_k)

        box_cls = ctdet_cls_decode(cls, inds)
        box_ftype = ctdet_cls_decode(ftype, inds)
        clses = torch.argmax(box_cls, dim=2, keepdim=True)
        ftypes = torch.argmax(box_ftype, dim=2, keepdim=True)
        dets = np.concatenate(
            (dets.detach().cpu().numpy(), clses.detach().cpu().numpy(), ftypes.detach().cpu().numpy()), axis=2)
        dets = np.array(dets)

        # add subfield
        dets_sub = np.concatenate(
            (dets_sub.detach().cpu().numpy(), clses.detach().cpu().numpy(), ftypes.detach().cpu().numpy()), axis=2)
        dets_sub = np.array(dets_sub)
        dets_sub[:, :, -3] += 11

        corner = 0
        return outputs, dets, dets_sub, corner, forward_time

    def post_process_bbox(self, dets, dets_sub, corner, meta, scale=1):
        dets, corner = self.post_process_dets(dets, corner, meta, scale)
        for j in range(1, self.num_classes + 1):
            dets[j] = self.duplicate_removal(dets[j])

        # add sub
        dets_sub, corner = self.post_process_dets(dets_sub, corner, meta, scale)
        for j in range(1, self.num_classes + 1):
            dets_sub[j] = self.duplicate_removal(dets_sub[j])

        dets[12] = dets_sub[12]
        dets[13] = dets_sub[13]

        return dets, dets_sub

    def post_process_dets(self, dets, corner, meta, scale=1):
        if self.nms:
            detn = pnms(dets[0], self.scores_thresh)
            if detn.shape[0] > 0:
                dets = detn.reshape(1, -1, detn.shape[1])
        k = dets.shape[2] if dets.shape[1] != 0 else 0
        if dets.shape[1] != 0:
            dets = dets.reshape(1, -1, dets.shape[2])
            # return dets is list and what in dets is dict. key of dict is classes, value of dict is [bbox,score]
            dets = ctdet_4ps_post_process(
                dets.copy(), [meta['c']], [meta['s']],
                meta['out_height'], meta['out_width'], self.num_classes)
            for j in range(1, self.num_classes + 1):
                dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, k)
                dets[0][j][:, :8] /= scale
        else:
            ret = {}
            dets = []
            for j in range(1, self.num_classes + 1):
                ret[j] = np.array([0] * k, dtype=np.float32)  # .reshape(-1, k)
            dets.append(ret)
        return dets[0], corner

    def duplicate_removal(self, results):
        bbox = []
        for box in results:
            if box[8] > self.scores_thresh:
                bbox.append(box)
        if len(bbox) > 0:
            return np.array(bbox)
        else:
            return np.array([[0] * 12])

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            # if len(self.scales) > 1 or self.opt.nms:
            #  results[j] = pnms(results[j],self.opt.nms_thresh)
        shape_num = 0
        for j in range(1, self.num_classes + 1):
            shape_num = shape_num + len(results[j])
        if shape_num != 0:
            # print(np.array(results[1]))
            scores = np.hstack(
                [results[j][:, 8] for j in range(1, self.num_classes + 1)])
        else:
            scores = []
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 8] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def convert_eval_format(self, all_bboxes):
        layout_detection_items = []
        subfield_detection_items = []
        for cls_ind in all_bboxes:
            for box in all_bboxes[cls_ind]:
                if box[8] < self.scores_thresh:
                    continue
                pts = np.round(box).tolist()[:8]
                score = box[8]
                category_id = box[9]
                direction_id = box[10]
                secondary_id = box[11]
                detection = {
                    "category_id": int(category_id),
                    # "secondary_id": int(secondary_id),
                    # "direction_id": int(direction_id),
                    "poly": pts,
                    "score": float("{:.2f}".format(score))
                }
                if cls_ind in (12, 13):
                    subfield_detection_items.append(detection)
                else:
                    layout_detection_items.append(detection)
        return layout_detection_items, subfield_detection_items

    def wrap_result(self, result, category_map):
        layout_detection_info = result["layout_dets"]
        subfield_detection_info = result["subfield_dets"]

        info = {'subfields': []}
        for itm in subfield_detection_info:
            subfield = {'category': category_map[itm['category_id']], 'pts': itm['poly'], 'confidence': itm['score'],
                        'layouts': []}
            info['subfields'].append(subfield)
        sort_pts(info['subfields'])

        if len(info['subfields']) > 0:
            other_subfield = {'category': 'other', 'pts': [0, 0, 0, 0, 0, 0, 0, 0], 'confidence': 0, 'layouts': []}
            for itm in layout_detection_info:
                layout = {'category': category_map[itm['category_id']], 'pts': itm['poly'], 'confidence': itm['score']}
                best_rate, best_idx = 0.0, -1
                for k in range(len(info['subfields'])):
                    inter_rate = pts_intersection_rate(layout['pts'], info['subfields'][k]['pts'])
                    if inter_rate > best_rate:
                        best_rate = inter_rate
                        best_idx = k
                if best_idx >= 0 and best_rate > 0.1:
                    info['subfields'][best_idx]['layouts'].append(layout)
                else:
                    other_subfield['layouts'].append(layout)
            if len(other_subfield['layouts']) > 0:
                info['subfields'].append(other_subfield)
        else:
            subfield = {'category': 'other', 'pts': [0, 0, 0, 0, 0, 0, 0, 0], 'confidence': 0, 'layouts': []}
            info['subfields'].append(subfield)
            for itm in layout_detection_info:
                layout = {'category': category_map[itm['category_id']], 'pts': itm['poly'], 'confidence': itm['score']}
                info['subfields'][0]['layouts'].append(layout)

        for subfield in info['subfields']:
            sort_pts(subfield['layouts'])

        new_subfields = []
        for subfield in info['subfields']:
            if subfield['category'] != 'other':
                new_subfields.append(subfield)
            else:
                for layout in subfield['layouts']:
                    layout_subfield = {'category': layout['category'], 'pts': layout['pts'],
                                       'confidence': layout['confidence'], 'layouts': [layout]}
                    new_subfields.append(layout_subfield)
        sort_pts(new_subfields)
        info['layouts'] = []
        for subfield in new_subfields:
            for layout in subfield['layouts']:
                info['layouts'].append(layout)

        return info
