#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：image_processing_pplcnet
# @Author  ：cycloneboy
# @Date    ：20xx/9/28 18:21
import os
import random
import time
import math
from functools import cmp_to_key, partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch import TensorType

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, logger, get_size_dict
from transformers.image_transforms import resize, get_resize_output_image_size, rescale, normalize, \
    to_channel_dimension_format
from transformers.image_utils import PILImageResampling, ImageInput, make_list_of_images, valid_images, to_numpy_array, \
    ChannelDimension

__all__ = [
    "PPLCNetImageProcessor",
    "PPLCNetImagePostProcessor",
]

from transformers.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

"""
pplcnet image processing
"""

CLASS_ID_MAP = {
    "image_orientation": {
        0: "0°",
        1: "90°",
        2: "180°",
        3: "270°",
    },
    "text_image_orientation": {
        0: "0",
        1: "90",
        2: "180",
        3: "270",
    },
    "textline_orientation": {
        0: "0_degree",
        1: "180_degree",
    },
    # 0 表示阿拉伯语（arabic）；1 表示中文繁体（chinese_cht）；2 表示斯拉夫语（cyrillic）；
    # 3 表示梵文（devanagari）；4 表示日语（japan）；5 表示卡纳达文（ka）；
    # 6 表示韩语（korean）；7 表示泰米尔文（ta）；8 表示泰卢固文（te）；9 表示拉丁语（latin）。
    "language_classification": {
        0: "arabic",
        1: "chinese_cht",
        2: "cyrillic",
        3: "devanagari",
        4: "japan",
        5: "ka",
        6: "korean",
        7: "ta",
        8: "te",
        9: "latin",
    }
}

IMAGE_SIZE_MAP = {
    "image_orientation": {
        "height": 224,
        "width": 224
    },
    "text_image_orientation": {
        "height": 224,
        "width": 224
    },
    "textline_orientation": {
        "height": 80,
        "width": 160
    },
    "language_classification": {
        "height": 80,
        "width": 160
    },
}

TOP_K_POST_PROCESS_MAP = {
    "image_orientation": {
        "topk": 1,
    },
    "text_image_orientation": {
        "topk": 2,
    },
    "textline_orientation": {
        "topk": 1,
    },
    "language_classification": {
        "topk": 2,
    },
}


class TableAttribute(object):
    def __init__(
            self,
            source_threshold=0.5,
            number_threshold=0.5,
            color_threshold=0.5,
            clarity_threshold=0.5,
            obstruction_threshold=0.5,
            angle_threshold=0.5, ):
        self.source_threshold = source_threshold
        self.number_threshold = number_threshold
        self.color_threshold = color_threshold
        self.clarity_threshold = clarity_threshold
        self.obstruction_threshold = obstruction_threshold
        self.angle_threshold = angle_threshold

    def __call__(self, batch_preds, file_names=None):
        # postprocess output of predictor
        batch_res = []

        for res in batch_preds:
            res = res.tolist()
            label_res = []
            source = 'Scanned' if res[0] > self.source_threshold else 'Photo'
            number = 'Little' if res[1] > self.number_threshold else 'Numerous'
            color = 'Black-and-White' if res[
                                             2] > self.color_threshold else 'Multicolor'
            clarity = 'Clear' if res[3] > self.clarity_threshold else 'Blurry'
            obstruction = 'Without-Obstacles' if res[
                                                     4] > self.number_threshold else 'With-Obstacles'
            angle = 'Horizontal' if res[
                                        5] > self.number_threshold else 'Tilted'

            label_res = [source, number, color, clarity, obstruction, angle]

            threshold_list = [
                self.source_threshold, self.number_threshold,
                self.color_threshold, self.clarity_threshold,
                self.obstruction_threshold, self.angle_threshold
            ]
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            batch_res.append({"attributes": label_res, "output": pred_res})
        return batch_res


class Topk(object):
    def __init__(self, topk=1, task_name=None, delimiter=None):
        assert isinstance(topk, (int,))
        self.topk = topk
        delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = CLASS_ID_MAP.get(task_name, {})

    def __call__(self, x, file_names=None):
        if isinstance(x, dict):
            x = x['logits']
        assert isinstance(x, torch.Tensor)
        if file_names is not None:
            assert x.shape[0] == len(file_names)
        x = F.softmax(x, dim=-1)
        x = x.detach().cpu().numpy()

        y = []
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = {
                "class_ids": clas_id_list,
                "scores": np.around(
                    score_list, decimals=5).tolist(),
            }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            if label_name_list is not None:
                result["label_names"] = label_name_list
            y.append(result)
        return y


class PPLCNetImageProcessor(BaseImageProcessor):
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
            do_pad: bool = False,
            task: str = "table_attribute",
            **kwargs,
    ) -> None:
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None if size is None else 224

        size = size if size is not None else IMAGE_SIZE_MAP.get(task, {"height": 224, "width": 224}, )
        size = get_size_dict(size, max_size=max_size, default_to_square=False)

        super().__init__(**kwargs)
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.task = task

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.resize
    def resize(
            self,
            image: np.ndarray,
            size: Dict[str, int],
            resample: PILImageResampling = PILImageResampling.BILINEAR,
            data_format: Optional[ChannelDimension] = None,
            **kwargs,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.
        """
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated and will be removed in v4.26. "
                "Please specify in `size['longest_edge'] instead`.",
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(image, size["shortest_edge"], size["longest_edge"])
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        image = resize(image, size=size, resample=resample, data_format=data_format)
        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale
    def rescale(
            self, image: np.ndarray, rescale_factor: Union[float, int], data_format: Optional[ChannelDimension] = None
    ) -> np.ndarray:
        """
        Rescale the image by the given factor.
        """
        return rescale(image, rescale_factor, data_format=data_format)

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.normalize
    def normalize(
            self,
            image: np.ndarray,
            mean: Union[float, Iterable[float]],
            std: Union[float, Iterable[float]],
            data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """
        Normalize the image with the given mean and standard deviation.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format)

    def preprocess(
            self,
            images: ImageInput,
            do_resize: Optional[bool] = None,
            size: Optional[Dict[str, int]] = None,
            resample=None,  # PILImageResampling
            do_rescale: Optional[bool] = None,
            rescale_factor: Optional[Union[int, float]] = None,
            do_normalize: Optional[bool] = None,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            do_pad: Optional[bool] = None,
            return_tensors: Optional[Union[TensorType, str]] = "pt",
            data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
            **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess.
            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
                List of annotations associated with the image or batch of images. If annotionation is for object
                detection, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a
                  dictionary. An image can have no annotations, in which case the list should be empty.
                If annotionation is for segmentation, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.
                  An image can have no segments, in which case the list should be empty.
                - "file_name" (`str`): The file name of the image.
            return_segmentation_masks (`bool`, *optional*, defaults to self.return_segmentation_masks):
                Whether to return segmentation masks.
            masks_path (`str` or `pathlib.Path`, *optional*):
                Path to the directory containing the segmentation masks.
            do_resize (`bool`, *optional*, defaults to self.do_resize):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to self.size):
                Size of the image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to self.resample):
                Resampling filter to use when resizing the image.
            do_rescale (`bool`, *optional*, defaults to self.do_rescale):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*, defaults to self.do_normalize):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to self.image_mean):
                Mean to use when normalizing the image.
            image_std (`float` or `List[float]`, *optional*, defaults to self.image_std):
                Standard deviation to use when normalizing the image.
            do_pad (`bool`, *optional*, defaults to self.do_pad):
                Whether to pad the image.
            format (`str` or `AnnotionFormat`, *optional*, defaults to self.format):
                Format of the annotations.
            return_tensors (`str` or `TensorType`, *optional*, defaults to self.return_tensors):
                Type of tensors to return. If `None`, will return the list of images.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to self.data_format):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        if isinstance(images, str):
            images = [images]

        if isinstance(images, list) and isinstance(images[0], str):
            old_images = []
            for image in images:
                img = cv2.imread(image)
                # to rgb
                img = img[:, :, ::-1]
                old_images.append(img)
            images = old_images

        max_size = None
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` argument is deprecated and will be removed in a future version, use"
                " `size['longest_edge']` instead.",
            )
            size = kwargs.pop("max_size")

        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size=size, max_size=max_size, default_to_square=False)
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_pad = self.do_pad if do_pad is None else do_pad

        if do_resize is not None and size is None:
            raise ValueError("Size and max_size must be specified if do_resize is True.")

        if do_rescale is not None and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize is not None and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        # transformations
        if do_resize:
            images = [self.resize(image, size=size, resample=resample) for image in images]

        if do_rescale:
            images = [self.rescale(image, rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image, image_mean, image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]
        data = {"pixel_values": images}

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process(self, outputs):
        """
        后处理操作

        :param outputs:
        :return:
        """

        if self.task == "table_attribute":
            process = TableAttribute()
        else:
            topk = TOP_K_POST_PROCESS_MAP[self.task]["topk"]
            process = Topk(topk=topk, task_name=self.task, )

        res = process(outputs)

        return res


class PPLCNetImagePostProcessor(object):

    def __init__(self, task: str = "table_attribute", ):
        self.task = task
        if self.task == "table_attribute":
            self.process = TableAttribute()
        else:
            topk = TOP_K_POST_PROCESS_MAP[self.task]["topk"]
            self.process = Topk(topk=topk, task_name=self.task, )

    def __call__(self, inputs):
        if 'results' in inputs:
            inputs = inputs['results']
        res = self.process(inputs)

        return res
