#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ：PdfTable 
# @File     : lgpma_preprocess.py
# @Author   : cycloneboy
# @Date     : 20xx/9/24 - 14:58
import collections
import functools
import random
import warnings
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from .base_utils import build_from_cfg, imread, is_list_of, \
    imrescale, imresize, imnormalize, impad, impad_to_multiple, DataContainer as DC, to_tensor, imfrombytes, bgr2gray, \
    gray2bgr
from .file_client import FileClient


class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


class DavarResize(Resize):
    """ Resize images & bbox & mask. Add new specialities of
        - support poly boxes resize
        - support cbboxes resize
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        """
        Args:
            img_scale (tuple or list[tuple]): Images scales for resizing.
            multiscale_mode (str): Either "range" or "value".
            ratio_range (tuple[float]): (min_ratio, max_ratio)
            keep_ratio (bool): Whether to keep the aspect ratio when resizing the
                image.
            bbox_clip_border (bool, optional): Whether clip the objects outside
                the border of the image. Defaults to True.
            backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
                These two backends generates slightly different results. Defaults
                to 'cv2'.
            override (bool, optional): Whether to override `scale` and
                `scale_factor` so as to call resize twice. Default False. If True,
                after the first resizing, the existed `scale` and `scale_factor`
                will be ignored so the second resizing can be allowed.
                This option is a work-around for multiple times of resize in DETR.
                Defaults to False.
        """
        super().__init__(img_scale=img_scale, multiscale_mode=multiscale_mode, ratio_range=ratio_range,
                         keep_ratio=keep_ratio, bbox_clip_border=bbox_clip_border, backend=backend,
                         override=override)

    def _resize_bboxes(self, results):
        """ Resize bboxes (support 'gt_bboxes', 'gt_poly_bboxes').
            Refactor this function to support multiple points

        Args:
            results (dict): input data flow

        Returns:
            dict: updated data flow. All keys in `bbox_fields` will be updated according to `scale_factor`.
        """
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = []
            for box in results[key]:
                tmp_box = np.array(box, dtype=np.float32)
                tmp_box[0::2] *= results['scale_factor'][0]
                tmp_box[1::2] *= results['scale_factor'][1]
                if self.bbox_clip_border:
                    tmp_box[0::2] = np.clip(tmp_box[0::2], 0, img_shape[1])
                    tmp_box[1::2] = np.clip(tmp_box[1::2], 0, img_shape[0])
                bboxes.append(tmp_box)
            if len(results[key]) > 0:
                results[key] = bboxes

    def _resize_cbboxes(self, results):
        """ Resize cbboxes (support 'gt_cbboxes').

        Args:
            results (dict): input data flow

        Returns:
            dict: updated data flow. All keys in `cbbox_fields` will be updated according to `scale_factor`.
        """
        img_shape = results['img_shape']
        for key in results.get('cbbox_fields', []):
            cbboxes = []
            for cbox in results[key]:
                tmp_cbox = np.array(cbox, dtype=np.float32)
                new_tmp_cbox = []
                for ccbox in tmp_cbox:
                    ccbox = np.array(ccbox, dtype=np.float32)
                    ccbox[0::2] *= results['scale_factor'][0]
                    ccbox[1::2] *= results['scale_factor'][1]
                    new_tmp_cbox.append(ccbox)
                tmp_cbox = np.array(new_tmp_cbox, dtype=np.float32)
                if self.bbox_clip_border:
                    tmp_cbox[:, 0::2] = np.clip(tmp_cbox[:, 0::2], 0, img_shape[1])
                    tmp_cbox[:, 1::2] = np.clip(tmp_cbox[:, 1::2], 0, img_shape[0])
                cbboxes.append(tmp_cbox)
            results[key] = cbboxes

    def __call__(self, results):
        """ Main process of davar_resize

        Args:
            results (dict): input data flow.

        Returns:
            dict: updated data flow.
        """
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple([int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, 'scale and scale_factor cannot be both set.'
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_cbboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)

        return results


class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = imnormalize(results[key], self.mean, self.std,
                                       self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = impad(
                results[key], shape=results['pad_shape'][:2])

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__


class DavarDefaultFormatBundle(DefaultFormatBundle):
    """ The common data format pipeline used by DavarCustom dataset. including, (1) transferred into Tensor
        (2) contained by DataContainer (3) put on device (GPU|CPU)

        - keys in ['img', 'gt_semantic_seg'] will be transferred into Tensor and put on GPU
        - keys in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore','gt_labels', 'stn_params']
          will be transferred into Tensor
        - keys in ['gt_masks', 'gt_poly_bboxes', 'gt_poly_bboxes_ignore', 'gt_cbboxes', 'gt_cbboxes_ignore',
                  'gt_texts', 'gt_text'] will be put on CPU
    """

    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'stn_params']:
            if key not in results:
                continue

            results[key] = DC(to_tensor(results[key]))

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        # Updated keys by DavarCustom dataset
        for key in ['gt_masks', 'gt_poly_bboxes', 'gt_poly_bboxes_ignore', 'gt_cbboxes',
                    'gt_cbboxes_ignore', 'gt_texts', 'gt_text', 'array_gt_texts', 'gt_bieo_labels']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)

        return results


class DavarLoadImageFromFile():
    """Loading image from file, add features of
       - load from nd.array
       - fix the bugs that orientation problem of cv2 reading.
    """

    def __init__(self,
                 decode_from_array=False,
                 to_float32=False):
        """ Initialization

        Args:
            decode_from_array (boolean): directly load image data from nd.array
            to_float32(boolean): transfer image data into float32
        """
        self.decode_from_array = decode_from_array
        self.to_float32 = to_float32

    def __call__(self, results):
        """ Main process

        Args:
            results(dict): Data flow used in DavarCustomDataset

        Returns:
            results(dict): Data flow used in DavarCustomDataset
        """
        if self.decode_from_array:
            data_array = results['img_info']
            assert isinstance(data_array, np.ndarray)
            data_list = [data_array[i] for i in range(data_array.size)]
            data_str = bytes(data_list)
            data_str = data_str.decode()
            data_list = data_str.split('&&')
            results['img_info'] = dict()
            results['img_info']['filename'] = data_list[0]
            results['img_info']['height'] = int(data_list[1])
            results['img_info']['width'] = int(data_list[2])

        if 'img_prefix' in results:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        elif 'img_info' in results:
            filename = results['img_info']['filename']
        else:
            filename = results['img']

        # Fix the problem of reading image reversely
        img = imread(filename, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)

        if not isinstance(img, np.ndarray):
            print("Reading Error at {}".format(filename))
            return None

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(self.to_float32)


class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=[(1333, 400), (1333, 800)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1333, 400), (1333, 400), (1333, 800), (1333, 800)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float | list[float] | None): Scale factors for resizing.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        assert (img_scale is None) ^ (scale_factor is None), (
            'Must have but only one variable can be setted')
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            self.scale_key = 'scale'
            assert is_list_of(self.img_scale, tuple)
        else:
            self.img_scale = scale_factor if isinstance(
                scale_factor, list) else [scale_factor]
            self.scale_key = 'scale_factor'

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.img_scale:
            for flip, direction in flip_args:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                _results['flip_direction'] = direction
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str


class DavarCollect():
    """ Collect specific data from the data flow (results)"""

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                            'img_norm_cfg')):
        """

        Args:
            keys(list[str]): keys that need to be collected
            meta_keys(tuple): keys of img_meta that need to be collected. e.g.,
                            - "img_shape": image shape, (h, w, c).
                            - "scale_factor": the scale factor of the re-sized image to the original image
                            - "flip": whether the image is flipped
                            - "filename": path to the image
                            - "ori_shape": original image shape
                            - "pad_shape": image shape after padding
                            - "img_norm_cfg": configuration of normalizations
        """
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """ Main process of davar_collect

        Args:
            results(dict): input data flow

        Returns:
            dict: collected data informations from original data flow
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        # Add feature to support situation without img_metas.
        if len(img_metas) != 0:
            data['img_metas'] = DC(img_metas, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)


class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

class LoadImageFromNdarray(LoadImageFromFile):
    """Load an image from np.ndarray.

    Similar with :obj:`LoadImageFromFile`, but the image read from
    ``results['img']``, which is np.ndarray.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert results['img'].dtype == 'uint8'

        img = results['img']
        if self.color_type == 'grayscale' and img.shape[2] == 3:
            img = bgr2gray(img, keepdim=True)
        if self.color_type == 'color' and img.shape[2] == 1:
            img = gray2bgr(img)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results



class TableResize:
    """Image resizing and padding for Table Recognition OCR, Table Structure Recognition.

    Args:
        height (int | tuple(int)): Image height after resizing.
        min_width (none | int | tuple(int)): Image minimum width
            after resizing.
        max_width (none | int | tuple(int)): Image maximum width
            after resizing.
        keep_aspect_ratio (bool): Keep image aspect ratio if True
            during resizing, Otherwise resize to the size height *
            max_width.
        img_pad_value (int): Scalar to fill padding area.
        width_downsample_ratio (float): Downsample ratio in horizontal
            direction from input image to output feature.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.
    """

    def __init__(self,
                 img_scale=None,
                 min_size=None,
                 ratio_range=None,
                 interpolation=None,
                 keep_ratio=True,
                 long_size=None):
        self.img_scale = img_scale
        self.min_size = min_size
        self.ratio_range = ratio_range
        self.interpolation = cv2.INTER_LINEAR
        self.long_size = long_size
        self.keep_ratio = keep_ratio

    def _get_resize_scale(self, w, h):
        if self.keep_ratio:
            if self.img_scale is None and isinstance(self.ratio_range, list):
                choice_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
                return (int(w * choice_ratio), int(h * choice_ratio))
            elif isinstance(self.img_scale, tuple) and -1 in self.img_scale:
                if self.img_scale[0] == -1:
                    resize_w = w / h * self.img_scale[1]
                    return (int(resize_w), self.img_scale[1])
                else:
                    resize_h = h / w * self.img_scale[0]
                    return (self.img_scale[0], int(resize_h))
            else:
                return (int(w), int(h))
        else:
            if isinstance(self.img_scale, tuple):
                return self.img_scale
            else:
                raise NotImplementedError

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        if 'img_info' in results.keys():
            # train and validate phase
            if results['img_info'].get('bbox', None) is not None:
                bboxes = results['img_info']['bbox']
                scale_factor = results['scale_factor']
                # bboxes[..., 0::2], bboxes[..., 1::2] = \
                #     bboxes[..., 0::2] * scale_factor[1], bboxes[..., 1::2] * scale_factor[0]
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2] * scale_factor[1], 0, img_shape[1] - 1)
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2] * scale_factor[0], 0, img_shape[0] - 1)
                results['img_info']['bbox'] = bboxes
            else:
                raise ValueError('results should have bbox keys.')
        else:
            # testing phase
            pass

    def _resize_img(self, results):
        img = results['img']
        h, w, _ = img.shape

        if self.min_size is not None:
            if w > h:
                w = self.min_size / h * w
                h = self.min_size
            else:
                h = self.min_size / w * h
                w = self.min_size

        if self.long_size is not None:
            if w < h:
                w = self.long_size / h * w
                h = self.long_size
            else:
                h = self.long_size / w * h
                w = self.long_size

        img_scale = self._get_resize_scale(w, h)
        resize_img = cv2.resize(img, img_scale, interpolation=self.interpolation)
        scale_factor = (resize_img.shape[0] / img.shape[0], resize_img.shape[1] / img.shape[1])

        results['img'] = resize_img
        results['img_shape'] = resize_img.shape
        results['pad_shape'] = resize_img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def __call__(self, results):
        self._resize_img(results)
        self._resize_bboxes(results)
        return results


class TablePad:
    """Pad the image & mask.
    Two padding modes:
    (1) pad to fixed size.
    (2) pad to the minium size that is divisible by some number.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=None,
                 keep_ratio=False,
                 return_mask=False,
                 mask_ratio=2,
                 train_state=True,
                 ):
        self.size = size[::-1]
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.keep_ratio = keep_ratio
        self.return_mask = return_mask
        self.mask_ratio = mask_ratio
        self.training = train_state
        # only one of size or size_divisor is valid.
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad(self, img, size, pad_val):
        if not isinstance(size, tuple):
            raise NotImplementedError

        if len(size) < len(img.shape):
            shape = size + (img.shape[-1],)
        else:
            shape = size

        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val

        h, w = img.shape[:2]
        size_w, size_h = size[:2]
        if h > size_h or w > size_w:
            if self.keep_ratio:
                if h / size_h > w / size_w:
                    size = (int(w / h * size_h), size_h)
                else:
                    size = (size_w, int(h / w * size_w))
            img = cv2.resize(img, size[::-1], cv2.INTER_LINEAR)
        pad[:img.shape[0], :img.shape[1], ...] = img
        if self.return_mask:
            mask = np.empty(size, dtype=img.dtype)
            mask[...] = 0
            mask[:img.shape[0], :img.shape[1]] = 1

            # mask_ratio is mean stride of backbone in (height, width)
            if isinstance(self.mask_ratio, int):
                mask = mask[::self.mask_ratio, ::self.mask_ratio]
            elif isinstance(self.mask_ratio, tuple):
                mask = mask[::self.mask_ratio[0], ::self.mask_ratio[1]]
            else:
                raise NotImplementedError

            mask = np.expand_dims(mask, axis=0)
        else:
            mask = None
        return pad, mask

    def _divisor(self, img, size_divisor, pad_val):
        pass

    def _pad_img(self, results):
        if self.size is not None:
            padded_img, mask = self._pad(results['img'], self.size, self.pad_val)
        elif self.size_divisor is not None:
            raise NotImplementedError
        results['img'] = padded_img
        results['mask'] = mask
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        # visual_img = visual_table_resized_bbox(results)
        # cv2.imwrite('/data_0/cache/{}_visual.jpg'.format(os.path.basename(results['filename']).split('.')[0]), visual_img)
        # if self.training:
        # scaleBbox(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


class ToTensorOCR:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor."""

    def __init__(self):
        pass

    def __call__(self, results):
        results['img'] = TF.to_tensor(results['img'].copy())

        return results


class NormalizeOCR:
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['img'] = TF.normalize(results['img'], self.mean, self.std)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
        return results


class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        # assert isinstance(transforms, collections.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


PIPELINES = {
    "DavarLoadImageFromFile": DavarLoadImageFromFile,
    "MultiScaleFlipAug": MultiScaleFlipAug,
    "DavarResize": DavarResize,
    "Normalize": Normalize,
    "Pad": Pad,
    "DavarDefaultFormatBundle": DavarDefaultFormatBundle,
    "DavarCollect": DavarCollect,
    "LoadImageFromFile": LoadImageFromFile,
    "TableResize": TableResize,
    "TablePad": TablePad,
    "ToTensorOCR": ToTensorOCR,
    "NormalizeOCR": NormalizeOCR,
    "Collect": Collect,
    "DefaultFormatBundle": DefaultFormatBundle,
    "LoadImageFromNdarray": LoadImageFromNdarray,
}
