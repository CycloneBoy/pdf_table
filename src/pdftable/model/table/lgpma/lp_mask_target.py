#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ：PdfTable 
# @File     : lp_mask_target.py
# @Author   : cycloneboy
# @Date     : 20xx/9/23 - 21:34
from math import ceil
import numpy as np
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_utils import rescale_size, imrescale, imresize, imflip, impad, imshear, imrotate, imtranslate


class BaseInstanceMasks(metaclass=ABCMeta):
    """Base class for instance masks."""

    @abstractmethod
    def rescale(self, scale, interpolation='nearest'):
        """Rescale masks as large as possible while keeping the aspect ratio.
        For details can refer to `mmcv.imrescale`.

        Args:
            scale (tuple[int]): The maximum size (h, w) of rescaled mask.
            interpolation (str): Same as :func:`mmcv.imrescale`.

        Returns:
            BaseInstanceMasks: The rescaled masks.
        """

    @abstractmethod
    def resize(self, out_shape, interpolation='nearest'):
        """Resize masks to the given out_shape.

        Args:
            out_shape: Target (h, w) of resized mask.
            interpolation (str): See :func:`mmcv.imresize`.

        Returns:
            BaseInstanceMasks: The resized masks.
        """

    @abstractmethod
    def flip(self, flip_direction='horizontal'):
        """Flip masks alone the given direction.

        Args:
            flip_direction (str): Either 'horizontal' or 'vertical'.

        Returns:
            BaseInstanceMasks: The flipped masks.
        """

    @abstractmethod
    def pad(self, out_shape, pad_val):
        """Pad masks to the given size of (h, w).

        Args:
            out_shape (tuple[int]): Target (h, w) of padded mask.
            pad_val (int): The padded value.

        Returns:
            BaseInstanceMasks: The padded masks.
        """

    @abstractmethod
    def crop(self, bbox):
        """Crop each mask by the given bbox.

        Args:
            bbox (ndarray): Bbox in format [x1, y1, x2, y2], shape (4, ).

        Return:
            BaseInstanceMasks: The cropped masks.
        """

    @abstractmethod
    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        device,
                        interpolation='bilinear'):
        """Crop and resize masks by the given bboxes.

        This function is mainly used in mask targets computation.
        It firstly align mask to bboxes by assigned_inds, then crop mask by the
        assigned bbox and resize to the size of (mask_h, mask_w)

        Args:
            bboxes (Tensor): Bboxes in format [x1, y1, x2, y2], shape (N, 4)
            out_shape (tuple[int]): Target (h, w) of resized mask
            inds (ndarray): Indexes to assign masks to each bbox,
                shape (N,) and values should be between [0, num_masks - 1].
            device (str): Device of bboxes
            interpolation (str): See `mmcv.imresize`

        Return:
            BaseInstanceMasks: the cropped and resized masks.
        """

    @abstractmethod
    def expand(self, expanded_h, expanded_w, top, left):
        """see :class:`Expand`."""

    @property
    @abstractmethod
    def areas(self):
        """ndarray: areas of each instance."""

    @abstractmethod
    def to_ndarray(self):
        """Convert masks to the format of ndarray.

        Return:
            ndarray: Converted masks in the format of ndarray.
        """

    @abstractmethod
    def to_tensor(self, dtype, device):
        """Convert masks to the format of Tensor.

        Args:
            dtype (str): Dtype of converted mask.
            device (torch.device): Device of converted masks.

        Returns:
            Tensor: Converted masks in the format of Tensor.
        """

    @abstractmethod
    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  fill_val=0,
                  interpolation='bilinear'):
        """Translate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            fill_val (int | float): Border value. Default 0.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            Translated masks.
        """

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        """Shear the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border. Default 0.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            ndarray: Sheared masks.
        """

    @abstractmethod
    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            fill_val (int | float): Border value. Default 0 for masks.

        Returns:
            Rotated masks.
        """


class BitmapMasks(BaseInstanceMasks):
    """This class represents masks in the form of bitmaps.

    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.core.mask.structures import *  # NOQA
        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(np.int)
        >>> self = BitmapMasks(masks, height=H, width=W)

        >>> # demo crop_and_resize
        >>> num_boxes = 5
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (14, 14)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> interpolation = 'bilinear'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation='nearest'):
        """See :func:`BaseInstanceMasks.rescale`."""
        if len(self.masks) == 0:
            new_w, new_h = rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack([
                imrescale(mask, scale, interpolation=interpolation)
                for mask in self.masks
            ])
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)

    def resize(self, out_shape, interpolation='nearest'):
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack([
                imresize(
                    mask, out_shape[::-1], interpolation=interpolation)
                for mask in self.masks
            ])
        return BitmapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical', 'diagonal')

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack([
                imflip(mask, direction=flip_direction)
                for mask in self.masks
            ])
        return BitmapMasks(flipped_masks, self.height, self.width)

    def pad(self, out_shape, pad_val=0):
        """See :func:`BaseInstanceMasks.pad`."""
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack([
                impad(mask, shape=out_shape, pad_val=pad_val)
                for mask in self.masks
            ])
        return BitmapMasks(padded_masks, *out_shape)

    def crop(self, bbox):
        """See :func:`BaseInstanceMasks.crop`."""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1:y1 + h, x1:x1 + w]
        return BitmapMasks(cropped_masks, h, w)

    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        device='cpu',
                        interpolation='bilinear'):
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(
            num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = torch.from_numpy(self.masks).to(device).index_select(
                0, inds).to(dtype=rois.dtype)
            targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape,
                                1.0, 0, 'avg', True).squeeze(1)
            resized_masks = (targets >= 0.5).cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    def expand(self, expanded_h, expanded_w, top, left):
        """See :func:`BaseInstanceMasks.expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w),
                                     dtype=np.uint8)
        else:
            expanded_mask = np.zeros((len(self), expanded_h, expanded_w),
                                     dtype=np.uint8)
            expanded_mask[:, top:top + self.height,
            left:left + self.width] = self.masks
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)

    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  fill_val=0,
                  interpolation='bilinear'):
        """Translate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            fill_val (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            BitmapMasks: Translated BitmapMasks.

        Example:
            >>> from mmdet.core.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random(dtype=np.uint8)
            >>> out_shape = (32, 32)
            >>> offset = 4
            >>> direction = 'horizontal'
            >>> fill_val = 0
            >>> interpolation = 'bilinear'
            >>> # Note, There seem to be issues when:
            >>> # * out_shape is different than self's shape
            >>> # * the mask dtype is not supported by cv2.AffineWarp
            >>> new = self.translate(out_shape, offset, direction, fill_val,
            >>>                      interpolation)
            >>> assert len(new) == len(self)
            >>> assert new.height, new.width == out_shape
        """
        if len(self.masks) == 0:
            translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            translated_masks = imtranslate(
                self.masks.transpose((1, 2, 0)),
                offset,
                direction,
                border_value=fill_val,
                interpolation=interpolation)
            if translated_masks.ndim == 2:
                translated_masks = translated_masks[:, :, None]
            translated_masks = translated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(translated_masks, *out_shape)

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        """Shear the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            BitmapMasks: The sheared masks.
        """
        if len(self.masks) == 0:
            sheared_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            sheared_masks = imshear(
                self.masks.transpose((1, 2, 0)),
                magnitude,
                direction,
                border_value=border_value,
                interpolation=interpolation)
            if sheared_masks.ndim == 2:
                sheared_masks = sheared_masks[:, :, None]
            sheared_masks = sheared_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(sheared_masks, *out_shape)

    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            fill_val (int | float): Border value. Default 0 for masks.

        Returns:
            BitmapMasks: Rotated BitmapMasks.
        """
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = imrotate(
                self.masks.transpose((1, 2, 0)),
                angle,
                center=center,
                scale=scale,
                border_value=fill_val)
            if rotated_masks.ndim == 2:
                # case when only one mask, (h, w)
                rotated_masks = rotated_masks[:, :, None]  # (h, w, 1)
            rotated_masks = rotated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(rotated_masks, *out_shape)

    @property
    def areas(self):
        """See :py:attr:`BaseInstanceMasks.areas`."""
        return self.masks.sum((1, 2))

    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        return torch.tensor(self.masks, dtype=dtype, device=device)

    @classmethod
    def random(cls,
               num_masks=3,
               height=32,
               width=32,
               dtype=np.uint8,
               rng=None):
        """Generate random bitmap masks for demo / testing purposes.

        Example:
            >>> from mmdet.core.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random()
            >>> print('self = {}'.format(self))
            self = BitmapMasks(num_masks=3, height=32, width=32)
        """
        from mmdet.utils.util_random import ensure_rng
        rng = ensure_rng(rng)
        masks = (rng.rand(num_masks, height, width) > 0.1).astype(dtype)
        self = cls(masks, height=height, width=width)
        return self


class BitmapMasksTable(BitmapMasks):
    """Inherited from BitmapMasks. Modify the data type of mask to store pyramid mask
    """

    def __init__(self, masks, height, width):
        """
        Args:
            masks (ndarray): ndarray of masks in shape (N, H, W), where N is the number of objects.
            height (int): height of masks
            width (int): width of masks
        """

        super().__init__(
            masks=masks,
            height=height,
            width=width)

    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        device='cpu',
                        interpolation='bilinear'):
        """The only difference from the original function is that change resized mask from np.uint8 to np.float.

        Args:
            bboxes (Tensor): Bboxes in format [x1, y1, x2, y2], shape (N, 4)
            out_shape (tuple[int]): Target (h, w) of resized mask
            inds (ndarray): Indexes to assign masks to each bbox, shape (N,)
                and values should be between [0, num_masks - 1].
            device (str): Device of bboxes
            interpolation (str): See `mmcv.imresize`

        Return:
            BitmapMasksTable: the cropped and resized masks.
        """

        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(
            num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = torch.from_numpy(self.masks).to(device).index_select(
                0, inds).to(dtype=rois.dtype)
            targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape,
                                1.0, 0, 'avg', True).squeeze(1)
            resized_masks = targets.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)


def get_lpmasks(gt_masks, gt_bboxes):
    """Produce local pyramid mask according to gt_bbox and gt_mask (for a batch of imags).

    Args:
        gt_masks(list(BitmapMasks)): masks of the text regions
        gt_bboxes(list(Tensor)): bboxes of the aligned cells

    Returns:
        list(BitmapMasks):pyramid masks in horizontal direction
        list(BitmapMasks):pyramid masks in vertical direction
    """

    gt_masks_temp = map(get_lpmask_single, gt_masks, gt_bboxes)
    gt_masks_temp = list(gt_masks_temp)
    gt_lpmasks_hor = [temp[0] for temp in gt_masks_temp]
    gt_lpmasks_ver = [temp[1] for temp in gt_masks_temp]

    return gt_lpmasks_hor, gt_lpmasks_ver


def get_lpmask_single(gt_mask, gt_bbox):
    """Produce local pyramid mask according to gt_bbox and gt_mask ((for one image).

    Args;
        gt_mask(BitmapMasks): masks of the text regions (for one image)
        gt_bbox(Tensor): (n x 4).bboxes of the aligned cells (for one image)

    Returns;
        BitmapMasksTable;pyramid masks in horizontal direction (for one image)
        BitmapMasksTable;pyramid masks in vertical direction (for one image)
    """

    (num, high, width) = gt_mask.masks.shape
    mask_s1 = np.zeros((num, high, width), np.float32)
    mask_s2 = np.zeros((num, high, width), np.float32)
    for ind, box_text in zip(range(num), gt_mask.masks):
        left_col, left_row, right_col, right_row = list(map(float, gt_bbox[ind, 0:4]))
        x_min, y_min, x_max, y_max = ceil(left_col), ceil(left_row), ceil(right_col) - 1, ceil(right_row) - 1
        middle_x, middle_y = round(np.where(box_text == 1)[1].mean()), round(np.where(box_text == 1)[0].mean())

        # Calculate the pyramid mask in horizontal direction
        col_np = np.arange(x_min, x_max + 1).reshape(1, -1)
        col_np_1 = (col_np[:, :middle_x - x_min] - left_col) / (middle_x - left_col)
        col_np_2 = (right_col - col_np[:, middle_x - x_min:]) / (right_col - middle_x)
        col_np = np.concatenate((col_np_1, col_np_2), axis=1)
        mask_s1[ind, y_min:y_max + 1, x_min:x_max + 1] = col_np

        # Calculate the pyramid mask in vertical direction
        row_np = np.arange(y_min, y_max + 1).reshape(-1, 1)
        row_np_1 = (row_np[:middle_y - y_min, :] - left_row) / (middle_y - left_row)
        row_np_2 = (right_row - row_np[middle_y - y_min:, :]) / (right_row - middle_y)
        row_np = np.concatenate((row_np_1, row_np_2), axis=0)
        mask_s2[ind, y_min:y_max + 1, x_min:x_max + 1] = row_np

    mask_s1 = BitmapMasksTable(mask_s1, high, width)
    mask_s2 = BitmapMasksTable(mask_s2, high, width)

    return mask_s1, mask_s2

