#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ：PdfTable 
# @File     : table_master.py
# @Author   : cycloneboy
# @Date     : 20xx/9/25 - 17:35
import inspect
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .master_convertor import MtlTabNetConvertor, TableMasterConvertor
from .master_decoder import PositionalEncoding, MtlTabNetDecoder, TableMasterDecoder
from .table_resnet_extra import TableResNetExtra
from ..lgpma.base_utils import imread, imshow_text_label

SUB_MODEL = {
    "MtlTabNetConvertor": MtlTabNetConvertor,
    "TableMasterConvertor": TableMasterConvertor,
    "TableResNetExtra": TableResNetExtra,
    "PositionalEncoding": PositionalEncoding,
    "TableMasterDecoder": TableMasterDecoder,
    "MtlTabNetDecoder": MtlTabNetDecoder,
}


def build_sub_model(config):
    args = config.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = SUB_MODEL.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Base class for text recognition."""

    def __init__(self):
        super().__init__()
        self.fp16_enabled = False

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (tensor): tensors with shape (N, C, H, W).
                Typically should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details of the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (list[tensor]): Tensor should have shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): The metadata of images.
        """
        pass

    def init_weights(self, pretrained=None):
        """Initialize the weights for detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (tensor | list[tensor]): Tensor should have shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[dict] | list[list[dict]]):
                The outer list indicates images in a batch.
        """
        if isinstance(imgs, list):
            assert len(imgs) > 0
            assert imgs[0].size(0) == 1, ('aug test does not support '
                                          f'inference with batch size '
                                          f'{imgs[0].size(0)}')
            assert len(imgs) == len(img_metas)
            return self.aug_test(imgs, img_metas, **kwargs)

        return self.simple_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_metas, return_loss=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note that img and img_meta are single-nested (i.e. tensor and
        list[dict]).
        """

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)

        if isinstance(img, list):
            for idx, each_img in enumerate(img):
                if each_img.dim() == 3:
                    img[idx] = each_img.unsqueeze(0)
        else:
            if len(img_metas) == 1 and isinstance(img_metas[0], list):
                img_metas = img_metas[0]

        return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw outputs of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer update, which are done by an optimizer
        hook. Note that in some complicated cases or models (e.g. GAN),
        the whole process (including the back propagation and optimizer update)
        is also defined by this method.

        Args:
            data (dict): The outputs of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which is a
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size used for
                averaging the logs (Note: for the
                DDP model, num_samples refers to the batch size for each GPU).
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but is
        used during val epochs. Note that the evaluation after training epochs
        is not implemented by this method, but by an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    gt_label='',
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Draw `result` on `img`.

        Args:
            img (str or tensor): The image to be displayed.
            result (dict): The results to draw on `img`.
            gt_label (str): Ground truth label of img.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The output filename.
                Default: None.

        Returns:
            img (tensor): Only if not `show` or `out_file`.
        """
        img = imread(img)
        img = img.copy()
        pred_label = None
        if 'text' in result.keys():
            pred_label = result['text']

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw text label
        if pred_label is not None:
            img = imshow_text_label(
                img,
                pred_label,
                gt_label,
                show=show,
                win_name=win_name,
                wait_time=wait_time,
                out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

        return img


class EncodeDecodeRecognizer(BaseRecognizer):
    """Base class for encode-decode recognizer."""

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None,
                 model_name="MtlTabNet"):
        super().__init__()

        self.model_name = model_name
        # Label convertor (str2tensor, tensor2str)
        assert label_convertor is not None
        label_convertor.update(max_seq_len=max_seq_len)
        self.label_convertor = build_sub_model(label_convertor)

        # Preprocessor module, e.g., TPS
        self.preprocessor = None
        if preprocessor is not None:
            self.preprocessor = build_sub_model(preprocessor)

        # Backbone
        assert backbone is not None
        self.backbone = build_sub_model(backbone)

        # Encoder module
        self.encoder = None
        if encoder is not None:
            self.encoder = build_sub_model(encoder)

        decoder = self.update_decoder_config(decoder=decoder,
                                             max_seq_len=max_seq_len)
        self.decoder = build_sub_model(decoder)

        # Loss
        assert loss is not None
        loss.update(ignore_index=self.label_convertor.padding_idx)
        self.loss = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_seq_len = max_seq_len
        self.init_weights(pretrained=pretrained)

    def update_decoder_config(self, decoder, max_seq_len):
        # Decoder module
        assert decoder is not None
        decoder.update(num_classes=self.label_convertor.num_classes())
        decoder.update(start_idx=self.label_convertor.start_idx)
        decoder.update(padding_idx=self.label_convertor.padding_idx)
        decoder.update(max_seq_len=max_seq_len)

        if self.model_name == "MtlTabNet":
            # Decoder module
            assert decoder is not None
            decoder.update(end_idx=self.label_convertor.end_idx)
            # author: namly
            # update parameter for the cell content decoder
            decoder.update(num_classes_cell=self.label_convertor.num_classes_cell())
            decoder.update(start_idx_cell=self.label_convertor.start_idx_cell)
            decoder.update(end_idx_cell=self.label_convertor.end_idx_cell)
            decoder.update(padding_idx_cell=self.label_convertor.padding_idx_cell)
            decoder.update(max_seq_len_cell=self.label_convertor.max_seq_len_cell)
            decoder.update(idx_tag_cell=self.label_convertor.idx_tag_cell())

        return decoder

    def init_weights(self, pretrained=None):
        """Initialize the weights of recognizer."""
        super().init_weights(pretrained)

        if self.preprocessor is not None:
            self.preprocessor.init_weights()

        self.backbone.init_weights()

        if self.encoder is not None:
            self.encoder.init_weights()

        self.decoder.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        if self.preprocessor is not None:
            img = self.preprocessor(img)

        x = self.backbone(img)

        return x

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)

        gt_labels = [img_meta['text'] for img_meta in img_metas]

        targets_dict = self.label_convertor.str2tensor(gt_labels)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat, img_metas)

        out_dec = self.decoder(
            feat, out_enc, targets_dict, img_metas, train_mode=True)

        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        losses = self.loss(*loss_inputs)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat, img_metas)

        out_dec = self.decoder(
            feat, out_enc, None, img_metas, train_mode=False)

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            out_dec, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results

    def merge_aug_results(self, aug_results):
        out_text, out_score = '', -1
        for result in aug_results:
            text = result[0]['text']
            score = sum(result[0]['score']) / max(1, len(text))
            if score > out_score:
                out_text = text
                out_score = score
        out_results = [dict(text=out_text, score=out_score)]
        return out_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function as well as time augmentation.

        Args:
            imgs (list[tensor]): Tensor should have shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): The metadata of images.
        """
        aug_results = []
        for img, img_meta in zip(imgs, img_metas):
            result = self.simple_test(img, img_meta, **kwargs)
            aug_results.append(result)

        return self.merge_aug_results(aug_results)


class MtlTabNet(EncodeDecodeRecognizer):
    # need to inherit BaseRecognizer or EncodeDecodeRecognizer in mmocr
    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 bbox_loss=None,
                 cell_loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None):
        super(MtlTabNet, self).__init__(preprocessor,
                                        backbone,
                                        encoder,
                                        decoder,
                                        loss,
                                        label_convertor,
                                        train_cfg,
                                        test_cfg,
                                        max_seq_len,
                                        pretrained,
                                        model_name="MtlTabNet")
        # build bbox loss
        self.bbox_loss = None

        # namly build cell loss
        self.cell_loss = None

    def init_weights(self, pretrained=None):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)
        feat = feat[-1]

        targets_dict = self.label_convertor.str_bbox_format(img_metas)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        out_dec, out_bbox, out_cell = self.decoder(
            feat, out_enc, targets_dict, img_metas, train_mode=True)

        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        losses = self.loss(*loss_inputs)

        bbox_loss_inputs = (
            out_bbox,
            targets_dict,
            img_metas,
        )
        bbox_losses = self.bbox_loss(*bbox_loss_inputs)

        losses.update(bbox_losses)

        # namly
        # cell loss
        cell_losses = 0.0
        for idx_, cell_padded_targets_i in enumerate(targets_dict['cell_padded_targets']):
            cell_padded_targets_i = torch.stack(cell_padded_targets_i, 0).long()
            cell_padded_targets = {'padded_targets': cell_padded_targets_i}
            cell_loss_inputs = (
                out_cell[idx_],
                cell_padded_targets,
                img_metas,
            )

            cell_losses += self.cell_loss(*cell_loss_inputs)

        cell_losses_ = {'loss_ce_cell': cell_losses / len(targets_dict['cell_padded_targets'])}
        losses.update(cell_losses_)
        # namly
        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)
        feat = feat[-1]

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        # namly: add cell content decoder
        out_dec, out_bbox, out_cell = self.decoder(
            feat, out_enc, None, img_metas, train_mode=False)

        strings, scores, pred_bboxes, pred_cells, pred_cells_scores = \
            self.label_convertor.output_format(out_dec, out_bbox, out_cell, img_metas)

        # flatten batch results
        results = []
        for string, score, pred_bbox, pred_cell in zip(strings, scores, pred_bboxes, pred_cells):
            results.append(dict(text=string, score=score, bbox=pred_bbox, cell=pred_cell))
        # namly
        # visual_pred_bboxes(img_metas, results)

        return results


class TableMaster(EncodeDecodeRecognizer):
    # need to inherit BaseRecognizer or EncodeDecodeRecognizer in mmocr
    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 bbox_loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None):
        super(TableMaster, self).__init__(preprocessor,
                                          backbone,
                                          encoder,
                                          decoder,
                                          loss,
                                          label_convertor,
                                          train_cfg,
                                          test_cfg,
                                          max_seq_len,
                                          pretrained,
                                          model_name="TableMaster")
        # build bbox loss
        self.bbox_loss = None

    def init_weights(self, pretrained=None):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)
        feat = feat[-1]

        targets_dict = self.label_convertor.str_bbox_format(img_metas)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        out_dec, out_bbox = self.decoder(
            feat, out_enc, targets_dict, img_metas, train_mode=True)

        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        losses = self.loss(*loss_inputs)

        bbox_loss_inputs = (
            out_bbox,
            targets_dict,
            img_metas,
        )
        bbox_losses = self.bbox_loss(*bbox_loss_inputs)

        losses.update(bbox_losses)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)
        feat = feat[-1]

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        out_dec, out_bbox = self.decoder(
            feat, out_enc, None, img_metas, train_mode=False)

        strings, scores, pred_bboxes = \
            self.label_convertor.output_format(out_dec, out_bbox, img_metas)

        # flatten batch results
        results = []
        for string, score, pred_bbox in zip(strings, scores, pred_bboxes):
            results.append(dict(text=string, score=score, bbox=pred_bbox))

        # visual_pred_bboxes(img_metas, results)

        return results
