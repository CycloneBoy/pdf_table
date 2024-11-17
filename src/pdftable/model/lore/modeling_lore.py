#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：modeling_lore
# @Author  ：cycloneboy
# @Date    ：20xx/5/26 14:05
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Union

import torch
from torch import nn
from transformers.utils import ModelOutput

from pdftable.utils import logger, FileUtils
from .configuration_lore import LoreConfig
from .lineless_table_process import load_lore_model, process_detect_output
from .lore_detector import LoreDetectModel
from .lore_dla_34 import get_dla_dcn
from .lore_processor import LoreProcessModel
from ...loss.lore_loss import TableLoreLoss

"""
model_scope lore

"""

__all__ = [
    'LoreModel',
    'LoreObjectDetectionOutput',
]


@dataclass
class LoreObjectDetectionOutput(ModelOutput):
    """
    Output type of [`LoreObjectDetectionOutput`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~YolosImageProcessor.post_process`] to retrieve the unnormalized bounding
            boxes.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None


class LoreModel(nn.Module):
    '''
    The model first locates table cells in the input image by key point segmentation.
    Then the logical locations are predicted along with the spatial locations
    employing two cascading regressors.
    See details in paper "LORE: Logical Location Regression Network for Table Structure Recognition"
    (https://arxiv.org/abs/2303.03730).
    '''

    def __init__(self, config: LoreConfig, **kwargs):
        '''initialize the LORE model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        '''
        super().__init__()
        self.config = config
        self.loss = TableLoreLoss(config=config)

        # init detect infer model
        if "resnet" in config.backbone.lower():
            self.detect_infer_model = LoreDetectModel()
        else:
            heads = {'hm': 2, 'st': 8, 'wh': 8, 'ax': 256, 'cr': 256, 'reg': 2}
            self.detect_infer_model = get_dla_dcn(
                num_layers=34,
                heads=heads,
                head_conv=256,
                pretrained=config.pretrained
            )

        # init process infer model
        self.process_infer_model = LoreProcessModel(config=config)

        if self.config.model_path != "" and os.path.exists(self.config.model_path) and not config.pretrained:
            self.load_model()

    def load_model(self, ):
        epoch = ""
        model_path = f"{self.config.model_path}/pytorch_model.pt"
        if FileUtils.check_file_exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            load_lore_model(self.detect_infer_model, checkpoint, 'model')
            load_lore_model(self.process_infer_model, checkpoint, 'processor')
        else:
            model_name_or_path = self.config.model_path
            model_path = f"{model_name_or_path}/model_best.pth"
            model_checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            load_lore_model(self.detect_infer_model, model_checkpoint, strict=True)

            epoch = model_checkpoint.get('epoch', "")

            processor_path = f"{model_name_or_path}/processor_best.pth"
            processor_checkpoint = torch.load(processor_path, map_location='cpu', weights_only=True)
            load_lore_model(self.process_infer_model, processor_checkpoint, strict=True)

        logger.info(f'加载模型：{self.config.model_name} - {self.config.backbone} - {self.config.task_type} '
                    f'- {self.config.model_path} epoch: {epoch}')

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            meta: Optional[torch.FloatTensor] = None,
            labels: Optional[List[Dict]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple, LoreObjectDetectionOutput]:
        """
        Args:
            img (`torch.Tensor`): image tensor,
                shape of each tensor is [3, H, W].

        Return:
            dets (`torch.Tensor`): the locations of detected table cells,
                shape of each tensor is [N_cell, 8].
            dets (`torch.Tensor`): the logical coordinates of detected table cells,
                shape of each tensor is [N_cell, 4].
            meta (`Dict`): the meta info of original image.
        """
        run_device = pixel_values.device
        outputs = self.detect_infer_model(pixel_values)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is None:
            output = outputs[-1]
            slct_logi_feat, slct_dets_feat, results, corner_st_reg = process_detect_output(output, meta,
                                                                                           upper_left=self.config.upper_left,
                                                                                           wiz_rev=self.config.wiz_rev,
                                                                                           vis_thresh=self.config.vis_thresh)

            slct_output_dets = results[1][:slct_logi_feat.shape[1], :8]
            # results = ps_convert_minmax(results, num_classes=2)

            if self.config.wiz_2dpe:
                if self.config.wiz_stacking:
                    _, slct_logi = self.process_infer_model(slct_logi_feat, dets=slct_dets_feat.to(torch.int64))
                else:
                    slct_logi = self.process_infer_model(slct_logi_feat, dets=slct_dets_feat.to(torch.int64))
            else:
                if self.config.wiz_stacking:
                    _, slct_logi = self.process_infer_model(slct_logi_feat)
                else:
                    slct_logi = self.process_infer_model(slct_logi_feat)

            pred_boxes = torch.tensor(slct_output_dets, dtype=torch.float32, device=run_device).reshape(1, -1, 8)
            logits = slct_logi

            loss = torch.ones([1], device=run_device, dtype=pixel_values.dtype)
            if len(slct_output_dets) == 0:
                pred_boxes = torch.zeros([1, 1, 8], device=run_device)
                logits = torch.zeros([1, 1, 4], device=run_device)
        else:
            if self.config.wiz_stacking:
                logic_axis, stacked_axis = self.process_infer_model(outputs, labels)

                loss, loss_dict = self.loss(outputs, labels, logic_axis, stacked_axis)
            else:
                logic_axis = self.process_infer_model(outputs, labels)
                loss, loss_dict = self.loss(outputs, labels, logic_axis)
                stacked_axis = None

            logits = logic_axis
            pred_boxes = stacked_axis
        if not return_dict:
            output = (pred_boxes, logits)
            return ((loss, loss_dict) + output) if loss is not None else output

        return LoreObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
        )
