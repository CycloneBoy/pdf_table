# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .pico_utils import ConvNormLayer, batch_distance2bbox, distance2bbox, varifocal_loss, MultiClassNMS


eps = 1e-9

__all__ = [
    'PicoHead',
    'PicoFeat'
]


class PicoSE(nn.Module):
    def __init__(self, feat_channels):
        super(PicoSE, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvNormLayer(feat_channels, feat_channels, 1, 1)

        self._init_weights()

    def _init_weights(self):
        # normal_(self.fc.weight, std=0.001)
        pass

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        out = self.conv(feat * weight)
        return out


# @register
class PicoFeat(nn.Module):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        share_cls_reg (bool): Whether to share the cls and reg output.
        act (str): The act of per layers.
        use_se (bool): Whether to use se module.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=96,
                 num_fpn_stride=3,
                 num_convs=2,
                 norm_type='bn',
                 share_cls_reg=False,
                 act='hard_swish',
                 use_se=False):
        super(PicoFeat, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.use_se = use_se
        self.cls_convs = []
        self.reg_convs = []
        if use_se:
            assert share_cls_reg == True, \
                'In the case of using se, share_cls_reg must be set to True'
            self.se = nn.ModuleList()
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                cls_conv_dw = ConvNormLayer(
                    ch_in=in_c,
                    ch_out=feat_out,
                    filter_size=5,
                    stride=1,
                    groups=feat_out,
                    norm_type=norm_type,
                    bias_on=False,
                    lr_scale=2.)
                self.add_module('cls_conv_dw{}_{}'.format(stage_idx, i), cls_conv_dw)
                cls_subnet_convs.append(cls_conv_dw)
                cls_conv_pw = ConvNormLayer(
                    ch_in=in_c,
                    ch_out=feat_out,
                    filter_size=1,
                    stride=1,
                    norm_type=norm_type,
                    bias_on=False,
                    lr_scale=2.)
                self.add_module('cls_conv_pw{}_{}'.format(stage_idx, i), cls_conv_pw)
                cls_subnet_convs.append(cls_conv_pw)

                if not self.share_cls_reg:
                    reg_conv_dw = ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=5,
                        stride=1,
                        groups=feat_out,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.)
                    self.add_module('reg_conv_dw{}_{}'.format(stage_idx, i), reg_conv_dw)
                    reg_subnet_convs.append(reg_conv_dw)
                    reg_conv_pw = ConvNormLayer(
                        ch_in=in_c,
                        ch_out=feat_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        bias_on=False,
                        lr_scale=2.)
                    self.add_module('reg_conv_pw{}_{}'.format(stage_idx, i), reg_conv_pw)
                    reg_subnet_convs.append(reg_conv_pw)
            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)
            if use_se:
                self.se.append(PicoSE(feat_out))

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        elif self.act == "relu6":
            x = F.relu6(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            reg_feat = cls_feat
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))
        if self.use_se:
            avg_feat = F.adaptive_avg_pool2d(cls_feat, (1, 1))
            se_feat = self.act_func(self.se[stage_idx](cls_feat, avg_feat))
            return cls_feat, se_feat
        return cls_feat, reg_feat


class ScaleReg(nn.Module):
    """
    Parameter for scaling the regression outputs.
    """

    def __init__(self):
        super(ScaleReg, self).__init__()
        self.scale_reg = nn.Parameter(torch.ones([1]), requires_grad=False)

    def forward(self, inputs):
        out = inputs * self.scale_reg
        return out


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape([-1, self.reg_max + 1]), dim=1)
        x = F.linear(x, self.project)
        if self.training:
            x = x.reshape([-1, 4])
        return x


class GFLHead(nn.Module):
    """
    GFLHead
    Args:
        conv_feat (object): Instance of 'FCOSFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of QualityFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 16.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='FCOSFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 loss_class='QualityFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 reg_max=16,
                 feat_in_chan=256,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(GFLHead, self).__init__()
        self.conv_feat = conv_feat
        self.dgqp_module = dgqp_module
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_qfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_qfl.use_sigmoid
        self.use_sigmoid = True
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        conv_cls_name = "gfl_head_cls"
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.gfl_head_cls = self.add_module(
            conv_cls_name,
            nn.Conv2d(
                in_channels=self.feat_in_chan,
                out_channels=self.cls_out_channels,
                kernel_size=3,
                stride=1,
                padding=1))

        conv_reg_name = "gfl_head_reg"
        self.gfl_head_reg = self.add_module(
            conv_reg_name,
            nn.Conv2d(
                in_channels=self.feat_in_chan,
                out_channels=4 * (self.reg_max + 1),
                kernel_size=3,
                stride=1,
                padding=1))

        self.scales_regs = []
        for i in range(len(self.fpn_stride)):
            lvl = int(math.log(int(self.fpn_stride[i]), 2))
            feat_name = 'p{}_feat'.format(lvl)
            scale_reg = self.add_module(feat_name, ScaleReg())
            self.scales_regs.append(scale_reg)

        self.distribution_project = Integral(self.reg_max)

    def forward(self, fpn_feats):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"
        cls_logits_list = []
        bboxes_reg_list = []
        for stride, scale_reg, fpn_feat in zip(self.fpn_stride,
                                               self.scales_regs, fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat)
            cls_score = self.gfl_head_cls(conv_cls_feat)
            bbox_pred = scale_reg(self.gfl_head_reg(conv_reg_feat))
            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score
            if not self.training:
                cls_score = F.sigmoid(cls_score.transpose([0, 2, 3, 1]))
                bbox_pred = bbox_pred.transpose([0, 2, 3, 1])
                b, cell_h, cell_w, _ = cls_score.shape
                y, x = self.get_single_level_center_point(
                    [cell_h, cell_w], stride, cell_offset=self.cell_offset)
                center_points = torch.stack([x, y], dim=-1)
                cls_score = cls_score.reshape([b, -1, self.cls_out_channels])
                bbox_pred = self.distribution_project(bbox_pred) * stride
                bbox_pred = bbox_pred.reshape([b, cell_h * cell_w, 4])

                # NOTE: If keep_ratio=False and image shape value that
                # multiples of 32, distance2bbox not set max_shapes parameter
                # to speed up model prediction. If need to set max_shapes,
                # please use inputs['im_shape'].
                bbox_pred = batch_distance2bbox(
                    center_points, bbox_pred, max_shapes=None)

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)

        return (cls_logits_list, bboxes_reg_list)

    def _images_to_levels(self, target, num_level_anchors):
        """
        Convert targets by image to targets by feature level.
        """
        level_targets = []
        start = 0
        for n in num_level_anchors:
            end = start + n
            level_targets.append(target[:, start:end].squeeze(0))
            start = end
        return level_targets

    def _grid_cells_to_center(self, grid_cells):
        """
        Get center location of each gird cell
        Args:
            grid_cells: grid cells of a feature map
        Returns:
            center points
        """
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return torch.stack([cells_cx, cells_cy], dim=-1)

    def get_loss(self, gfl_head_outs, gt_meta):
        cls_logits, bboxes_reg = gfl_head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_logits
        ]
        grid_cells_list = self._images_to_levels(gt_meta['grid_cells'],
                                                 num_level_anchors)
        labels_list = self._images_to_levels(gt_meta['labels'],
                                             num_level_anchors)
        label_weights_list = self._images_to_levels(gt_meta['label_weights'],
                                                    num_level_anchors)
        bbox_targets_list = self._images_to_levels(gt_meta['bbox_targets'],
                                                   num_level_anchors)
        num_total_pos = sum(gt_meta['pos_num'])
        try:
            num_total_pos = torch.distributed.all_reduce(num_total_pos.clone(
            )) / torch.distributed.get_world_size()
        except:
            num_total_pos = max(num_total_pos, 1)

        loss_bbox_list, loss_dfl_list, loss_qfl_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, grid_cells, labels, label_weights, bbox_targets, stride in zip(
                cls_logits, bboxes_reg, grid_cells_list, labels_list,
                label_weights_list, bbox_targets_list, self.fpn_stride):
            grid_cells = grid_cells.reshape([-1, 4])
            cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])
            bbox_targets = bbox_targets.reshape([-1, 4])
            labels = labels.reshape([-1])
            label_weights = label_weights.reshape([-1])

            bg_class_ind = self.num_classes
            pos_inds = torch.nonzero(
                torch.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
            score = np.zeros(labels.shape)
            if len(pos_inds) > 0:
                pos_bbox_targets = torch.gather(bbox_targets, pos_inds, dim=0)
                pos_bbox_pred = torch.gather(bbox_pred, pos_inds, dim=0)
                pos_grid_cells = torch.gather(grid_cells, pos_inds, dim=0)
                pos_grid_cell_centers = self._grid_cells_to_center(
                    pos_grid_cells) / stride

                weight_targets = F.sigmoid(cls_score.detach())
                weight_targets = torch.gather(
                    weight_targets.max(dim=1, keepdim=True), pos_inds, dim=0)
                pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_grid_cell_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                bbox_iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach().numpy(),
                    pos_decode_bbox_targets.detach().numpy(),
                    is_aligned=True)
                score[pos_inds.numpy()] = bbox_iou
                pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
                target_corners = bbox2distance(pos_grid_cell_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape([-1])
                # regression loss
                loss_bbox = torch.sum(
                    self.loss_bbox(pos_decode_bbox_pred,
                                   pos_decode_bbox_targets) * weight_targets)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=4.0)
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = torch.to_tensor([0], dtype='float32')

            # qfl loss
            score = torch.to_tensor(score)
            loss_qfl = self.loss_qfl(
                cls_score, (labels, score),
                weight=label_weights,
                avg_factor=num_total_pos)
            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_qfl_list.append(loss_qfl)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        try:
            torch.distributed.all_reduce(avg_factor)
            avg_factor = torch.clip(
                avg_factor / torch.distributed.get_world_size(), min=1)
        except:
            avg_factor = max(avg_factor.item(), 1)
        if avg_factor <= 0:
            loss_qfl = torch.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = torch.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = torch.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
            losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
            loss_qfl = sum(loss_qfl_list)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss_states = dict(
            loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states

    def get_single_level_center_point(self, featmap_size, stride,
                                      cell_offset=0):
        """
        Generate pixel centers of a single stage feature map.
        Args:
            featmap_size: height and width of the feature map
            stride: down sample stride of the feature map
        Returns:
            y and x of the center points
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype='float32') + cell_offset) * stride
        y_range = (torch.arange(h, dtype='float32') + cell_offset) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        return y, x

    def post_process(self, gfl_head_outs, im_shape, scale_factor):
        cls_scores, bboxes_reg = gfl_head_outs
        bboxes = torch.concat(bboxes_reg, dim=1)
        # rescale: [h_scale, w_scale] -> [w_scale, h_scale, w_scale, h_scale]
        im_scale = scale_factor.flip([1]).tile([1, 2]).unsqueeze(1)
        bboxes /= im_scale
        mlvl_scores = torch.concat(cls_scores, dim=1)
        mlvl_scores = mlvl_scores.transpose([0, 2, 1])
        bbox_pred, bbox_num, _ = self.nms(bboxes, mlvl_scores)
        return bbox_pred, bbox_num


class OTAHead(GFLHead):
    """
    OTAHead
    Args:
        conv_feat (object): Instance of 'FCOSFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_qfl (object): Instance of QualityFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 16.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'assigner', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='FCOSFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 loss_class='QualityFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 assigner='SimOTAAssigner',
                 reg_max=16,
                 feat_in_chan=256,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(OTAHead, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.dgqp_module = dgqp_module
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_qfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_qfl.use_sigmoid

        self.assigner = assigner

    def _get_target_single(self, flatten_cls_pred, flatten_center_and_stride,
                           flatten_bbox, gt_bboxes, gt_labels):
        """Compute targets for priors in a single image.
        """
        pos_num, label, label_weight, bbox_target = self.assigner(
            F.sigmoid(flatten_cls_pred), flatten_center_and_stride,
            flatten_bbox, gt_bboxes, gt_labels)

        return (pos_num, label, label_weight, bbox_target)

    def get_loss(self, head_outs, gt_meta):
        cls_scores, bbox_preds = head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_scores
        ]
        num_imgs = gt_meta['im_id'].shape[0]
        featmap_sizes = [[featmap.shape[-2], featmap.shape[-1]]
                         for featmap in cls_scores]

        decode_bbox_preds = []
        center_and_strides = []
        for featmap_size, stride, bbox_pred in zip(featmap_sizes,
                                                   self.fpn_stride, bbox_preds):
            # center in origin image
            yy, xx = self.get_single_level_center_point(featmap_size, stride,
                                                        self.cell_offset)

            center_and_stride = torch.stack([xx, yy, stride, stride], -1).tile(
                [num_imgs, 1, 1])
            center_and_strides.append(center_and_stride)
            center_in_feature = center_and_stride.reshape(
                [-1, 4])[:, :-2] / stride
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, 4 * (self.reg_max + 1)])
            pred_distances = self.distribution_project(bbox_pred)
            decode_bbox_pred_wo_stride = distance2bbox(
                center_in_feature, pred_distances).reshape([num_imgs, -1, 4])
            decode_bbox_preds.append(decode_bbox_pred_wo_stride * stride)

        flatten_cls_preds = [
            cls_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, self.cls_out_channels])
            for cls_pred in cls_scores
        ]
        flatten_cls_preds = torch.concat(flatten_cls_preds, axis=1)
        flatten_bboxes = torch.concat(decode_bbox_preds, axis=1)
        flatten_center_and_strides = torch.concat(center_and_strides, axis=1)

        gt_boxes, gt_labels = gt_meta['gt_bbox'], gt_meta['gt_class']
        pos_num_l, label_l, label_weight_l, bbox_target_l = [], [], [], []
        for flatten_cls_pred, flatten_center_and_stride, flatten_bbox, gt_box, gt_label \
                in zip(flatten_cls_preds.detach(), flatten_center_and_strides.detach(), \
                       flatten_bboxes.detach(), gt_boxes, gt_labels):
            pos_num, label, label_weight, bbox_target = self._get_target_single(
                flatten_cls_pred, flatten_center_and_stride, flatten_bbox,
                gt_box, gt_label)
            pos_num_l.append(pos_num)
            label_l.append(label)
            label_weight_l.append(label_weight)
            bbox_target_l.append(bbox_target)

        labels = torch.to_tensor(np.stack(label_l, axis=0))
        label_weights = torch.to_tensor(np.stack(label_weight_l, axis=0))
        bbox_targets = torch.to_tensor(np.stack(bbox_target_l, axis=0))

        center_and_strides_list = self._images_to_levels(
            flatten_center_and_strides, num_level_anchors)
        labels_list = self._images_to_levels(labels, num_level_anchors)
        label_weights_list = self._images_to_levels(label_weights,
                                                    num_level_anchors)
        bbox_targets_list = self._images_to_levels(bbox_targets,
                                                   num_level_anchors)
        num_total_pos = sum(pos_num_l)
        try:
            torch.distributed.all_reduce(num_total_pos)
            num_total_pos = num_total_pos / torch.distributed.get_world_size()
        except:
            num_total_pos = max(num_total_pos, 1)

        loss_bbox_list, loss_dfl_list, loss_qfl_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, center_and_strides, labels, label_weights, bbox_targets, stride in zip(
                cls_scores, bbox_preds, center_and_strides_list, labels_list,
                label_weights_list, bbox_targets_list, self.fpn_stride):
            center_and_strides = center_and_strides.reshape([-1, 4])
            cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])
            bbox_targets = bbox_targets.reshape([-1, 4])
            labels = labels.reshape([-1])
            label_weights = label_weights.reshape([-1])

            bg_class_ind = self.num_classes
            pos_inds = torch.nonzero(
                torch.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
            score = np.zeros(labels.shape)

            if len(pos_inds) > 0:
                pos_bbox_targets = torch.gather(bbox_targets, pos_inds, axis=0)
                pos_bbox_pred = torch.gather(bbox_pred, pos_inds, axis=0)
                pos_centers = torch.gather(
                    center_and_strides[:, :-2], pos_inds, axis=0) / stride

                weight_targets = F.sigmoid(cls_score.detach())
                weight_targets = torch.gather(
                    weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
                pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                bbox_iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach().numpy(),
                    pos_decode_bbox_targets.detach().numpy(),
                    is_aligned=True)
                score[pos_inds.numpy()] = bbox_iou

                pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
                target_corners = bbox2distance(pos_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape([-1])
                # regression loss
                loss_bbox = torch.sum(
                    self.loss_bbox(pos_decode_bbox_pred,
                                   pos_decode_bbox_targets) * weight_targets)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=4.0)
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = torch.to_tensor([0], dtype='float32')

            # qfl loss
            score = torch.to_tensor(score)
            loss_qfl = self.loss_qfl(
                cls_score, (labels, score),
                weight=label_weights,
                avg_factor=num_total_pos)
            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_qfl_list.append(loss_qfl)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        try:
            torch.distributed.all_reduce(avg_factor)
            avg_factor = torch.clip(
                avg_factor / torch.distributed.get_world_size(), min=1)
        except:
            avg_factor = max(avg_factor.item(), 1)
        if avg_factor <= 0:
            loss_qfl = torch.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = torch.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = torch.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
            losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
            loss_qfl = sum(loss_qfl_list)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss_states = dict(
            loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states


class OTAVFLHead(OTAHead):
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'assigner', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='FCOSFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 loss_class='VarifocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 assigner='SimOTAAssigner',
                 reg_max=16,
                 feat_in_chan=256,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(OTAVFLHead, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.dgqp_module = dgqp_module
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_vfl.use_sigmoid

        self.assigner = assigner

    def get_loss(self, head_outs, gt_meta):
        cls_scores, bbox_preds = head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_scores
        ]
        num_imgs = gt_meta['im_id'].shape[0]
        featmap_sizes = [[featmap.shape[-2], featmap.shape[-1]]
                         for featmap in cls_scores]

        decode_bbox_preds = []
        center_and_strides = []
        for featmap_size, stride, bbox_pred in zip(featmap_sizes,
                                                   self.fpn_stride, bbox_preds):
            # center in origin image
            yy, xx = self.get_single_level_center_point(featmap_size, stride,
                                                        self.cell_offset)
            strides = torch.full((len(xx),), stride)
            center_and_stride = torch.stack([xx, yy, strides, strides],
                                            -1).tile([num_imgs, 1, 1])
            center_and_strides.append(center_and_stride)
            center_in_feature = center_and_stride.reshape(
                [-1, 4])[:, :-2] / stride
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, 4 * (self.reg_max + 1)])
            pred_distances = self.distribution_project(bbox_pred)
            decode_bbox_pred_wo_stride = distance2bbox(
                center_in_feature, pred_distances).reshape([num_imgs, -1, 4])
            decode_bbox_preds.append(decode_bbox_pred_wo_stride * stride)

        flatten_cls_preds = [
            cls_pred.transpose([0, 2, 3, 1]).reshape(
                [num_imgs, -1, self.cls_out_channels])
            for cls_pred in cls_scores
        ]
        flatten_cls_preds = torch.concat(flatten_cls_preds, axis=1)
        flatten_bboxes = torch.concat(decode_bbox_preds, axis=1)
        flatten_center_and_strides = torch.concat(center_and_strides, axis=1)

        gt_boxes, gt_labels = gt_meta['gt_bbox'], gt_meta['gt_class']
        pos_num_l, label_l, label_weight_l, bbox_target_l = [], [], [], []
        for flatten_cls_pred, flatten_center_and_stride, flatten_bbox, gt_box, gt_label \
                in zip(flatten_cls_preds.detach(), flatten_center_and_strides.detach(), \
                       flatten_bboxes.detach(), gt_boxes, gt_labels):
            pos_num, label, label_weight, bbox_target = self._get_target_single(
                flatten_cls_pred, flatten_center_and_stride, flatten_bbox,
                gt_box, gt_label)
            pos_num_l.append(pos_num)
            label_l.append(label)
            label_weight_l.append(label_weight)
            bbox_target_l.append(bbox_target)

        labels = torch.to_tensor(np.stack(label_l, axis=0))
        label_weights = torch.to_tensor(np.stack(label_weight_l, axis=0))
        bbox_targets = torch.to_tensor(np.stack(bbox_target_l, axis=0))

        center_and_strides_list = self._images_to_levels(
            flatten_center_and_strides, num_level_anchors)
        labels_list = self._images_to_levels(labels, num_level_anchors)
        label_weights_list = self._images_to_levels(label_weights,
                                                    num_level_anchors)
        bbox_targets_list = self._images_to_levels(bbox_targets,
                                                   num_level_anchors)
        num_total_pos = sum(pos_num_l)
        try:
            torch.distributed.all_reduce(num_total_pos)
            num_total_pos = num_total_pos / torch.distributed.get_world_size()
        except:
            num_total_pos = max(num_total_pos, 1)

        loss_bbox_list, loss_dfl_list, loss_vfl_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, center_and_strides, labels, label_weights, bbox_targets, stride in zip(
                cls_scores, bbox_preds, center_and_strides_list, labels_list,
                label_weights_list, bbox_targets_list, self.fpn_stride):
            center_and_strides = center_and_strides.reshape([-1, 4])
            cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])
            bbox_targets = bbox_targets.reshape([-1, 4])
            labels = labels.reshape([-1])

            bg_class_ind = self.num_classes
            pos_inds = torch.nonzero(
                torch.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
            # vfl
            vfl_score = np.zeros(cls_score.shape)

            if len(pos_inds) > 0:
                pos_bbox_targets = torch.gather(bbox_targets, pos_inds, axis=0)
                pos_bbox_pred = torch.gather(bbox_pred, pos_inds, axis=0)
                pos_centers = torch.gather(
                    center_and_strides[:, :-2], pos_inds, axis=0) / stride

                weight_targets = F.sigmoid(cls_score.detach())
                weight_targets = torch.gather(
                    weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
                pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                bbox_iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach().numpy(),
                    pos_decode_bbox_targets.detach().numpy(),
                    is_aligned=True)

                # vfl
                pos_labels = torch.gather(labels, pos_inds, axis=0)
                vfl_score[pos_inds.numpy(), pos_labels] = bbox_iou

                pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
                target_corners = bbox2distance(pos_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape([-1])
                # regression loss
                loss_bbox = torch.sum(
                    self.loss_bbox(pos_decode_bbox_pred,
                                   pos_decode_bbox_targets) * weight_targets)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=4.0)
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = torch.to_tensor([0], dtype='float32')

            # vfl loss
            num_pos_avg_per_gpu = num_total_pos
            vfl_score = torch.to_tensor(vfl_score)
            loss_vfl = self.loss_vfl(
                cls_score, vfl_score, avg_factor=num_pos_avg_per_gpu)

            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_vfl_list.append(loss_vfl)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        try:
            torch.distributed.all_reduce(avg_factor)
            avg_factor = torch.clip(
                avg_factor / torch.distributed.get_world_size(), min=1)
        except:
            avg_factor = max(avg_factor.item(), 1)
        if avg_factor <= 0:
            loss_vfl = torch.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = torch.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = torch.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
            losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
            loss_vfl = sum(loss_vfl_list)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss_states = dict(
            loss_vfl=loss_vfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states


# @register
class PicoHead(OTAVFLHead):
    """
    PicoHead
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_class', 'loss_dfl', 'loss_bbox',
        'assigner', 'nms'
    ]
    __shared__ = ['num_classes', 'eval_size']

    def __init__(self,
                 conv_feat='PicoFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32],
                 prior_prob=0.01,
                 loss_class='VariFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 assigner='SimOTAAssigner',
                 reg_max=16,
                 feat_in_chan=96,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0,
                 eval_size=None):

        conv_feat = PicoFeat(**conv_feat)
        loss_class = VarifocalLoss(**loss_class)
        nms = MultiClassNMS(**nms)
        super(PicoHead, self).__init__(
            conv_feat=conv_feat,
            dgqp_module=dgqp_module,
            num_classes=num_classes,
            fpn_stride=fpn_stride,
            prior_prob=prior_prob,
            loss_class=loss_class,
            loss_dfl=loss_dfl,
            loss_bbox=loss_bbox,
            assigner=assigner,
            reg_max=reg_max,
            feat_in_chan=feat_in_chan,
            nms=nms,
            nms_pre=nms_pre,
            cell_offset=cell_offset)
        self.conv_feat = conv_feat
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_class
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.assigner = assigner
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.eval_size = eval_size

        self.use_sigmoid = self.loss_vfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # Clear the super class initialization
        self.gfl_head_cls = None
        self.gfl_head_reg = None
        self.scales_regs = None

        self.head_cls_list = []
        self.head_reg_list = []
        for i in range(len(fpn_stride)):
            head_cls = nn.Conv2d(
                in_channels=self.feat_in_chan,
                out_channels=self.cls_out_channels + 4 * (self.reg_max + 1)
                if self.conv_feat.share_cls_reg else self.cls_out_channels,
                kernel_size=1,
                stride=1,
                padding=0)
            self.add_module("head_cls" + str(i), head_cls)
            self.head_cls_list.append(head_cls)
            if not self.conv_feat.share_cls_reg:
                head_reg = nn.Conv2d(
                    in_channels=self.feat_in_chan,
                    out_channels=4 * (self.reg_max + 1),
                    kernel_size=1,
                    stride=1,
                    padding=0, )
                self.add_module("head_reg" + str(i), head_reg)
                self.head_reg_list.append(head_reg)

        # initialize the anchor points
        if self.eval_size:
            self.anchor_points, self.stride_tensor = self._generate_anchors()

    def forward(self, fpn_feats, export_post_process=True):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"

        if self.training:
            return self.forward_train(fpn_feats)
        else:
            return self.forward_eval(
                fpn_feats, export_post_process=export_post_process)

    def forward_train(self, fpn_feats):
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = torch.split(
                    cls_logits,
                    [self.cls_out_channels, 4 * (self.reg_max + 1)],
                    dim=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)

            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)

        return (cls_logits_list, bboxes_reg_list)

    def forward_eval(self, fpn_feats, export_post_process=True):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(fpn_feats)
        cls_logits_list, bboxes_reg_list = [], []
        for i, fpn_feat in enumerate(fpn_feats):
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = torch.split(
                    cls_logits,
                    [self.cls_out_channels, 4 * (self.reg_max + 1)],
                    dim=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)

            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score

            if not export_post_process:
                # Now only supports batch size = 1 in deploy
                # TODO(ygh): support batch size > 1
                cls_score_out = F.sigmoid(cls_score).reshape(
                    [1, self.cls_out_channels, -1]).permute([0, 2, 1])
                bbox_pred = bbox_pred.reshape([1, (self.reg_max + 1) * 4,
                                               -1]).permute([0, 2, 1])
            else:
                b, _, h, w = fpn_feat.shape
                l = h * w
                cls_score_out = F.sigmoid(
                    cls_score.reshape([b, self.cls_out_channels, l]))
                bbox_pred = bbox_pred.permute([0, 2, 3, 1])
                bbox_pred = self.distribution_project(bbox_pred)
                bbox_pred = bbox_pred.reshape([b, l, 4])

            cls_logits_list.append(cls_score_out)
            bboxes_reg_list.append(bbox_pred)

        if export_post_process:
            cls_logits_list = torch.concat(cls_logits_list, dim=-1)
            bboxes_reg_list = torch.concat(bboxes_reg_list, dim=1)

            run_device = cls_logits_list.device
            anchor_points = anchor_points.to(run_device)
            stride_tensor = stride_tensor.to(run_device)

            bboxes_reg_list = batch_distance2bbox(anchor_points, bboxes_reg_list)
            bboxes_reg_list *= stride_tensor

        return (cls_logits_list, bboxes_reg_list)

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_stride):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = math.ceil(self.eval_size[0] / stride)
                w = math.ceil(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.cell_offset
            shift_y = torch.arange(end=h) + self.cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).float()
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full([h * w, 1], stride, dtype=torch.float32))
        anchor_points = torch.concat(anchor_points)
        stride_tensor = torch.concat(stride_tensor)
        return anchor_points, stride_tensor

    def post_process(self, head_outs, scale_factor, export_nms=True):
        pred_scores, pred_bboxes = head_outs
        if not export_nms:
            return pred_bboxes, pred_scores
        else:
            # rescale: [h_scale, w_scale] -> [w_scale, h_scale, w_scale, h_scale]
            scale_y, scale_x = torch.split(scale_factor, [1, 1], dim=-1)
            scale_factor = torch.concat([scale_x, scale_y, scale_x, scale_y], dim=-1).reshape([-1, 1, 4])
            # scale bbox to origin image size.
            pred_bboxes /= scale_factor
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num



class VarifocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='mean',
                 loss_weight=1.0):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(VarifocalLoss, self).__init__()
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * varifocal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                iou_weighted=self.iou_weighted,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
