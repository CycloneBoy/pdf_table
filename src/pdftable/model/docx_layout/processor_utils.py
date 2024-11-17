#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : processor_utils.py
# @Author   : cycloneboy
# @Date     : 20xx/12/5 - 21:46

import os
import time
import math
from functools import cmp_to_key
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from shapely.geometry import Polygon
from torch import TensorType


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def ctdet_4ps_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    # import pdb; pdb.set_trace()
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 8)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 8).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 8)
    else:
        wh = wh.view(batch, K, 8)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    '''
    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs + wh[..., 2:3], 
                        ys - wh[..., 3:4],
                        xs + wh[..., 4:5],
                        ys + wh[..., 5:6],
                        xs - wh[..., 6:7],
                        ys + wh[..., 7:8]], dim=2)
    '''
    bboxes = torch.cat([xs - wh[..., 0:1],
                        ys - wh[..., 1:2],
                        xs - wh[..., 2:3],
                        ys - wh[..., 3:4],
                        xs - wh[..., 4:5],
                        ys - wh[..., 5:6],
                        xs - wh[..., 6:7],
                        ys - wh[..., 7:8]], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections, inds


def ctdet_cls_decode(cls, inds):
    clses = _tranpose_and_gather_feat(cls, inds)
    return clses


def pnms(dets, thresh):
    if len(dets) < 2:
        return dets
    scores = dets[:, 8]
    index_keep = []
    keep = []
    for i in range(len(dets)):
        box = dets[i]
        if box[8] < thresh:
            continue
        max_score_index = -1
        ctx = (dets[i][0] + dets[i][2] + dets[i][4] + dets[i][6]) / 4
        cty = (dets[i][1] + dets[i][3] + dets[i][5] + dets[i][7]) / 4
        for j in range(len(dets)):
            if i == j or dets[j][8] < thresh:
                continue
            x1, y1 = dets[j][0], dets[j][1]
            x2, y2 = dets[j][2], dets[j][3]
            x3, y3 = dets[j][4], dets[j][5]
            x4, y4 = dets[j][6], dets[j][7]
            a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
            b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
            c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
            d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
            if ((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
                if dets[i][8] > dets[j][8] and max_score_index < 0:
                    max_score_index = i
                elif dets[i][8] < dets[j][8]:
                    max_score_index = -2
                    break
        if max_score_index > -1:
            index_keep.append(max_score_index)
        elif max_score_index == -1:
            index_keep.append(i)
    for i in range(0, len(index_keep)):
        keep.append(dets[index_keep[i]])

    return np.array(keep)


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
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


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def ctdet_4ps_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, 0:2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        dets[i, :, 4:6] = transform_preds(dets[i, :, 4:6], c[i], s[i], (w, h))
        dets[i, :, 6:8] = transform_preds(dets[i, :, 6:8], c[i], s[i], (w, h))
        classes = dets[i, :, 9]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :8].astype(np.float32),
                dets[i, inds, 8:].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def calc_main_angle(pts_list):
    if len(pts_list) == 0:
        return 0
    good_angles, other_angles = [], []
    for pts in pts_list:
        d_x_1, d_y_1 = pts[2] - pts[0], pts[3] - pts[1]
        d_x_2, d_y_2 = pts[4] - pts[2], pts[5] - pts[3]

        width = math.sqrt(d_x_1 ** 2 + d_y_1 ** 2)
        height = math.sqrt(d_x_2 ** 2 + d_y_2 ** 2)
        angle = math.atan2(d_y_1, d_x_1)

        if width > height * 3:
            good_angles.append(angle)
        else:
            other_angles.append(angle)

    if len(good_angles) > 0:
        good_angles.sort()
        return good_angles[len(good_angles) // 2]
    else:
        other_angles.sort()
        return other_angles[len(other_angles) // 2]


def calc_x_type(a, b):
    x_type = 0
    minx_a, maxx_a = a[0], a[0] + a[2]
    minx_b, maxx_b = b[0], b[0] + b[2]

    start_left = 0
    if minx_a < minx_b:
        start_left = 1
    elif minx_a > minx_b:
        start_left = -1
    end_right = 0
    if maxx_a > maxx_b:
        end_right = 1
    elif maxx_a < maxx_b:
        end_right = -1

    if maxx_a < minx_b + 1e-4 and maxx_a < maxx_b - 1e-4:
        x_type = 1  # left
    elif minx_a > maxx_b - 1e-4 and minx_a > minx_b + 1e-4:
        x_type = 2  # right
    elif start_left == 1 and end_right == -1:
        x_type = 3  # near left
    elif start_left == -1 and end_right == 1:
        x_type = 4  # near right
    elif start_left >= 0 and end_right >= 0:
        x_type = 5  # contain
    elif start_left <= 0 and end_right <= 0:
        x_type = 6  # inside
    else:
        x_type = 0

    return x_type


def calc_y_type(a, b):
    y_type = 0
    miny_a, maxy_a = a[1], a[1] + a[3]
    miny_b, maxy_b = b[1], b[1] + b[3]

    start_up = 0
    if miny_a < miny_b:
        start_up = 1
    elif miny_a > miny_b:
        start_up = -1
    end_down = 0
    if maxy_a > maxy_b:
        end_down = 1
    elif maxy_a < maxy_b:
        end_down = -1

    if maxy_a < miny_b + 1e-4 and maxy_a < maxy_b - 1e-4:
        y_type = 1  # up
    elif miny_a > maxy_b - 1e-4 and miny_a > miny_b + 1e-4:
        y_type = 2  # down
    elif start_up == 1 and end_down == -1:
        y_type = 3  # near up
    elif start_up == -1 and end_down == 1:
        y_type = 4  # near down
    elif start_up >= 0 and end_down >= 0:
        y_type = 5  # contain
    elif start_up <= 0 and end_down <= 0:
        y_type = 6  # inside
    else:
        y_type = 0

    return y_type


def sort_pts(blocks):
    main_angle = calc_main_angle([blk['pts'] for blk in blocks])
    main_sin, main_cos = math.sin(main_angle), math.cos(main_angle)

    def pts2rect(pts):
        xs, ys = [], []
        for k in range(0, len(pts), 2):
            x0 = pts[k] * main_cos + pts[k + 1] * main_sin
            y0 = pts[k + 1] * main_cos - pts[k] * main_sin
            xs.append(x0)
            ys.append(y0)
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        rect = [minx, miny, maxx - minx, maxy - miny]
        # print('===', pts, '->', rect)
        return rect

    def cmp_pts_udlr(a, b, thres=0.5):
        rect_a, rect_b = pts2rect(a['pts']), pts2rect(b['pts'])
        minx_a, miny_a, maxx_a, maxy_a = rect_a[0], rect_a[1], rect_a[0] + rect_a[2], rect_a[1] + rect_a[3]
        minx_b, miny_b, maxx_b, maxy_b = rect_b[0], rect_b[1], rect_b[0] + rect_b[2], rect_b[1] + rect_b[3]

        x_type, y_type = calc_x_type(rect_a, rect_b), calc_y_type(rect_a, rect_b)

        y_near_rate = 0.0
        if y_type == 3:
            y_near_rate = (maxy_a - miny_b) / min(maxy_a - miny_a, maxy_b - miny_b)
        elif y_type == 4:
            y_near_rate = (maxy_b - miny_a) / min(maxy_a - miny_a, maxy_b - miny_b)

        # print(rect_a, rect_b, x_type, y_type, y_near_rate)
        # exit(0)

        if y_type == 1:
            return -1
        elif y_type == 2:
            return 1
        elif y_type == 3:
            if x_type in [2, 4]:
                if y_near_rate < thres:
                    return -1
                else:
                    return 1
            else:
                return -1
        elif y_type == 4:
            if x_type in [1, 3]:
                if y_near_rate < thres:
                    return 1
                else:
                    return -1
            else:
                return 1
        else:
            if x_type == 1 or x_type == 3:
                return -1
            elif x_type == 2 or x_type == 4:
                return 1
            else:
                center_y_diff = abs(0.5 * (miny_a + maxy_a) - 0.5 * (miny_b + maxy_b))
                max_h = max(maxy_a - miny_a, maxy_b - miny_b)
                if center_y_diff / max_h < 0.1:
                    if (minx_a + maxx_a) < (minx_b + maxx_b):
                        return -1
                    elif (minx_a + maxx_a) > (minx_b + maxx_b):
                        return 1
                    else:
                        return 0
                else:
                    if (miny_a + maxy_a) < (miny_b + maxy_b):
                        return -1
                    elif (miny_a + maxy_a) > (miny_b + maxy_b):
                        return 1
                    else:
                        return 0

    # print(blocks)
    # print(cmp_pts_udlr(blocks[0], blocks[1]))
    blocks.sort(key=cmp_to_key(cmp_pts_udlr))
    # print(blocks)
    # exit(0)


def pts2poly(pts):
    new_pts = [(pts[k], pts[k + 1]) for k in range(0, len(pts), 2)]
    return Polygon(new_pts)


def pts_intersection_rate(src, tgt):
    src_poly, tgt_poly = pts2poly(src), pts2poly(tgt)
    src_area = src_poly.area
    inter_area = src_poly.intersection(tgt_poly).area
    return inter_area / src_area

