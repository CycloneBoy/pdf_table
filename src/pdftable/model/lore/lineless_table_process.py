# ------------------------------------------------------------------------------
# Part of implementation is adopted from CenterNet,
# made publicly available under the MIT License at https://github.com/xingyizhou/CenterNet.git
# ------------------------------------------------------------------------------
import time

import cv2
import numpy as np
import shapely
import torch
import torch.nn as nn
from shapely.geometry import MultiPoint, Point, Polygon

__all__ = [
    "load_lore_model"
]


def _gather_feat(feat, ind, mask=None):
    # mandatory
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    # mandatory
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _get_4ps_feat(cc_match, output):
    run_device = cc_match.device
    # mandatory
    if isinstance(output, dict):
        feat = output['cr']
    else:
        feat = output
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.contiguous().view(feat.size(0), -1, feat.size(3))
    feat = feat.unsqueeze(3).expand(
        feat.size(0), feat.size(1), feat.size(2), 4)

    dim = feat.size(2)

    cc_match = cc_match.unsqueeze(2).expand(
        cc_match.size(0), cc_match.size(1), dim, cc_match.size(2))
    if not (isinstance(output, dict)):
        cc_match = torch.where(
            cc_match < feat.shape[1], cc_match, (feat.shape[0] - 1)
            * torch.ones(cc_match.shape, dtype=torch.int64, device=run_device))
        cc_match = torch.where(
            cc_match >= 0, cc_match,
            torch.zeros(cc_match.shape, dtype=torch.int64, device=run_device))
    feat = feat.gather(1, cc_match)
    return feat


def _nms(heat, name, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    # save_map(hmax.cpu().numpy()[0],name)
    keep = (hmax == heat).float()
    return heat * keep, keep


def _topk(scores, K=40):
    run_device = scores.device
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (
                torch.Tensor([height]).to(torch.int64).cuda() * torch.Tensor([width]).to(torch.int64).cuda())
    topk_ys = (topk_inds / torch.Tensor([width]).cuda()).int().float()
    topk_xs = (topk_inds % torch.Tensor([width]).to(torch.int64).cuda()).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1),
                             topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def corner_decode(mk, st_reg, mk_reg=None, K=400):
    batch, cat, height, width = mk.size()
    mk, keep = _nms(mk, 'mk.0.maxpool')
    scores, inds, clses, ys, xs = _topk(mk, K=K)
    if mk_reg is not None:
        reg = _tranpose_and_gather_feat(mk_reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    scores = scores.view(batch, K, 1)
    st_Reg = _tranpose_and_gather_feat(st_reg, inds)
    bboxes_vec = [
        xs - st_Reg[..., 0:1], ys - st_Reg[..., 1:2], xs - st_Reg[..., 2:3],
        ys - st_Reg[..., 3:4], xs - st_Reg[..., 4:5], ys - st_Reg[..., 5:6],
        xs - st_Reg[..., 6:7], ys - st_Reg[..., 7:8]
    ]
    bboxes = torch.cat(bboxes_vec, dim=2)
    corner_dict = {
        'scores': scores,
        'inds': inds,
        'ys': ys,
        'xs': xs,
        'gboxes': bboxes
    }
    return scores, inds, ys, xs, bboxes, corner_dict


def ctdet_4ps_decode(heat, wh, ax, cr, corner_dict=None, reg=None, cat_spec_wh=False, K=100, wiz_rev=False):
    # if wiz_rev :
    #     print('Grouping and Parsing ...')
    batch, cat, height, width = heat.size()
    run_device = heat.device
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat, keep = _nms(heat, 'hm.0.maxpool')

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    ax = _tranpose_and_gather_feat(ax, inds)

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

    rev_time_s1 = time.time()
    if wiz_rev:
        bboxes_rev = bboxes.clone()
        bboxes_cpu = bboxes.clone().cpu()

        gboxes = corner_dict['gboxes']
        gboxes_cpu = gboxes.cpu()

        num_bboxes = bboxes.shape[1]
        num_gboxes = gboxes.shape[1]

        corner_xs = corner_dict['xs']
        corner_ys = corner_dict['ys']
        corner_scores = corner_dict['scores']

        for i in range(num_bboxes):
            if scores[0, i, 0] >= 0.2:
                count = 0  # counting the number of ends of st head in bbox i
                for j in range(num_gboxes):
                    if corner_scores[0, j, 0] >= 0.3:
                        # here comes to one pair of valid bbox and gbox
                        # step1 is there an overlap

                        bbox = bboxes_cpu[0, i, :]
                        gbox = gboxes_cpu[0, j, :]
                        # rev_time_s3 = time.time()
                        if is_group_faster_faster(bbox, gbox):
                            # step2 find which corner point to refine, and do refine
                            cr_x = corner_xs[0, j, 0]
                            cr_y = corner_ys[0, j, 0]

                            ind4ps = find4ps(bbox, cr_x, cr_y)
                            if bboxes_rev[0, i, 2 * ind4ps] == bboxes[0, i, 2 * ind4ps] and bboxes_rev[
                                0, i, 2 * ind4ps + 1] == bboxes[0, i, 2 * ind4ps + 1]:
                                # first_shift
                                count = count + 1
                                bboxes_rev[0, i, 2 * ind4ps] = cr_x
                                bboxes_rev[0, i, 2 * ind4ps + 1] = cr_y
                            else:
                                origin_x = bboxes[0, i, 2 * ind4ps]
                                origin_y = bboxes[0, i, 2 * ind4ps + 1]

                                old_x = bboxes_rev[0, i, 2 * ind4ps]
                                old_y = bboxes_rev[0, i, 2 * ind4ps + 1]

                                if dist(origin_x, origin_y, old_x, old_y) >= dist(origin_x, origin_y, cr_x, cr_y):
                                    count = count + 1
                                    bboxes_rev[0, i, 2 * ind4ps] = cr_x
                                    bboxes_rev[0, i, 2 * ind4ps + 1] = cr_y
                                else:
                                    continue
                        else:

                            continue
                    else:
                        break
                if count <= 2:
                    scores[0, i, 0] = scores[0, i, 0] * 0.4
            else:
                break

    if wiz_rev:

        cc_match = torch.cat([(bboxes_rev[:, :, 0:1]) + width * torch.round(bboxes_rev[:, :, 1:2]),
                              (bboxes_rev[:, :, 2:3]) + width * torch.round(bboxes_rev[:, :, 3:4]),
                              (bboxes_rev[:, :, 4:5]) + width * torch.round(bboxes_rev[:, :, 5:6]),
                              (bboxes_rev[:, :, 6:7]) + width * torch.round(bboxes_rev[:, :, 7:8])], dim=2)

    else:
        cc_match = torch.cat([(xs - wh[..., 0:1]) + width * torch.round(ys - wh[..., 1:2]),
                              (xs - wh[..., 2:3]) + width * torch.round(ys - wh[..., 3:4]),
                              (xs - wh[..., 4:5]) + width * torch.round(ys - wh[..., 5:6]),
                              (xs - wh[..., 6:7]) + width * torch.round(ys - wh[..., 7:8])], dim=2)

    cc_match = torch.round(cc_match).to(torch.int64)
    cc_match = cc_match.to(run_device)

    cr_feat = _get_4ps_feat(cc_match, cr)
    cr_feat = cr_feat.sum(axis=3)
    if wiz_rev:
        detections = torch.cat([bboxes_rev, scores, clses], dim=2).to(run_device)
        _, sorted_ind = torch.sort(scores, descending=True, dim=1)
        sorted_inds = sorted_ind.expand(detections.size(0), detections.size(1), detections.size(2))
        detections = detections.gather(1, sorted_inds)
        sorted_inds2 = sorted_ind.expand(detections.size(0), detections.size(1), ax.size(2))
        ax = ax.gather(1, sorted_inds2)
    else:

        detections = torch.cat([bboxes, scores, clses], dim=2).to(run_device)

    return detections, keep, ax, cr_feat


def wireless_decode(heat, wh, ax, cr, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat, keep = _nms(heat, 'hm.0.maxpool')

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    ax = _tranpose_and_gather_feat(ax, inds)

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

    cc_match = torch.cat([(xs - wh[..., 0:1]) + width * torch.round(ys - wh[..., 1:2]),
                          (xs - wh[..., 2:3]) + width * torch.round(ys - wh[..., 3:4]),
                          (xs - wh[..., 4:5]) + width * torch.round(ys - wh[..., 5:6]),
                          (xs - wh[..., 6:7]) + width * torch.round(ys - wh[..., 7:8])], dim=2)

    cc_match = torch.round(cc_match).to(torch.int64)

    cr_feat = _get_4ps_feat(cc_match, cr)
    cr_feat = cr_feat.sum(axis=3)

    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections, keep, ax, cr_feat


def find4ps(bbox, x, y):
    xs = torch.Tensor([bbox[0], bbox[2], bbox[4], bbox[6]]).cuda()
    ys = torch.Tensor([bbox[1], bbox[3], bbox[5], bbox[7]]).cuda()

    dx = xs - x
    dy = ys - y

    l = dx ** 2 + dy ** 2
    return torch.argmin(l)


def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    l = dx ** 2 + dy ** 2
    return l


def rect_inter(b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2):
    if (b1_x1 <= b2_x1 and b2_x1 <= b1_x2) or (b1_x1 <= b2_x2 and b2_x2 <= b1_x2):
        if (b1_y1 <= b2_y1 and b2_y1 <= b1_y2) or (b1_y1 <= b2_y2 and b2_y2 <= b1_y2):
            return True
        else:
            return False
    else:
        return False


def is_group_faster_faster(bbox, gbox):
    bbox = bbox.view(4, 2)
    gbox = gbox.view(4, 2)

    bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax = bbox[:, 0].min(), bbox[:, 0].max(), bbox[:, 1].min(), bbox[:,
                                                                                                       1].max()  # min(bbox_xs), max(bbox_xs), min(bbox_ys), max(bbox_ys)
    gbox_xmin, gbox_xmax, gbox_ymin, gbox_ymax = gbox[:, 0].min(), gbox[:, 0].max(), gbox[:, 1].min(), gbox[:, 1].max()

    if bbox_xmin > gbox_xmax or gbox_xmin > bbox_xmax or bbox_ymin > gbox_ymax or gbox_ymin > bbox_ymax:
        return False
    else:
        bpoly = Polygon(bbox)

        flag = 0
        for i in range(4):
            p = Point(gbox[i])
            if p.within(bpoly):
                flag = 1
                break
        if flag == 0:
            return False
        else:
            return True


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


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
    # logger.info(f"dst_w = {dst_w}, dst_h = {dst_h}")

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift  # [0,0] #
    src[1, :] = center + src_dir + scale_tmp * shift  # scale #
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]  # [0,0] #
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5],
                         np.float32) + dst_dir  # output_size #

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_affine_transform_upper_left(center,
                                    scale,
                                    rot,
                                    output_size,
                                    shift=np.array([0, 0], dtype=np.float32),
                                    inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    dst[0, :] = [0, 0]
    if center[0] < center[1]:
        src[1, :] = [scale[0], center[1]]
        dst[1, :] = [output_size[0], 0]
    else:
        src[1, :] = [center[0], scale[0]]
        dst[1, :] = [0, output_size[0]]
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_preds(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def transform_preds_upper_left(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)

    trans = get_affine_transform_upper_left(
        center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def ctdet_4ps_post_process(dets, c, s, h, w, num_classes, rot=0):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []

    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, 0:2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h), rot)
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h), rot)
        dets[i, :, 4:6] = transform_preds(dets[i, :, 4:6], c[i], s[i], (w, h), rot)
        dets[i, :, 6:8] = transform_preds(dets[i, :, 6:8], c[i], s[i], (w, h), rot)
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :8].astype(np.float32),
                dets[i, inds, 8:9].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def ctdet_4ps_post_process_upper_left(dets, c, s, h, w, num_classes, rot=0):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, 0:2] = transform_preds_upper_left(dets[i, :, 0:2], c[i],
                                                     s[i], (w, h), rot)
        dets[i, :, 2:4] = transform_preds_upper_left(dets[i, :, 2:4], c[i],
                                                     s[i], (w, h), rot)
        dets[i, :, 4:6] = transform_preds_upper_left(dets[i, :, 4:6], c[i],
                                                     s[i], (w, h), rot)
        dets[i, :, 6:8] = transform_preds_upper_left(dets[i, :, 6:8], c[i],
                                                     s[i], (w, h), rot)
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            tmp_top_pred = [
                dets[i, inds, :8].astype(np.float32),
                dets[i, inds, 8:9].astype(np.float32)
            ]
            top_preds[j + 1] = np.concatenate(tmp_top_pred, axis=1).tolist()
        ret.append(top_preds)
    return ret


def ctdet_corner_post_process(corner_st_reg, c, s, h, w, num_classes):
    for i in range(corner_st_reg.shape[0]):
        corner_st_reg[i, :, 0:2] = transform_preds(corner_st_reg[i, :, 0:2],
                                                   c[i], s[i], (w, h))
        corner_st_reg[i, :, 2:4] = transform_preds(corner_st_reg[i, :, 2:4],
                                                   c[i], s[i], (w, h))
        corner_st_reg[i, :, 4:6] = transform_preds(corner_st_reg[i, :, 4:6],
                                                   c[i], s[i], (w, h))
        corner_st_reg[i, :, 6:8] = transform_preds(corner_st_reg[i, :, 6:8],
                                                   c[i], s[i], (w, h))
        corner_st_reg[i, :, 8:10] = transform_preds(corner_st_reg[i, :, 8:10],
                                                    c[i], s[i], (w, h))
    return corner_st_reg


def merge_outputs(detections):
    # thresh_conf, thresh_min, thresh_max = 0.1, 0.5, 0.7
    num_classes, max_per_image = 2, 3000
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate([detection[j] for detection in detections],
                                    axis=0).astype(np.float32)
    scores = np.hstack([results[j][:, 8] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 8] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def filter(results, logi, ps, vis_thresh=0.15):
    # this function select boxes
    batch_size, feat_dim = logi.shape[0], logi.shape[2]
    num_valid = sum(results[1][:, 8] >= vis_thresh)

    slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
    slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.int32)
    for i in range(batch_size):
        for j in range(num_valid):
            slct_logi[i, j, :] = logi[i, j, :].detach().cpu()
            slct_dets[i, j, :] = ps[i, j, :].detach().cpu()

    dtype = logi.dtype
    device = logi.device
    return torch.tensor(slct_logi, dtype=dtype, device=device), torch.tensor(slct_dets, dtype=dtype, device=device)


def normalized_ps(ps, vocab_size):
    ps = torch.round(ps).to(torch.int64)
    ps = torch.where(ps < vocab_size, ps, (vocab_size - 1) * torch.ones(ps.shape).to(torch.int64).cuda())
    ps = torch.where(ps >= 0, ps, torch.zeros(ps.shape).to(torch.int64).cuda())
    return ps


def process_detect_output(output, meta, upper_left=True, wiz_rev=False, vis_thresh=0.15):
    K, MK = 3000, 5000
    num_classes = 2
    scale = 1.0

    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']
    st = output['st']
    ax = output['ax']
    cr = output['cr']

    run_meta = meta.cpu().numpy()[0]
    meta_c = [run_meta[:2]]
    meta_s = [run_meta[2]]
    meta_out_height = run_meta[5]
    meta_out_width = run_meta[6]


    # run_meta = meta
    # meta_c = run_meta["c"].cpu().numpy()
    # meta_s = run_meta["s"].cpu().numpy()
    # meta_out_height = run_meta["out_height"].cpu().numpy()[0]
    # meta_out_width = run_meta["out_width"].cpu().numpy()[0]

    # logger.info(f"meta: {meta}")

    scores, inds, ys, xs, st_reg, corner_dict = corner_decode(
        hm[:, 1:2, :, :], st, reg, K=MK)
    dets, keep, logi, cr = ctdet_4ps_decode(
        hm[:, 0:1, :, :], wh, ax, cr, corner_dict, reg=reg, K=K, wiz_rev=wiz_rev)
    corner_output = np.concatenate(
        (np.transpose(xs.detach().cpu()), np.transpose(ys.detach().cpu()), np.array(st_reg.detach().cpu()), np.transpose(scores.detach().cpu())), axis=2)

    raw_dets = dets
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])

    if upper_left:
        dets = ctdet_4ps_post_process_upper_left(dets.copy(),
                                                 meta_c,
                                                 meta_s, meta_out_height,
                                                 meta_out_width, num_classes)
    else:
        dets = ctdet_4ps_post_process(
            dets.copy(), meta_c, meta_s,
            meta_out_height, meta_out_width, num_classes)

    corner_st = ctdet_corner_post_process(
        corner_output.copy(), meta_c, meta_s,
        meta_out_height, meta_out_width, num_classes)

    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 9)
        dets[0][j][:, :8] /= scale
    dets = dets[0]
    detections = [dets]

    logi = logi + cr
    results = merge_outputs(detections)
    slct_logi_feat, slct_dets_feat = filter(results, logi, raw_dets[:, :, :8], vis_thresh=vis_thresh)
    slct_dets_feat = normalized_ps(slct_dets_feat, 256)

    return slct_logi_feat, slct_dets_feat, results, corner_st[0]


def process_logic_output(logi):
    logi_floor = logi.floor()
    dev = logi - logi_floor
    logi = torch.where(dev > 0.5, logi_floor + 1, logi_floor)

    return logi


def load_lore_model(model, checkpoint, mtype=None, strict=False):
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            if mtype is not None:
                if k.startswith('model'):
                    state_dict[k[6:]] = state_dict_[k]
                if k.startswith('processor'):
                    state_dict[k[10:]] = state_dict_[k]
                else:
                    # print(f"ignore param : {k}")
                    continue
            else:
                state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(k, model_state_dict[k].shape,
                                               state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=strict)

    return model


def ps_convert_minmax(results, num_classes):
    detection = {}
    for j in range(1, num_classes + 1):
        detection[j] = []
    for j in range(1, num_classes + 1):
        for bbox in results[j]:
            minx = min(bbox[0], bbox[2], bbox[4], bbox[6])
            miny = min(bbox[1], bbox[3], bbox[5], bbox[7])
            maxx = max(bbox[0], bbox[2], bbox[4], bbox[6])
            maxy = max(bbox[1], bbox[3], bbox[5], bbox[7])
            detection[j].append([minx, miny, maxx, maxy, bbox[-1]])
    for j in range(1, num_classes + 1):
        detection[j] = np.array(detection[j])
    return detection


def _get_wh_feat(ind, output, ttype):
    width = output['hm'].shape[2]
    xs = (ind % width).unsqueeze(2).int().float()
    ys = (ind // width).unsqueeze(2).int().float()
    if ttype == 'gt':
        wh = output['wh']
    elif ttype == 'pred':
        wh = _tranpose_and_gather_feat(output['wh'], ind)
    ct = torch.cat([xs, ys, xs, ys, xs, ys, xs, ys], dim=2)
    bbx = ct - wh

    return bbx

def _normalized_ps(ps, vocab_size):
  ps = torch.round(ps).to(torch.int64)
  ps = torch.where(ps < vocab_size, ps, (vocab_size-1) * torch.ones(ps.shape).to(torch.int64).cuda())
  ps = torch.where(ps >= 0, ps, torch.zeros(ps.shape).to(torch.int64).cuda())
  return ps
