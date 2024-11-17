#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：lore_coco_utils
# @Author  ：cycloneboy
# @Date    ：20xx/11/7 16:30
import random

import numpy as np
import cv2
import os
import math

from pdftable.model.lore.configuration_lore import LoreConfig
from pdftable.model.lore.lineless_table_process import get_affine_transform_upper_left, get_affine_transform, \
    affine_transform


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


class LoreCocoUtils:
    def __init__(self, config: LoreConfig,
                 split='train'):
        super().__init__()
        self.opt = config
        self.split = split

        self.max_objs = 300
        self.max_pairs = 900
        self.max_cors = 1200
        self.num_classes = 2
        self.down_ratio = 4

        self.class_name = ['__background__', 'center', 'corner']
        self._valid_ids = [1, 2]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                             dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835],
                            dtype=np.float32).reshape(1, 1, 3)

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_border_upper_left(self, border, size):
        i = 1
        while size / 2 - border // i <= border // i:
            i *= 2
        return border // i

    def _get_radius(self, r, w, h):
        if w > h:
            k = float(w) / float(h)
        else:
            k = float(h) / float(w)
        ratio = k ** 0.5
        if w > h:
            r_w = r * ratio
            r_h = r
        else:
            r_h = r * ratio
            r_w = r
        return int(r_w), int(r_h)

    def color(self, image, p, magnitude):
        if np.random.randint(0, 10) > p * 10:
            return image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        img_float, bgr_img_float = img.astype(float), bgr_img.astype(float)
        diff = img_float - bgr_img_float
        diff = diff * magnitude
        diff_img_ = diff + bgr_img_float
        diff_img_ = diff_img_.astype(np.uint8)
        diff_img_ = np.array(diff_img_)
        diff_img_ = np.clip(diff_img_, 0, 255)
        diff_img_ = cv2.cvtColor(diff_img_, cv2.COLOR_BGR2RGB)
        diff_img_ = cv2.cvtColor(diff_img_, cv2.COLOR_RGB2BGR)
        return diff_img_

    def rotate(self, p, magnitude):
        if np.random.randint(0, 10) > p * 10:
            return 0
        rot = np.random.randint(magnitude[0], magnitude[1])
        return rot

    def hisEqulColor(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return img

    def _judge(self, box):
        countx = len(list(set([box[0], box[2], box[4], box[6]])))
        county = len(list(set([box[1], box[3], box[5], box[7]])))
        if countx < 2 or county < 2:
            return False

        return True

    def _get_Center(self, point):
        x1 = point[0]
        y1 = point[1]
        x3 = point[2]
        y3 = point[3]
        x2 = point[4]
        y2 = point[5]
        x4 = point[6]
        y4 = point[7]
        w1 = math.sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3))
        w2 = math.sqrt((x2 - x4) * (x2 - x4) + (y2 - y4) * (y2 - y4))
        h1 = math.sqrt((x1 - x4) * (x1 - x4) + (y1 - y4) * (y1 - y4))
        h2 = math.sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3))
        nw = min(w1, w2)
        nh = min(h1, h2)
        x_dev = x4 * y2 - x4 * y1 - x3 * y2 + x3 * y1 - x2 * y4 + x2 * y3 + x1 * y4 - x1 * y3
        y_dev = y4 * x2 - y4 * x1 - y3 * x2 + x1 * y3 - y2 * x4 + y2 * x3 + y1 * x4 - y1 * x3
        c_x = 0
        c_y = 0
        if x_dev != 0:
            c_x = (
                          y3 * x4 * x2 - y4 * x3 * x2 - y3 * x4 * x1 + y4 * x3 * x1 - y1 * x2 * x4 + y2 * x1 * x4 + y1 * x2 * x3 - y2 * x1 * x3) / x_dev
        if y_dev != 0:
            c_y = (
                          -y3 * x4 * y2 + y4 * x3 * y2 + y3 * x4 * y1 - y4 * x3 * y1 + y1 * x2 * y4 - y1 * x2 * y3 - y2 * x1 * y4 + y2 * x1 * y3) / y_dev
        return nw, nh, c_x, c_y

    def _rank(self, bbox, cter, file_name):
        init_bbox = bbox
        # bbox = list(map(float,bbox))
        continue_sign = False
        bbox = [bbox[0:2], bbox[2:4], bbox[4:6], bbox[6:8]]
        bbox_ = np.array(bbox) - np.array(cter)
        i, box_y, sign = 0, [], 'LT'
        choice = []
        for box in bbox_:
            if box[0] < 0 and box[1] < 0:
                box_y.append(box)
                choice.append(i)
            i = i + 1
        if len(choice) == 0:
            i, box_y, sign = 0, [], 'RT'
            for box in bbox_:
                if box[0] > 0 and box[1] < 0:
                    box_y.append(box)
                    choice.append(i)
                i = i + 1
        if sign == 'LT':
            ylist = np.array(box_y)[:, 1]
            # index = list(ylist).index(max(ylist))
            index = list(ylist).index(min(ylist))
        elif sign == 'RT':
            try:
                xlist = np.array(box_y)[:, 0]
            except Exception as e:
                print("center:", cter, "box:", init_bbox, "box_y:", box_y)
                return True, bbox
            index = list(xlist).index(min(xlist))

        index = choice[index]
        p = []
        for i in range(4):
            if i + index < 4:
                p.append(bbox[index + i])
            else:
                p.append(bbox[index + i - 4])
        return continue_sign, [p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], p[3][0], p[3][1]]

    def get_image_eval(self,image_path,image_id):
        img = cv2.imread(image_path)
        img_size = img.shape

        height, width = img.shape[0], img.shape[1]

        if self.opt.upper_left:
            c = np.array([0, 0], dtype=np.float32)
        else:
            c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.opt.resolution

        rot = 0
        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio

        if self.opt.upper_left:
            trans_input = get_affine_transform_upper_left(c, s, rot, [input_w, input_h])
        else:
            trans_input = get_affine_transform(c, s, rot, [input_w, input_h])

        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        ret = {'pixel_values': inp,}

        if not self.split == 'train':
            # gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
            #     np.zeros((1, 10), dtype=np.float32)
            meta = {'c': c, 's': s,
                    'out_height': output_h, 'out_width': output_w,
                    'input_height': input_h, 'input_width': input_w,
                    # 'rot': rot,
                    # 'gt_det': gt_det,
                    'img_id': image_id}
            ret['meta'] = meta
            # ret['inputs'] = image_path

        return ret


    def get_label_from_coco(self, image_path, anns, image_id):
        num_objs = min(len(anns), self.max_objs)
        num_cors = self.max_cors

        img = cv2.imread(image_path)
        img_size = img.shape

        height, width = img.shape[0], img.shape[1]

        if self.opt.upper_left:
            c = np.array([0, 0], dtype=np.float32)
        else:
            c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.opt.resolution

        flipped = False
        if self.split == 'train':
            if self.opt.upper_left:
                c = np.array([0, 0], dtype=np.float32)
            else:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

        rot = 0
        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio

        if self.opt.upper_left:
            trans_input = get_affine_transform_upper_left(c, s, rot, [input_w, input_h])
            trans_output = get_affine_transform_upper_left(c, s, rot, [output_w, output_h])
            trans_output_mk = get_affine_transform_upper_left(c, s, rot, [output_w, output_h])
        else:
            trans_input = get_affine_transform(c, s, rot, [input_w, input_h])
            trans_output = get_affine_transform(c, s, rot, [output_w, output_h])
            trans_output_mk = get_affine_transform(c, s, rot, [output_w, output_h])

        num_classes = self.num_classes

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 8), dtype=np.float32)
        reg = np.zeros((self.max_objs * 5, 2), dtype=np.float32)
        st = np.zeros((self.max_cors, 8), dtype=np.float32)
        hm_ctxy = np.zeros((self.max_objs, 2), dtype=np.float32)
        hm_ind = np.zeros((self.max_objs), dtype=np.int64)
        hm_mask = np.zeros((self.max_objs), dtype=np.uint8)
        mk_ind = np.zeros((self.max_cors), dtype=np.int64)
        mk_mask = np.zeros((self.max_cors), dtype=np.uint8)
        reg_ind = np.zeros((self.max_objs * 5), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs * 5), dtype=np.uint8)
        ctr_cro_ind = np.zeros((self.max_objs * 4), dtype=np.int64)
        log_ax = np.zeros((self.max_objs, 4), dtype=np.float32)
        cc_match = np.zeros((self.max_objs, 4), dtype=np.int64)
        h_pair_ind = np.zeros((self.max_pairs), dtype=np.int64)
        v_pair_ind = np.zeros((self.max_pairs), dtype=np.int64)
        draw_gaussian = draw_umich_gaussian
        gt_det = []
        corList = []
        point = []
        pair_mark = 0
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        for k in range(num_objs):
            ann = anns[k]

            seg_mask = ann['segmentation'][0]  # [[351.0, 73.0, 172.0, 70.0, 174.0, 127.0, 351.0, 129.0, 351.0, 73.0]]
            x1, y1 = seg_mask[0], seg_mask[1]
            x2, y2 = seg_mask[2], seg_mask[3]
            x3, y3 = seg_mask[4], seg_mask[5]
            x4, y4 = seg_mask[6], seg_mask[7]

            CorNer = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
            boxes = [[CorNer[0], CorNer[1]], [CorNer[2], CorNer[3]],
                     [CorNer[4], CorNer[5]], [CorNer[6], CorNer[7]]]
            cls_id = int(self.cat_ids[ann['category_id']])

            if flipped:
                CorNer[[0, 2, 4, 6]] = width - CorNer[[2, 0, 6, 4]] - 1

            CorNer[0:2] = affine_transform(CorNer[0:2], trans_output_mk)
            CorNer[2:4] = affine_transform(CorNer[2:4], trans_output_mk)
            CorNer[4:6] = affine_transform(CorNer[4:6], trans_output_mk)
            CorNer[6:8] = affine_transform(CorNer[6:8], trans_output_mk)
            CorNer[[0, 2, 4, 6]] = np.clip(CorNer[[0, 2, 4, 6]], 0, output_w - 1)
            CorNer[[1, 3, 5, 7]] = np.clip(CorNer[[1, 3, 5, 7]], 0, output_h - 1)
            if not self._judge(CorNer):
                continue

            maxx = max([CorNer[2 * I] for I in range(0, 4)])
            minx = min([CorNer[2 * I] for I in range(0, 4)])
            maxy = max([CorNer[2 * I + 1] for I in range(0, 4)])
            miny = min([CorNer[2 * I + 1] for I in range(0, 4)])
            h, w = maxy - miny, maxx - minx  # bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:

                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                ct = np.array([(maxx + minx) / 2.0, (maxy + miny) / 2.0], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                draw_gaussian(hm[cls_id], ct_int, radius)

                for i in range(4):
                    Cor = np.array([CorNer[2 * i], CorNer[2 * i + 1]], dtype=np.float32)
                    Cor_int = Cor.astype(np.int32)
                    Cor_key = str(Cor_int[0]) + "_" + str(Cor_int[1])
                    if Cor_key not in corList:

                        corNum = len(corList)

                        corList.append(Cor_key)
                        reg[self.max_objs + corNum] = np.array([abs(Cor[0] - Cor_int[0]), abs(Cor[1] - Cor_int[1])])
                        mk_ind[corNum] = Cor_int[1] * output_w + Cor_int[0]
                        cc_match[k][i] = mk_ind[corNum]
                        reg_ind[self.max_objs + corNum] = Cor_int[1] * output_w + Cor_int[0]
                        mk_mask[corNum] = 1
                        reg_mask[self.max_objs + corNum] = 1
                        draw_gaussian(hm[num_classes - 1], Cor_int, 2)
                        st[corNum][i * 2:(i + 1) * 2] = np.array([Cor[0] - ct[0], Cor[1] - ct[1]])
                        ctr_cro_ind[4 * k + i] = corNum * 4 + i

                    else:
                        index_of_key = corList.index(Cor_key)
                        cc_match[k][i] = mk_ind[index_of_key]
                        st[index_of_key][i * 2:(i + 1) * 2] = np.array([Cor[0] - ct[0], Cor[1] - ct[1]])
                        ctr_cro_ind[4 * k + i] = index_of_key * 4 + i

                wh[k] = ct[0] - 1. * CorNer[0], ct[1] - 1. * CorNer[1], \
                        ct[0] - 1. * CorNer[2], ct[1] - 1. * CorNer[3], \
                        ct[0] - 1. * CorNer[4], ct[1] - 1. * CorNer[5], \
                        ct[0] - 1. * CorNer[6], ct[1] - 1. * CorNer[7]

                hm_ind[k] = ct_int[1] * output_w + ct_int[0]
                hm_mask[k] = 1
                reg_ind[k] = ct_int[1] * output_w + ct_int[0]
                reg_mask[k] = 1
                reg[k] = ct - ct_int
                hm_ctxy[k] = ct[0], ct[1]

                log_ax[k] = ann['logic_axis'][0][0], ann['logic_axis'][0][1], ann['logic_axis'][0][2], \
                    ann['logic_axis'][0][3]

                gt_det.append([ct[0] - 1. * CorNer[0], ct[1] - 1. * CorNer[1],
                               ct[0] - 1. * CorNer[2], ct[1] - 1. * CorNer[3],
                               ct[0] - 1. * CorNer[4], ct[1] - 1. * CorNer[5],
                               ct[0] - 1. * CorNer[6], ct[1] - 1. * CorNer[7], 1, cls_id])

        hm_mask_v = hm_mask.reshape(1, hm_mask.shape[0])

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train':
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        ret = {'pixel_values': inp, 'hm': hm, 'hm_ind': hm_ind, 'hm_mask': hm_mask, 'mk_ind': mk_ind, 'mk_mask': mk_mask,
               'reg': reg, 'reg_ind': reg_ind, 'reg_mask': reg_mask,
               'wh': wh, 'st': st, 'ctr_cro_ind': ctr_cro_ind, 'cc_match': cc_match, 'hm_ctxy': hm_ctxy,
               'logic': log_ax, 'h_pair_ind': h_pair_ind, 'v_pair_ind': v_pair_ind}

        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 10), dtype=np.float32)
            meta = {'c': c, 's': s,
                    'out_height': output_h, 'out_width': output_w,
                    'input_height': input_h, 'input_width': input_w,
                    # 'rot': rot,
                    # 'gt_det': gt_det,
                    'img_id': image_id}
            ret['meta'] = meta
            # ret['inputs'] = image_path
        return ret
