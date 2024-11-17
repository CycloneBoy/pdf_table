#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：eval_utils
# @Author  ：cycloneboy
# @Date    ：20xx/11/4 15:28
import os
import time
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pycocotools.coco as coco
from tqdm import tqdm

from pdftable.entity import TableEval
from pdftable.utils import BaseUtil, logger, FileUtils, TimeUtils, CommonUtils

EMPTY = 'empty'
NULL = 'null'


class PairTable:
    def __init__(self, pred_table: TableEval, gt_table: TableEval, iou_threshold=0.5):
        self.gt_list = gt_table.ulist
        self.pred_list = pred_table.ulist
        self.iou_threshold = iou_threshold

        self.match_list = []
        self.matching()

    def matching(self):
        for tunit in self.gt_list:
            if_find = 0
            for sunit in self.pred_list:
                if self.compute_iou(tunit.bbox, sunit.bbox) >= self.iou_threshold:
                    self.match_list.append(sunit)
                    if_find = 1
                    break
            if if_find == 0:
                self.match_list.append(EMPTY)

    def eval_bbox(self, ):
        tp = self.get_box_match_total()

        ap = len(self.pred_list)
        at = len(self.gt_list)

        recall = tp / at if at > 0 else NULL
        precision = tp / ap if ap > 0 else NULL

        return precision, recall, ap - tp, at - tp

    def compute_iou(self, bbox1, bbox2):
        rec1 = (bbox1.point1[0][0], bbox1.point1[0][1], bbox1.point3[0][0], bbox1.point3[0][1])
        rec2 = (bbox2.point1[0][0], bbox2.point1[0][1], bbox2.point3[0][0], bbox2.point3[0][1])

        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])

        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)

            return S_cross / (S1 + S2 - S_cross)

    def axis_equal(self, preds: List, labels: List) -> bool:
        """
        判断预测的axis 是否正确

        :param preds:
        :param labels:
        :return:
        """
        flag = True
        for j in range(4):
            if preds[j] != labels[j]:
                flag = False
                break
        return flag

    def eval_axis(self):
        tp = self.get_box_match_total()

        truep = 0
        for i in range(len(self.gt_list)):
            pred_unit = self.match_list[i]
            if pred_unit != EMPTY:
                label_unit = self.gt_list[i]

                # all four axis are correctly predicted
                if self.axis_equal(preds=pred_unit.axis, labels=label_unit.axis):
                    truep += 1

        if len(self.gt_list) == 0:
            return NULL
        else:
            if tp == 0:
                return NULL
            else:
                return truep / tp

    def get_box_match_total(self):
        tp = 0
        for u in self.match_list:
            if u != EMPTY:
                tp += 1
        return tp


class EvalUtils(BaseUtil):

    def init(self):
        pass

    @staticmethod
    def tsr_coco_into_labels(annot_path, label_path):
        gt_center_dir = '{}/gt_center/'.format(label_path)
        gt_logi_dir = '{}/gt_logi/'.format(label_path)

        if not os.path.exists(gt_center_dir):
            os.makedirs(gt_center_dir)
        else:
            logger.info(f'current directory exists: {gt_center_dir}')
            return 0

        os.makedirs(gt_logi_dir, exist_ok=True)

        coco_data = coco.COCO(annot_path)
        images = coco_data.getImgIds()

        print('Changing COCO Labels into TXT Labels...')
        for i in tqdm(range(len(images))):
            img_id = images[i]
            file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name']
            # file_names.append(file_name)

            # using this for your dataset
            center_file = '{}/gt_center/'.format(label_path) + file_name + '.txt'
            logi_file = '{}/gt_logi/'.format(label_path) + file_name + '.txt'

            # TODO: revise the file names in the annotation of PubTabNet
            # center_file = gt_center_dir + file_name.replace('.jpg', '.png') +'.txt'
            # logi_file = gt_logi_dir + file_name.replace('.jpg', '.png') +'.txt'

            ann_ids = coco_data.getAnnIds(imgIds=[img_id])
            anns = coco_data.loadAnns(ids=ann_ids)

            fc = open(center_file, 'w')
            fl = open(logi_file, 'w')
            for j in range(len(anns)):
                ann = anns[j]
                bbox = ann['segmentation'][0]
                logi = ann['logic_axis'][0]
                for i in range(0, 3):
                    fc.write(str(bbox[2 * i]) + ',' + str(bbox[2 * i + 1]) + ';')
                    fl.write(str(int(logi[i])) + ',')

                fc.write(str(bbox[6]) + ',' + str(bbox[7]) + '\n')
                fl.write(str(int(logi[3])) + '\n')

        print('Finished: Changing COCO Labels into TXT Labels!')

    @staticmethod
    def load_table_pred_and_label(predict_path, label_path):
        gt_bbox_path = os.path.join(label_path, 'gt_center')
        gt_logi_path = os.path.join(label_path, 'gt_logi')
        bbox_path = os.path.join(predict_path, 'center')
        logi_path = os.path.join(predict_path, 'logi')

        table_dict = []
        error_list = []
        for file_name in tqdm(os.listdir(gt_bbox_path), desc="load_table"):
            if not os.path.exists(os.path.join(bbox_path, file_name)) or \
                    not os.path.exists(os.path.join(logi_path, file_name)):
                error_list.append(file_name)
                continue

            if file_name.endswith(".txt"):
                pred_table = TableEval(bbox_path, logi_path, file_name)
                gt_table = TableEval(gt_bbox_path, gt_logi_path, file_name)
                if len(pred_table.ulist) > 0 and len(gt_table.ulist) > 0:
                    table_dict.append({'file_name': file_name, 'pred_table': pred_table, 'gt_table': gt_table})
                else:
                    error_list.append(file_name)

        logger.info(f"加载测试数据集：{len(table_dict)} - {len(error_list)}")
        return table_dict, error_list

    @staticmethod
    def eval_table(table_dict: Dict, error_list: List, output_dir):
        total_label = len(table_dict)

        start_time = time.time()
        acs = []
        axis_ture = 0

        bbox_recalls = []
        bbox_precisions = []
        bbox_accs = []
        bbox_fps = defaultdict(int)
        bbox_fns = defaultdict(int)

        acs_null = 0
        recall_null = 0
        precision_null = 0
        for i in tqdm(range(total_label), desc="eval_table"):
            pair = PairTable(table_dict[i]['pred_table'], table_dict[i]['gt_table'])
            # Acc of Logical Locations
            ac = pair.eval_axis()
            if ac != NULL:
                acs.append(ac)
            else:
                acs_null += 1

            if ac == 1:
                axis_ture += 1

            # Recall of Cell Detection
            precision, recall, fp, fn = pair.eval_bbox()

            if precision != NULL:
                bbox_precisions.append(precision)
            else:
                precision_null += 1

            if recall != NULL:
                bbox_recalls.append(recall)
            else:
                recall_null += 1

            if precision == 1 and recall == 1:
                bbox_accs.append(i)
            bbox_fps[fp] += 1
            bbox_fns[fn] += 1

        logger.info(f"评估完成：{total_label} - {len(acs)}")

        result_acc = np.array(acs).mean()
        axis_ture_radio = axis_ture / total_label
        print(f'Evaluation Results | Accuracy of Logical Location: {result_acc:.2f}.'
              f' {axis_ture_radio:.2f} = {axis_ture} / {total_label}')

        det_precision = np.array(bbox_precisions).mean()
        det_recall = np.array(bbox_recalls).mean()
        f = 2 * det_precision * det_recall / (det_precision + det_recall)
        bbox_acc = len(bbox_accs) / total_label

        print(f'Evaluation Results | det_precision: {det_precision:.2f}.')
        print(f'Evaluation Results | det_recall: {det_recall:.2f}.')
        print(f'Evaluation Results | det_f1 score: {f:.2f}.')
        print(f'Evaluation Results | bbox_acc score: {bbox_acc:.2f}. = {len(bbox_accs)} / {total_label}')

        bbox_fps_sort = CommonUtils.sorted_dict(bbox_fps)
        bbox_fns_sort = CommonUtils.sorted_dict(bbox_fns)

        end_time = time.time()
        use_time = end_time - start_time
        metric = {
            "start_time": start_time,
            "end_time": end_time,
            "use_time": use_time,
            "total": len(table_dict),
            "error_total": len(error_list),
            "acs_null": acs_null,
            "recall_null": recall_null,
            "precision_null": precision_null,
            "acc": result_acc,
            "axis_ture_total": axis_ture,
            "axis_ture_radio": axis_ture_radio,
            "det_precision": det_precision,
            "det_recall": det_recall,
            "f1": f,
            "bbox_acc": bbox_acc,
            "bbox_true_total": len(bbox_accs),
            "len_acc": len(acs),
            "len_det_precision": len(bbox_precisions),
            "len_det_recall": len(bbox_recalls),
        }

        metric_file_name = f"{output_dir}/metric_{TimeUtils.now_str_short()}.json"
        FileUtils.dump_json(metric_file_name, metric)
        print(f'Evaluation metric: {metric} - {metric_file_name}')

        error_file_name = metric_file_name.replace('metric', "error")
        FileUtils.dump_json(error_file_name, error_list)

        bbox_fps_file_name = metric_file_name.replace('metric', "bbox_fps")
        FileUtils.dump_json(bbox_fps_file_name, bbox_fps_sort)

        bbox_fns_file_name = metric_file_name.replace('metric', "bbox_fns")
        FileUtils.dump_json(bbox_fns_file_name, bbox_fns_sort)

        return metric
