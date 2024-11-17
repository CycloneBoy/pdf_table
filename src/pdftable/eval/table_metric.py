#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_metric
# @Author  ：cycloneboy
# @Date    ：20xx/11/6 18:29
from dataclasses import dataclass

from pdftable.utils import logger
from pdftable.utils.eval.eval_utils import EvalUtils


@dataclass
class TableWtwComputeMetric:
    predict_dir: str = ""
    label_path: str = ""
    metric_dir: str = ""

    def __call__(self, eval_pred):
        # predictions, labels = eval_pred
        # pred = np.argmax(predictions, axis=1)

        table_dict, error_list = EvalUtils.load_table_pred_and_label(predict_path=self.predict_dir,
                                                                     label_path=self.label_path)

        metric = EvalUtils.eval_table(table_dict=table_dict,
                                      error_list=error_list,
                                      output_dir=self.metric_dir)

        result = {
            "acc": metric["acc"],
            "axis_ture_radio": metric["axis_ture_radio"],
            "axis_ture_total": metric["axis_ture_total"],
            "det_precision": metric["det_precision"],
            "det_recall": metric["det_recall"],
            "det_f1": metric["f1"],
            "det_bbox_acc": metric["bbox_acc"],
            "bbox_true_total": metric["bbox_true_total"],
            "total": metric["total"],
        }

        logger.info(f"metric_result: {result}")
        # logger.info(f"Evaluation Precision: {precision:.5f}| Recall: {recall:.5f} | F1: {f1:.5f}")

        return result


class TableMetric(object):

    @staticmethod
    def compute_metrics(eval_pred):
        """
        计算metric

        :param eval_pred:
        :return:
        """
        predictions, labels = eval_pred
        # pred = np.argmax(predictions, axis=1)

        result = {
            "acc": 0.0,
            "auc_score": 0.0,
            # "f1": f1,
            # "precision": precision,
            # "recall": recall,
            "class_report_dict": 0.0,
            "confusion_matrix": 0.0,
        }

        logger.info(f"metric_result: {result}")
        # logger.info(f"Evaluation Precision: {precision:.5f}| Recall: {recall:.5f} | F1: {f1:.5f}")

        return result
