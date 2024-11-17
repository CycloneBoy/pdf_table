#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_text_task
# @Author  ：cycloneboy
# @Date    ：20xx/9/4 10:38

"""
OCR text task
"""
import time
import traceback
from copy import deepcopy
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from addict import Dict

from pdftable.entity import OcrCell, HtmlContentType

from pdftable.model.ocr_pdf.ocr_output import OcrSystemModelOutput

from pdftable.model import TableProcessUtils
from pdftable.model.ocr_pdf.ocr_recognition_task import OcrRecognitionTask
from pdftable.utils.ocr import OcrInferUtils, OcrCommonUtils

from pdftable.model.ocr_pdf import OCRDocumentConfig, OcrDetectionTask, OcrPdfTextTask

from pdftable.utils import TimeUtils, FileUtils, logger
from pdftable.utils.table.image_processing import PdfImageProcessor

"""
OCR TEXT TASK
"""

__all__ = [
    'OcrTextTask'
]


class OcrTextTask(object):

    def __init__(self, config: OCRDocumentConfig, debug=True,
                 output_dir=None, predictor_type="onnx",
                 delete_check_success=True, task="ocr_text", **kwargs):
        super().__init__()
        self.config = config
        self.debug = debug
        self.output_dir = output_dir
        self.predictor_type = predictor_type
        self.delete_check_success = delete_check_success
        self.task = task

        self.text_detector = None
        self.text_recognizer = None
        self.pdf_text_recognizer = OcrPdfTextTask(output_dir=self.output_dir, debug=self.debug)


    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        if self.text_detector is not None:
            self.text_detector.output_dir = output_dir

        if self.text_recognizer is not None:
            self.text_recognizer.output_dir = output_dir

        if self.pdf_text_recognizer is not None:
            self.pdf_text_recognizer.output_dir = output_dir

        FileUtils.check_file_exists(f"{output_dir}/demo.txt")

    def init_ocr_model(self):
        start = time.time()
        logger.info(f"begin to init OCR model.")
        init_model = []
        if self.text_detector is None:
            self.text_detector = OcrDetectionTask(task="ocr_detection",
                                                  model=self.config.detect_model,
                                                  backbone=self.config.detector,
                                                  thresh=self.config.thresh,
                                                  predictor_type="pytorch",
                                                  lang=self.config.lang,
                                                  debug=self.config.debug,
                                                  )
            init_model.append(self.text_detector.task)

        if self.text_recognizer is None:
            self.text_recognizer = OcrRecognitionTask(task="ocr_recognition",
                                                      model=self.config.recognizer,
                                                      task_type=self.config.task_type,
                                                      predictor_type=self.predictor_type,
                                                      lang=self.config.lang,
                                                      debug=self.config.debug,
                                                      )
            init_model.append(self.text_recognizer.task)

    def text_detection(self, image):
        start = time.time()

        if self.text_detector is None:
            self.init_ocr_model()

        result = self.text_detector(image)
        det_result = result[0]
        use_time = time.time() - start
        logger.info(f"检测耗时：{use_time:3f} s. det_result: ")

        dt_boxes = OcrInferUtils.sorted_boxes(det_result.reshape(-1, 2, 4))

        h, w = image.shape[:2]
        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)

        result = dt_boxes.reshape(-1, 8)

        metric = {
            "use_time": use_time
        }
        return result, metric

    def _text_recognition(self, image, index=0, total=0):
        """
        识别一张图片

        :param image:
        :param index:
        :return:
        """
        start = time.time()
        try:
            if self.text_recognizer is None:
                self.init_ocr_model()

            out_preds = self.text_recognizer(image)
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"识别文字异常：{e}")
            out_preds = [""]

        preds = {"text": out_preds[0]}

        use_time = time.time() - start
        # logger.info(f"识别【{index}/{total}】耗时：{use_time:3f} s. result:{preds}")

        metric = {
            "use_time": use_time
        }

        return preds, metric

    def text_recognition(self, det_result, image_full):
        """
        文字识别

        :param det_result:
        :param image_full:
        :return:
        """

        use_times = []
        output = []
        total = len(det_result)
        logger.info(f"识别开始：总共待识别：{total} 个。")
        for i in range(det_result.shape[0]):
            pts = OcrCommonUtils.order_point(det_result[i])
            image_crop = OcrCommonUtils.crop_image(image_full, pts)
            result, metric = self._text_recognition(image_crop, index=i, total=total)
            one_use_time = metric['use_time']
            use_times.append(one_use_time)

            output.append({
                "index": i + 1,
                "text": result['text'],
                "bbox": pts,
            })

        total_use_time = sum(use_times)
        avg_use_time = total_use_time / len(use_times) if len(use_times) > 0 else 0
        metric = {
            "use_time": total_use_time,
            "avg_use_time": avg_use_time,
            "total": len(use_times),
        }
        logger.info(f"识别结束：{metric}")

        return output, metric

    def pre_process_image(self, image_name, image_full):
        """
        判断图片是否需要旋转180度

        :param image_name:
        :param image_full:
        :return:
        """
        det_result, det_metric = self.text_detection(image_full, )
        ocr_result, rec_metric = self.text_recognition(det_result=det_result, image_full=image_full)

        all_texts = []
        for one_item in ocr_result:
            all_texts.append(one_item["text"])

        flag = False
        # flag = TableProcessUtils.check_pdf_text_need_rotate(all_texts)
        if flag:
            src_image = cv2.imread(image_name)
            PdfImageProcessor.rotate_image_angle_v2(image=src_image,
                                                    angle=180,
                                                    save_image_file=image_name)
            logger.info(f"pdf图片宽需要翻转，逆时针旋转180度：{image_name}")

        logger.info(f"开始图片旋转修复: {image_name}")
        try:
            rotated_image, angle = PdfImageProcessor.rotate_image(image_name, save_image_file=image_name,
                                                                  threshold_block_size=15,
                                                                  threshold_constant=-2,
                                                                  line_scale_horizontal=40,
                                                                  line_scale_vertical=40,
                                                                  iterations=0,
                                                                  diff_angle=400,
                                                                  angle_threshold=0.2
                                                                  )
            logger.info(f"完成图片旋转修复: {image_name}")
        except Exception as e:
            traceback.print_exc()

        return flag, det_result, ocr_result, det_metric, rec_metric

    def show_ocr_result(self, ocr_result: List):
        output = []
        for item in ocr_result:
            output.append([item['index'], item['text'], ','.join([str(e) for e in list(item['bbox'].reshape(-1))])])

        ocr_result_pd = pd.DataFrame(output, columns=['检测框序号', '行识别结果', '检测框坐标'])

        return ocr_result_pd

    def pdf_text_extract(self, ocr_system_output: OcrSystemModelOutput) -> Tuple[List[OcrCell], Dict]:
        """
        pdf文本提取

        :param ocr_system_output:
        :return:
        """
        start = time.time()
        text_result = self.pdf_text_recognizer(ocr_system_output)
        use_time = time.time() - start
        logger.info(f"pdf文本提取耗时：{use_time:3f} s. text_result: {len(text_result)}")

        metric = {
            "use_time": use_time
        }
        return text_result, metric


    def __call__(self, inputs, save_result=True, src_id=None, page=None, **kwargs):
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()
        run_time = TimeUtils.now_str_short()

        is_pdf = FileUtils.is_pdf_file(inputs)
        logger.info(f"数据源是{'PDF' if is_pdf else '图片'}: {inputs}")

        raw_filename = FileUtils.get_file_name(inputs)
        image_name = FileUtils.get_pdf_to_image_file_name(inputs)
        image_full = OcrCommonUtils.read_image(image_name, return_nparray=True)

        image_rotate = False
        det_result = None
        ocr_result = None
        rotate_180_flag = False
        if not is_pdf:
            image_rotate = True
            rotate_180_flag, det_result, ocr_result, det_metric, rec_metric = self.pre_process_image(image_name=inputs,
                                                                                                     image_full=image_full)
            image_full = OcrCommonUtils.read_image(image_name, return_nparray=True)

        ocr_system_output = OcrSystemModelOutput(src_id=src_id, page=page,
                                                 run_time=run_time,
                                                 file_name=inputs,
                                                 raw_filename=raw_filename,
                                                 image_shape=image_full.shape,
                                                 is_pdf=is_pdf,
                                                 image_rotate=image_rotate,
                                                 image_full=image_full,
                                                 )

        if is_pdf:
            ocr_result, rec_metric = self.pdf_text_extract(ocr_system_output, )
            det_result = []
            for one_item in ocr_result:
                det_result.append(one_item["bbox"].reshape([1, 8]))
            det_result = np.concatenate(det_result)

            det_metric = {"use_time": 0}
        else:
            if rotate_180_flag:
                logger.info(f"图片旋转了180度需要重新进行OCR识别")
                det_result, det_metric = self.text_detection(image_full, )
                ocr_result, rec_metric = self.text_recognition(det_result=det_result, image_full=image_full)

        ocr_cells = []
        for one_item in ocr_result:
            cell = OcrCell(raw_data=one_item, cell_type=HtmlContentType.TXT)
            ocr_cells.append(cell)

        # ocr to html
        ocr_system_output.det_result = det_result
        ocr_system_output.ocr_result = ocr_cells

        use_time = time.time() - begin_time

        end_time_str = TimeUtils.now_str()
        use_time_str = TimeUtils.calc_diff_time(begin_time_str, end_time_str)

        metric = {
            "begin_time": begin_time_str,
            "end_time": end_time_str,
            "use_time": use_time_str,
            "use_time2": use_time,

            "detection": det_metric,
            "recognition": rec_metric,
        }

        ocr_system_output.metric = metric
        logger.info(f"OCR识别结束：{metric}")

        if self.output_dir is not None and save_result:

            ocr_result_show = self.show_ocr_result(ocr_result)
            # logger.info(f"det_result: {det_result}")
            # logger.info(f"ocr_result: {ocr_result}")
            logger.info(f"ocr_result_show: {ocr_result_show}")
            logger.info(f"metric: {metric}")

            save_file = f"{self.output_dir}/ocr_{raw_filename}_{run_time}.jpg"
            FileUtils.check_file_exists(save_file)

            # 绘制Text cell
            save_image_file = f"{self.output_dir}/{raw_filename}_text_cell.jpg"
            image = cv2.imread(ocr_system_output.file_name)
            cell_text_image_file = TableProcessUtils.show_text_cell(image=image,
                                                                    save_image_file=save_image_file,
                                                                    text_cells=[ocr_system_output.ocr_result], )

            image_draw = OcrCommonUtils.draw_boxes(image_full=image_full, det_result=det_result)

            image_draw = Image.fromarray(image_draw)
            image_draw.save(save_file)
            logger.info(f"save_file: {save_file}")

            # 保存左右对比结果
            draw_img = OcrInferUtils.show_compare_result(image_full=image_full,
                                                         ocr_result=ocr_cells,
                                                         lang=self.config.lang)
            ocr_compare_file = save_file.replace(".jpg", "_compare.jpg")
            Image.fromarray(draw_img).save(ocr_compare_file)
            logger.info(f"ocr_compare_file: {ocr_compare_file}")

            ocr_result_file = save_file.replace(".jpg", ".txt")
            ocr_result_show.to_csv(ocr_result_file, header=True, index=False, sep="\t", )

            ocr_result_to_save = []
            for item in ocr_result:
                new_item = deepcopy(item)
                new_item['bbox'] = item['bbox'].tolist()
                ocr_result_to_save.append(new_item)

            metric["result"] = ocr_result_to_save

            ocr_result_json_file = save_file.replace(".jpg", ".json")
            FileUtils.dump_json(ocr_result_json_file, metric)

        if self.delete_check_success and ocr_system_output.all_table_valid_check:
            FileUtils.delete_ocr_result_file(output_dir=self.output_dir, raw_filename=raw_filename)

        return ocr_system_output
