#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_system_task
# @Author  ：cycloneboy
# @Date    ：20xx/7/17 15:01
import os.path
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List, Union, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from pdftable.entity import HtmlContentType
from pdftable.entity.table_entity import OcrCell
from pdftable.model import TableProcessUtils
from pdftable.model.ocr_pdf import OCRDocumentConfig, OcrLayoutTask, OcrTableCellTask, OcrPdfTextTask, \
    OcrTableToHtmlTask, OcrToHtmlTask, OcrDetectionTask, OcrTableStructureTask
from pdftable.model.ocr_pdf.cls_image_pulc_task import ClsImagePulcTask
from pdftable.model.ocr_pdf.ocr_output import OcrSystemModelOutput
from pdftable.model.ocr_pdf.ocr_recognition_task import OcrRecognitionTask
from pdftable.model.ocr_pdf.ocr_table_preprocess_task import OcrTablePreprocessTask
from pdftable.utils import logger, FileUtils, TimeUtils
from pdftable.utils.benchmark_utils import print_timings
from pdftable.utils.ocr import OcrCommonUtils, OcrInferUtils
from pdftable.utils.table.image_processing import PdfImageProcessor

"""
OcrSystemTask
"""

__all__ = [
    'OcrSystemTask'
]


class OcrSystemTask(object):

    def __init__(self, config: OCRDocumentConfig, debug=True,
                 output_dir=None, predictor_type="onnx",
                 delete_check_success=True,
                 need_save_to_db=False,
                 **kwargs):
        super().__init__()
        self.config = config
        self.debug = debug
        self.output_dir = output_dir
        self.predictor_type = predictor_type
        self.delete_check_success = delete_check_success
        self.need_save_to_db = need_save_to_db

        # # TODO: onnx 推理占用显存过大,暂时采用pytorch推理
        # self.text_detector = OcrDetectionTask(task="ocr_detection", model="db",
        #                                       backbone=config.detector,
        #                                       thresh=config.thresh,
        #                                       predictor_type="pytorch")
        #
        # self.text_recognizer = OcrRecognitionTask(task="ocr_recognition",
        #                                           model=config.recognizer,
        #                                           task_type=config.task_type,
        #                                           predictor_type=predictor_type,
        #                                           )
        #
        # # TODO: onnx 推理占用显存过大,暂时采用pytorch推理
        # self.table_structure_recognizer = OcrTableStructureTask(predictor_type="pytorch")

        # 延迟加载模型
        self.text_detector = None
        self.text_recognizer = None
        self.table_structure_recognizer = None

        self.layout_detector = OcrLayoutTask(model=self.config.layout_model,
                                             task_type=self.config.layout_model_task_type,
                                             debug=self.debug,
                                             output_dir=self.output_dir)

        self.table_cell_detector = OcrTableCellTask(output_dir=self.output_dir)
        self.pdf_text_recognizer = OcrPdfTextTask(output_dir=self.output_dir, debug=self.debug)

        self.ocr_table_to_html = OcrTableToHtmlTask(output_dir=self.output_dir)

        self.ocr_to_html = OcrToHtmlTask(output_dir=self.output_dir)
        self.ocr_result_to_db = None

        self.image_pre_process_task = OcrTablePreprocessTask(output_dir=self.output_dir)
        self.textline_orientation_task = ClsImagePulcTask(task_type="textline_orientation")

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        self.layout_detector.output_dir = output_dir
        self.table_cell_detector.set_output_dir(output_dir)
        self.pdf_text_recognizer.output_dir = output_dir
        self.ocr_table_to_html.output_dir = output_dir
        self.ocr_to_html.output_dir = output_dir
        # self.ocr_result_to_db.output_dir = output_dir

        if self.text_detector is not None:
            self.text_detector.output_dir = output_dir

        if self.text_recognizer is not None:
            self.text_recognizer.output_dir = output_dir

        if self.table_structure_recognizer is not None:
            self.table_structure_recognizer.output_dir = output_dir

        if self.image_pre_process_task is not None:
            self.image_pre_process_task.output_dir = output_dir

        FileUtils.check_file_exists(f"{output_dir}/demo.txt")

    def init_ocr_model(self, is_pdf=False):
        start = time.time()
        logger.info(f"开始加载OCR模型")
        init_model = []
        if self.text_detector is None and not is_pdf:
            self.text_detector = OcrDetectionTask(task="ocr_detection",
                                                  model=self.config.detect_model,
                                                  backbone=self.config.detector,
                                                  thresh=self.config.thresh, )
            init_model.append(self.text_detector.task)

        if self.text_recognizer is None and not is_pdf:
            self.text_recognizer = OcrRecognitionTask(task="ocr_recognition",
                                                      model=self.config.recognizer,
                                                      task_type=self.config.task_type,
                                                      predictor_type=self.predictor_type,
                                                      )
            init_model.append(self.text_recognizer.task)

        if self.table_structure_recognizer is None:
            self.table_structure_recognizer = OcrTableStructureTask(model=self.config.table_structure_model,
                                                                    predictor_type=self.predictor_type,
                                                                    output_dir=self.output_dir,
                                                                    lang=self.config.lang,
                                                                    debug=self.debug,
                                                                    task_type=self.config.table_structure_task_type,
                                                                    table_structure_merge=self.config.table_structure_merge)
            init_model.append(self.table_structure_recognizer.task)

        use_time = time.time() - start
        logger.info(f"加载OCR模型完毕： {len(init_model)} - {init_model}, 耗时：{use_time:3f} s.")

    def text_detection(self, image):
        start = time.time()

        if self.text_detector is None:
            self.init_ocr_model()

        result = self.text_detector(image)
        det_result = result[0]
        use_time = time.time() - start
        logger.info(f"检测耗时：{use_time:3f} s. det_result: ")

        # sort detection result with coord
        det_result_list = det_result.tolist()
        det_result_list = sorted(det_result_list, key=lambda x: 0.01 * sum(x[::2]) / 4 + sum(x[1::2]) / 4)
        result = np.array(det_result_list)
        metric = {
            "use_time": use_time
        }
        return result, metric

    def table_structure_detection(self, image, image_full=None, layout_result=None, is_pdf=False):
        """
        采用 layout 分割出表格 进行识别

        :param image:
        :param image_full:
        :param layout_result:
        :return:
        """
        start = time.time()

        if self.table_structure_recognizer is None:
            self.init_ocr_model(is_pdf=is_pdf)

        # result0 = self.table_structure_recognizer(image)
        result0 = None
        if layout_result and not self.table_structure_recognizer.is_line_cell_model():
            layout_tables = TableProcessUtils.get_layout_by_type(layout_result=layout_result,
                                                                 label="table",
                                                                 score_threshold=0.2)

            outputs = []
            total = len(layout_tables)
            logger.info(f"表格结构识别开始：总共待识别：{total} 个。")
            for index, table in enumerate(layout_tables):
                bbox = table["bbox"]
                save_name = f"{self.output_dir}/{FileUtils.get_file_name(image)}_{index}.jpg"
                FileUtils.check_file_exists(save_name)
                image_crop = OcrCommonUtils.crop_image_by_box(image_full, bbox=bbox, save_name=save_name)
                one_result = self.table_structure_recognizer(save_name)
                outputs.append([bbox, one_result[0]])

            result = TableProcessUtils.convert_table_sep_to_merge(outputs, result0)
            result[0]["inputs"] = image
        else:
            result = self.table_structure_recognizer(image)
        table_structure_result = result[0]
        use_time = time.time() - start
        logger.info(f"表格结构检测耗时：{use_time:3f} s. 表格数量: {len(table_structure_result['structure_str_list'])}")

        result = table_structure_result
        metric = {
            "use_time": use_time
        }
        return result, metric

    def layout_analysis(self, image):
        start = time.time()
        result = self.layout_detector(image)
        layout_result = result[0]
        use_time = time.time() - start
        logger.info(f"版面分析检测耗时：{use_time:3f} s. layout_result: {layout_result}")

        result = layout_result
        metric = {
            "use_time": use_time
        }
        return result, metric

    def table_cell_detection(self, image, is_pdf=False, page=None, ocr_system_output: OcrSystemModelOutput = None):
        """
        表格结构检测

        :param image:
        :param is_pdf:
        :param page:
        :param ocr_system_output:
        :return:
        """
        start = time.time()
        result = self.table_cell_detector(image, is_pdf=is_pdf, page=page, ocr_system_output=ocr_system_output)
        table_cell_result = result[0]
        use_time = time.time() - start
        logger.info(f"表格结构Cell检测耗时：{use_time:3f} s. table_cell_result: {len(table_cell_result)}")

        result = table_cell_result
        metric = {
            "use_time": use_time
        }
        return result, metric

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

    def text_recognition(self, det_result, image_full) -> Tuple[List[OcrCell], Dict]:
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
        avg_use_time = total_use_time / len(use_times)
        metric = {
            "use_time": total_use_time,
            "avg_use_time": avg_use_time,
            "total": len(use_times),
        }
        logger.info(f"识别结束：{metric}")

        ocr_cells = []
        for one_item in output:
            cell = OcrCell(raw_data=one_item, cell_type=HtmlContentType.TXT)
            ocr_cells.append(cell)

        return ocr_cells, metric

    def table_to_html(self, ocr_system_output: OcrSystemModelOutput):
        """
        表格转html

        :param ocr_system_output:
        :return:
        """
        start = time.time()
        ocr_system_output = self.ocr_table_to_html(ocr_system_output)

        use_time = time.time() - start
        logger.info(f"表格内容转html耗时：{use_time:3f} s. 提取表格数量: {len(ocr_system_output.table_cell_result)}")

        result = ocr_system_output
        metric = {
            "use_time": use_time
        }
        return result, metric

    def ocr_result_to_html(self, ocr_system_output: OcrSystemModelOutput):
        """
        ocr result转html

        :param ocr_system_output:
        :return:
        """
        start = time.time()
        ocr_system_output = self.ocr_to_html(ocr_system_output)

        use_time = time.time() - start
        logger.info(f"OCR内容转html耗时：{use_time:3f} s.")

        result = ocr_system_output
        metric = {
            "use_time": use_time
        }
        return result, metric

    def ocr_result_save_to_db(self, ocr_system_output: OcrSystemModelOutput):
        """
        ocr result 保存数据库

        :param ocr_system_output:
        :return:
        """
        start = time.time()
        # ocr_system_output = self.ocr_result_to_db(ocr_system_output)

        use_time = time.time() - start
        logger.info(f"OCR结果保存数据库耗时：{use_time:3f} s.")

        result = ocr_system_output
        metric = {
            "use_time": use_time
        }
        return result, metric

    def text_line_orientation(self, det_result, image_full, score_threshold=0.9) -> Tuple[bool, Dict]:
        """
        文字方向识别

        :param det_result:
        :param image_full:
        :param score_threshold:
        :return:
        """

        use_times = []
        degree_0_total = 0
        degree_180_total = 0
        total = len(det_result)
        logger.info(f"文字方向识别开始：总共待识别：{total} 个。")
        start = time.time()
        for i in range(det_result.shape[0]):
            pts = OcrCommonUtils.order_point(det_result[i])
            image_crop = OcrCommonUtils.crop_image(image_full, pts)
            result = self.textline_orientation_task(image_crop)
            # one_use_time = metric['use_time']
            # use_times.append(one_use_time)

            label = result['label_names'][0]
            score = result['scores'][0]
            if score > score_threshold:
                if label == "0_degree":
                    degree_0_total += 1
                else:
                    degree_180_total += 1

        orientation = True if degree_0_total > degree_180_total else False
        logger.info(f"text_line_result: degree_0: {orientation} -degree_0_total: {degree_0_total} "
                    f"-degree_180_total: {degree_180_total}")

        total_use_time = time.time() - start
        avg_use_time = total_use_time / total
        metric = {
            "use_time": total_use_time,
            "avg_use_time": avg_use_time,
            "total": total,
        }
        logger.info(f"文字方向识别结束：{metric}")

        return orientation, metric

    def image_pre_process(self, inputs, is_pdf=True, src_id=None):
        """
        图片预处理
            - 图片方向旋转：0,90,180,270
            - 图片小角度微调旋转；

        :param inputs:
        :param is_pdf:
        :return:
        """
        start = time.time()
        image_full, raw_metric = self.image_pre_process_task(inputs, src_id=src_id)

        use_time = time.time() - start
        logger.info(f"图片预处理【图片方向旋转,图片小角度微调旋转】耗时：{use_time:3f} s.")

        image_name = raw_metric["image_name"]

        det_result = None
        if not is_pdf:
            det_result, det_metric = self.text_detection(image_full, )
            rotate90_flag = TableProcessUtils.check_pdf_text_need_rotate90(det_result)
            if rotate90_flag:
                # src_image = cv2.imread(image_name)
                image_full = PdfImageProcessor.rotate_image_angle_v2(image=image_full,
                                                                     angle=270,
                                                                     save_image_file=image_name)
                logger.info(f"pdf图片文本框宽度和高度之比，逆时针旋转90度：{image_name}")
                det_result, det_metric = self.text_detection(image_full, )

            text_line_result, rec_metric = self.text_line_orientation(det_result=det_result, image_full=image_full)

            if not text_line_result:
                image_full = PdfImageProcessor.rotate_image_angle_v2(image=image_full,
                                                                     angle=180,
                                                                     save_image_file=image_name)
                logger.info(f"pdf图片文本方向180度，顺时针旋转180度：{image_name}")
                det_result = None
        else:
            logger.info(f"数字型pdf不进行文本方向校验：{image_name}")

        result = image_full
        metric = {
            "use_time": use_time,
            # "angle": raw_metric["angle"],
            # "angle2": raw_metric["angle2"],
            "angle_metric": raw_metric["angle_metric"],
            "image_name": raw_metric["image_name"],
            "det_result": det_result,
        }
        return result, metric

    def show_ocr_result(self, ocr_result: List[OcrCell]):
        output = []
        for item in ocr_result:
            output.append([item.index, item.text, ','.join([str(e) for e in list(item.bbox.reshape(-1))])])

        ocr_result_pd = pd.DataFrame(output, columns=['检测框序号', '行识别结果', '检测框坐标'])

        return ocr_result_pd

    def convert_to_html(self, ocr_system_output: OcrSystemModelOutput):
        pass

    def rotate_image(self, image_name, image_full):
        det_result, det_metric = self.text_detection(image_full, )
        rotate90_flag = TableProcessUtils.check_pdf_text_need_rotate90(det_result)
        if rotate90_flag:
            # src_image = cv2.imread(image_name)
            image_full = PdfImageProcessor.rotate_image_angle_v2(image=image_full,
                                                                 angle=270,
                                                                 save_image_file=image_name)
            logger.info(f"pdf图片文本框宽度和高度之比，逆时针旋转90度：{image_name}")
            det_result, det_metric = self.text_detection(image_full, )

        ocr_result, rec_metric = self.text_recognition(det_result=det_result, image_full=image_full)
        ocr_text1 = [one_item.text for one_item in ocr_result]

        # 旋转180度后的结果
        src_image = cv2.imread(image_name)
        image_full2 = PdfImageProcessor.rotate_image_angle_v2(image=src_image,
                                                              angle=180,
                                                              save_image_file=image_name)
        logger.info(f"pdf图片宽需要翻转，逆时针旋转180度：{image_name}")
        det_result2, det_metric2 = self.text_detection(image_full2, )
        ocr_result2, rec_metric2 = self.text_recognition(det_result=det_result2, image_full=image_full2)
        ocr_text2 = [one_item.text for one_item in ocr_result2]

        flag = TableProcessUtils.check_pdf_text_need_rotate(ocr_text1, ocr_text2)
        if flag:
            return image_full, det_result, ocr_result, det_metric, rec_metric
        else:
            return image_full2, det_result2, ocr_result2, det_metric2, rec_metric2

    def run_debug(self, inputs, save_result=True, src_id=None, page=None, **kwargs):
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()
        run_time = TimeUtils.now_str_short()

        is_pdf = FileUtils.is_pdf_file(inputs)
        logger.info(f"数据源是{'PDF' if is_pdf else '图片'}: {inputs}")

        raw_filename = FileUtils.get_file_name(inputs)

        image_full, image_pre_metric = self.image_pre_process(inputs)

        return image_full, image_pre_metric

    def __call__(self, inputs, save_result=True, src_id=None, page=None, **kwargs) \
            -> Tuple[OcrSystemModelOutput, Dict]:
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()
        run_time = FileUtils.get_file_name(f"{self.output_dir}.txt")
        if len(run_time) != 15:
            run_time = TimeUtils.now_str_short()

        is_pdf = FileUtils.is_pdf_file(inputs)
        logger.info(f"数据源是{'PDF' if is_pdf else '图片'}: {inputs}")

        raw_filename = FileUtils.get_file_name(inputs)

        # PDF前处理：文本方向旋转
        image_full, image_pre_metric = self.image_pre_process(inputs, is_pdf=is_pdf, src_id=src_id)
        image_name = image_pre_metric["image_name"]
        src_det_result = image_pre_metric["det_result"]

        ocr_system_output = OcrSystemModelOutput(src_id=src_id, page=page,
                                                 run_time=run_time,
                                                 file_name=inputs,
                                                 raw_filename=raw_filename,
                                                 image_shape=image_full.shape,
                                                 is_pdf=is_pdf,
                                                 image_name=image_name
                                                 )
        layout_result, layout_metric = self.layout_analysis(image_full, )
        ocr_system_output.layout_result = layout_result

        table_structure_result, table_structure_metric = self.table_structure_detection(image_name,
                                                                                        image_full=image_full,
                                                                                        layout_result=layout_result,
                                                                                        is_pdf=is_pdf)
        ocr_system_output.table_structure_result = table_structure_result

        # 绘制版面分析结果
        save_layout_path = f"{self.output_dir}/{raw_filename}_layout.jpg"
        if self.debug:
            save_layout_json_path = save_layout_path.replace(".jpg", ".json")
            layout_result_list = deepcopy(layout_result)
            for v in layout_result_list:
                v['bbox'] = np.round(v['bbox']).astype(np.int32).tolist()
            FileUtils.dump_json(save_layout_json_path, layout_result_list)

        if table_structure_result is not None and self.debug:
            tmp_layout_img = OcrInferUtils.draw_text_layout_res(img=image_full,
                                                                layout_res=layout_result,
                                                                save_path=save_layout_path, )

            image_draw = OcrCommonUtils.draw_boxes(image_full=tmp_layout_img,
                                                   det_result=ocr_system_output.get_table_structure_bboxs(),
                                                   color="green", width=3)
            image_draw = Image.fromarray(image_draw)
            image_draw.save(save_layout_path)

        # 表格 cell 提取
        if self.config.table_structure_merge:
            logger.info(f"表格结构识别采用: Lore和LineCell两个模型合并结果")
            table_cell_result, table_cell_metric = self.table_cell_detection(inputs, is_pdf=is_pdf,
                                                                             page=page,
                                                                             ocr_system_output=ocr_system_output)
        else:
            table_cell_result = table_structure_result["table_cells"]
            table_cell_metric = table_structure_result["table_cell_metric"]
        ocr_system_output.table_cell_result = table_cell_result

        if is_pdf:
            ocr_result, rec_metric = self.pdf_text_extract(ocr_system_output, )
            det_result = []
            for one_item in ocr_result:
                det_result.append(one_item.bbox.reshape([1, 8]))
            det_result = np.concatenate(det_result)

            det_metric = {"use_time": 0}
        else:
            # 采用之前换存的
            if src_det_result is not None:
                det_result = src_det_result
                det_metric = {"use_time": 0}
            else:
                det_result, det_metric = self.text_detection(image_full, )
            ocr_result, rec_metric = self.text_recognition(det_result=det_result, image_full=image_full)

        # ocr to html
        ocr_system_output.det_result = det_result
        ocr_system_output.ocr_result = ocr_result

        if self.config.table_structure_model:
            ocr_system_output.use_master = self.config.tsr_match_need_use_master()
        table_html_result, table_html_metric = self.table_to_html(ocr_system_output)
        ocr_html_result, ocr_html_metric = self.ocr_result_to_html(ocr_system_output)

        use_time = time.time() - begin_time

        end_time_str = TimeUtils.now_str()
        use_time_str = TimeUtils.calc_diff_time(begin_time_str, end_time_str)

        metric = {
            "begin_time": begin_time_str,
            "end_time": end_time_str,
            "use_time": use_time_str,
            "use_time2": use_time,

            "table_cell": table_cell_metric,
            "detection": det_metric,
            "table_structure": table_structure_metric,
            "layout_analysis": layout_metric,
            "recognition": rec_metric,
            "table_html": table_html_metric,
            "ocr_html": ocr_html_metric,
            # "ocr_to_db": ocr_to_db_metric,
        }

        ocr_system_output.metric = metric
        if self.need_save_to_db:
            ocr_to_db_result, ocr_to_db_metric = self.ocr_result_save_to_db(ocr_system_output)
            metric["ocr_to_db"] = ocr_to_db_metric

        logger.info(f"OCR识别结束：{metric}")

        if self.debug:
            ocr_result_json_file = f"{self.output_dir}/ocr_{raw_filename}_{run_time}.json"
            FileUtils.dump_json(ocr_result_json_file, metric)

        if self.output_dir is not None and save_result and self.debug:

            ocr_result_show = self.show_ocr_result(ocr_result)
            # logger.info(f"det_result: {det_result}")
            # logger.info(f"ocr_result: {ocr_result}")
            # logger.info(f"ocr_result_show: {ocr_result_show}")
            logger.info(f"metric: {metric}")

            save_file = f"{self.output_dir}/ocr_{raw_filename}_{run_time}.jpg"
            FileUtils.check_file_exists(save_file)

            # 绘制Text cell
            save_image_file = f"{self.output_dir}/{raw_filename}_text_cell.jpg"
            image_file = FileUtils.get_pdf_to_image_file_name(ocr_system_output.file_name)
            image = cv2.imread(image_file)
            cell_text_image_file = TableProcessUtils.show_text_cell(image=image,
                                                                    save_image_file=save_image_file,
                                                                    text_cells=[ocr_system_output.ocr_result], )

            image_draw = OcrCommonUtils.draw_boxes(image_full=image_full, det_result=det_result)
            # 绘制表格结构
            if table_structure_result is not None:
                image_draw = OcrCommonUtils.draw_boxes(image_full=image_draw,
                                                       det_result=ocr_system_output.get_table_structure_bboxs(),
                                                       color="red", width=3)

            # 绘制表格CELL结构
            if len(table_cell_result) > 0 and table_cell_result[0]["table_cells"] is not None:
                image_draw = OcrCommonUtils.draw_table_cell_boxes(image_full=image_full, cell_result=table_cell_result)

            # 绘制版面分析结果
            OcrInferUtils.draw_text_layout_res(img=image_full, layout_res=layout_result, return_new_image=False)

            image_draw = Image.fromarray(image_draw)
            image_draw.save(save_file)
            logger.info(f"save_file: {save_file}")

            # 保存左右对比结果
            draw_img = OcrInferUtils.show_compare_result(image_full=image_full,
                                                         ocr_result=ocr_result,
                                                         table_structure_result=ocr_system_output.get_table_structure_bboxs(),
                                                         table_cell_result=table_cell_result,
                                                         layout_result=layout_result)
            ocr_compare_file = save_file.replace(".jpg", "_compare.jpg")
            Image.fromarray(draw_img).save(ocr_compare_file)
            logger.info(f"ocr_compare_file: {ocr_compare_file}")

            ocr_result_file = save_file.replace(".jpg", ".txt")
            ocr_result_show.to_csv(ocr_result_file, header=True, index=False, sep="\t", )

            ocr_result_to_save = [item.to_json_save() for item in ocr_result]

            metric["result"] = ocr_result_to_save

            if self.debug:
                ocr_result_json_file = save_file.replace(".jpg", ".json")
                FileUtils.dump_json(ocr_result_json_file, metric)

        if self.delete_check_success and ocr_system_output.all_table_valid_check:
            FileUtils.delete_ocr_result_file(output_dir=self.output_dir, raw_filename=raw_filename)

        return ocr_system_output, metric

    def ocr(self, file_list: Union[str, List], end_with=".jpg", src_id=None):
        """
        批量提取

        :param file_list:
        :param end_with:
        :param src_id:
        :return:
        """
        if isinstance(file_list, str) and os.path.isdir(file_list):
            file_list = FileUtils.list_file_prefix(file_dir=file_list,
                                                   add_parent=True,
                                                   end_with=end_with,
                                                   sort=True, )
        logger.info(f"总共需要提取：{len(file_list)}")

        begin = time.time()

        all_use_time = defaultdict(list)

        all_result = []
        for index, image in enumerate(file_list):
            det_result, ocr_result, metric = self.__call__(image, src_id=src_id)

            for k, v in metric.items():
                if k in ["result"]:
                    continue
                all_use_time[k].append(v["use_time"])

            save_metric = deepcopy(metric)
            save_metric.pop("result")
            one_result = {
                "det_result": det_result,
                "ocr_result": ocr_result,
                "metric": save_metric,
            }
            all_result.append(one_result)

        use_time = time.time() - begin
        logger.info(f"解析完成[{self.predictor_type}], 耗时：{use_time:.3f} s / {use_time / 60:.3f} min. "
                    f"总量：{len(file_list)}")

        total_time = {}
        for key, val in all_use_time.items():
            total = sum(val)
            total_time[key] = total
            infer_time_results = print_timings(name=f"Ocr {key} inference speed[{self.predictor_type}]:",
                                               timings=val)

        all_total_time = sum(total_time.values())
        for key, val in total_time.items():
            logger.info(
                f"解析任务[{key}], 耗时：{val:.3f} s / {val / 60:.3f} min. 耗时总的占比：{val / all_total_time:.3f} "
                f"总量：{len(file_list)}")

        return all_result
