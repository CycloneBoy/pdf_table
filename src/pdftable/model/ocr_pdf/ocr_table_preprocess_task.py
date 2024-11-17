#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : ocr_table_preprocess_task.py
# @Author   : cycloneboy
# @Date     : 20xx/10/15 - 13:25
import shutil
import time

from pdftable.entity import ImagePreProcessOutput
from pdftable.model.ocr_pdf.cls_image_pulc_task import ClsImagePulcTask
from pdftable.utils import FileUtils, logger, TimeUtils, Constants
from pdftable.utils.ocr import OcrCommonUtils
from pdftable.utils.table.image_processing import PdfImageProcessor

"""
表格图片预处理
    - 文档方向判断
    - 表格属性识别
"""

__all__ = [
    'OcrTablePreprocessTask'
]


class OcrTablePreprocessTask(object):

    def __init__(self, config=None, task_list=None, debug=True,
                 output_dir=None, predictor_type="pytorch",
                 task="ocr_table_preprocess", **kwargs):
        super().__init__()
        self.config = config
        self.debug = debug
        self.output_dir = output_dir
        self.predictor_type = predictor_type
        self.task = task

        self.task_list = task_list if task_list is not None and isinstance(task_list, list) else [
            "text_image_orientation",
            "table_attribute"
            # "textline_orientation"
            # "language_classification"
        ]

        self.inner_task = {task: ClsImagePulcTask(task_type=task) for task in self.task_list}

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def pre_process_image(self, file_name, is_pdf, src_id=None):
        """
        判断图片是否需要旋转180度

        :param file_name:
        :param is_pdf:
        :return:
        """
        raw_filename = FileUtils.get_file_name(file_name)
        png_filename = f"{raw_filename}.png"
        cache_name = f"{Constants.PDF_PAGE_DIR}/{src_id}/{png_filename}"
        metric = None
        angle2 = None
        if is_pdf:

            image_name = f"{self.output_dir}/{png_filename}"
            if src_id is not None and FileUtils.check_file_exists(cache_name):
                image_name = cache_name
                logger.info(f"读取PDF缓存图片：{image_name}")
            else:
                if src_id is not None:
                    image_name = cache_name
                image_name, do_convert = PdfImageProcessor.convert_pdf_to_image(file_name=file_name,
                                                                                image_name=image_name)

            image_full = OcrCommonUtils.read_image(image_name, return_nparray=True)
        else:
            image_name = file_name

            image_full = OcrCommonUtils.read_image(image_name, return_nparray=True)
            image_full, metric, angle2 = self.pre_rotate_image(image_name=image_name, image_full=image_full)

        return image_full, metric, angle2, image_name

    def pre_rotate_image(self, image_name, image_full):
        """
        判断图片是否需要旋转180度

        :param image_name:
        :param image_full:
        :return:
        """

        logger.info(f"开始图片旋转微调修复: {image_name}")
        rotated_image, angle2 = PdfImageProcessor.rotate_image(image_name, save_image_file=image_name,
                                                               threshold_block_size=15,
                                                               threshold_constant=-2,
                                                               line_scale_horizontal=40,
                                                               line_scale_vertical=40,
                                                               iterations=0,
                                                               diff_angle=400,
                                                               angle_threshold=0.2
                                                               )
        logger.info(f"完成图片旋转微调修复: {image_name}")

        # estimated_angle = get_angle(image=image_full)
        # logger.info(f"旋转微调修复[skew]: {estimated_angle}")

        image_full, metric = self.rotate_image_v2(image_name, rotated_image)

        metric["rotate_small"] = angle2
        # metric["rotate_small2"] = estimated_angle

        return image_full, metric, angle2

    def rotate_image_v2(self, image_name, image_full):
        image_orientation_result = self.get_image_cls_result(image=image_name)

        angle = "0"
        metric = {
            "angle": image_orientation_result.image_orientation,
            "score": image_orientation_result.image_orientation_score,
        }

        # angle_list = [90, 180, 270]
        # angle_result = {}
        # for run_angle in angle_list:
        #     new_image_name = image_name.replace(".jpg", f"_{run_angle}.jpg")
        #     new_image_name = new_image_name.replace(".png", f"_{run_angle}.png")
        #     image_full2 = PdfImageProcessor.rotate_image_angle_v2(image=image_full,
        #                                                           angle=run_angle,
        #                                                           save_image_file=new_image_name)
        #     # check
        #     new_image_orientation_result = self.get_image_cls_result(image=new_image_name)
        #     angle_result[run_angle] = new_image_orientation_result.get_image_orientation()
        # metric["angle_result"] = angle_result

        if image_orientation_result.check_rotate():
            # if True:
            angle = image_orientation_result.image_orientation

            image_name1 = image_name.replace(".jpg", "_v1.jpg")
            image_name1 = image_name1.replace(".png", "_v1.png")
            image_full2 = PdfImageProcessor.rotate_image_angle_v2(image=image_full,
                                                                  angle=angle,
                                                                  save_image_file=image_name1)
            # check
            image_orientation_result2 = self.get_image_cls_result(image=image_name1)
            angle2 = image_orientation_result2.image_orientation
            # if int(angle) == int(angle2) or abs(int(angle) - int(angle2)) == 180:
            #     logger.info(f"pdf图片文字方向分类，两次旋转后结果相反，不旋转。")
            if angle2 in ["0", "180"]:
                shutil.move(image_name1, image_name)
                image_full = image_full2

                logger.info(f"pdf图片文字方向分类，逆时针旋转{angle}度：{image_name}")
            else:
                logger.info(f"pdf图片文字方向分类，不旋转：{angle} - {angle2}")

            metric["angle2"] = angle2
            metric["score2"] = image_orientation_result2.image_orientation_score

        return image_full, metric

    def get_image_cls_result(self, image) -> ImagePreProcessOutput:
        result = {}
        for task in self.task_list:
            result[task] = self.inner_task[task](image)

        text_image_orientation_result = result["text_image_orientation"]
        score = text_image_orientation_result['scores'][0]
        predict = text_image_orientation_result['label_names'][0]

        table_attribute_result = result["table_attribute"]

        output = ImagePreProcessOutput(image_orientation=predict,
                                       image_orientation_score=score,
                                       table_attribute=table_attribute_result["attributes"],
                                       table_attribute_score=table_attribute_result["output"])

        logger.info(f"text_image_orientation_result: {FileUtils.get_file_name(image)} -  {result}")

        return output

    def __call__(self, inputs, src_id=None):
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()
        run_time = TimeUtils.now_str_short()

        is_pdf = FileUtils.is_pdf_file(inputs)
        logger.info(f"数据源是{'PDF' if is_pdf else '图片'}: {inputs}")

        raw_filename = FileUtils.get_file_name(inputs)

        image_full, angle_metric, angle2, image_name = self.pre_process_image(file_name=inputs, is_pdf=is_pdf,
                                                                              src_id=src_id)

        use_time = time.time() - begin_time
        metric = {
            "use_time": use_time,
            "angle_metric": angle_metric,
            # "angle": angle_metric["angle"],
            "angle2": angle2,
            "image_name": image_name,
        }
        return image_full, metric
