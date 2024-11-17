#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：image_processing
# @Author  ：cycloneboy
# @Date    ：20xx/5/29 16:59
import os
import time
import traceback
from typing import List, Tuple

import cv2
import fitz
import numpy as np
from pdfminer.layout import LTImage

from pdftable.entity import LineDirectionType
from pdftable.entity.table_entity import Point, Line
from pdftable.model.pdf_table.ghostscript_backend import GhostscriptBackend
from pdftable.utils import logger, FileUtils, MathUtils

"""
PDF 表格识别

"""


class PdfImageProcessor(object):

    @staticmethod
    def adaptive_threshold(imagename, process_background=False, blocksize=15, c=-2):
        """Thresholds an image using OpenCV's adaptiveThreshold.

            href: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

        Parameters
        ----------
        imagename : string
            Path to image file.
        process_background : bool, optional (default: False)
            Whether or not to process lines that are in background.
        blocksize : int, optional (default: 15)
            Size of a pixel neighborhood that is used to calculate a
            threshold value for the pixel: 3, 5, 7, and so on.

            For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/4.7.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3>`_.
        c : int, optional (default: -2)
            Constant subtracted from the mean or weighted mean.
            Normally, it is positive but may be zero or negative as well.

            For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/4.7.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3>`_.

        Returns
        -------
        img : object
            numpy.ndarray representing the original image.
        threshold : object
            numpy.ndarray representing the thresholded image.

        """
        if isinstance(imagename, str):
            img = cv2.imread(imagename)
        else:
            img = imagename
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not process_background:
            gray = np.invert(gray)

        threshold = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c
        )

        return img, threshold

    @staticmethod
    def find_lines(
            threshold, regions=None, direction="horizontal", line_scale=15, iterations=0
    ):
        """Finds horizontal and vertical lines by applying morphological
        transformations on an image.

        Parameters
        ----------
        threshold : object
            numpy.ndarray representing the thresholded image.
        regions : list, optional (default: None)
            List of page regions that may contain tables of the form x1,y1,x2,y2
            where (x1, y1) -> left-top and (x2, y2) -> right-bottom
            in image coordinate space.
        direction : string, optional (default: 'horizontal')
            Specifies whether to find vertical or horizontal lines.
        line_scale : int, optional (default: 15)
            Factor by which the page dimensions will be divided to get
            smallest length of lines that should be detected.

            The larger this value, smaller the detected lines. Making it
            too large will lead to text being detected as lines.
        iterations : int, optional (default: 0)
            Number of times for erosion/dilation is applied.

            For more information, refer `OpenCV's dilate <https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#dilate>`_.

        Returns
        -------
        dmask : object
            numpy.ndarray representing pixels where vertical/horizontal
            lines lie.
        lines : list
            List of tuples representing vertical/horizontal lines with
            coordinates relative to a left-top origin in
            image coordinate space.

        """
        lines = []

        if direction == "vertical":
            size = threshold.shape[0] // line_scale
            el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
        elif direction == "horizontal":
            size = threshold.shape[1] // line_scale
            el = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
        elif direction is None:
            raise ValueError("Specify direction as either 'vertical' or 'horizontal'")

        if regions is not None:
            region_mask = np.zeros(threshold.shape)
            for region in regions:
                x, y, w, h = region
                region_mask[y: y + h, x: x + w] = 1
            threshold = np.multiply(threshold, region_mask)

        threshold = cv2.erode(threshold, el)
        threshold = cv2.dilate(threshold, el)
        dmask = cv2.dilate(threshold, el, iterations=iterations)

        # try:
        #     _, contours, _ = cv2.findContours(
        #         threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        #     )
        # except ValueError:
        # for opencv backward compatibility
        contours, _ = cv2.findContours(
            threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x1, x2 = x, x + w
            y1, y2 = y, y + h
            if direction == "vertical":
                lines.append(((x1 + x2) // 2, y2, (x1 + x2) // 2, y1))
            elif direction == "horizontal":
                lines.append((x1, (y1 + y2) // 2, x2, (y1 + y2) // 2))

        return dmask, lines

    @staticmethod
    def find_lines_angle(
            threshold, regions=None, direction="horizontal", line_scale=15, iterations=0,
            diff_angle=100
    ):
        """Finds horizontal and vertical lines by applying morphological
        transformations on an image.

        Parameters
        ----------
        threshold : object
            numpy.ndarray representing the thresholded image.
        regions : list, optional (default: None)
            List of page regions that may contain tables of the form x1,y1,x2,y2
            where (x1, y1) -> left-top and (x2, y2) -> right-bottom
            in image coordinate space.
        direction : string, optional (default: 'horizontal')
            Specifies whether to find vertical or horizontal lines.
        line_scale : int, optional (default: 15)
            Factor by which the page dimensions will be divided to get
            smallest length of lines that should be detected.

            The larger this value, smaller the detected lines. Making it
            too large will lead to text being detected as lines.
        iterations : int, optional (default: 0)
            Number of times for erosion/dilation is applied.

            For more information, refer `OpenCV's dilate <https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#dilate>`_.

        Returns
        -------
        dmask : object
            numpy.ndarray representing pixels where vertical/horizontal
            lines lie.
        lines : list
            List of tuples representing vertical/horizontal lines with
            coordinates relative to a left-top origin in
            image coordinate space.

        """
        lines = []
        lines_v2 = []

        if direction == "vertical":
            size = threshold.shape[0] // line_scale
            el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
        elif direction == "horizontal":
            size = threshold.shape[1] // line_scale
            el = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
        elif direction is None:
            raise ValueError("Specify direction as either 'vertical' or 'horizontal'")

        if regions is not None:
            region_mask = np.zeros(threshold.shape)
            for region in regions:
                x, y, w, h = region
                region_mask[y: y + h, x: x + w] = 1
            threshold = np.multiply(threshold, region_mask)

        threshold = cv2.erode(threshold, el)
        threshold = cv2.dilate(threshold, el)
        dmask = cv2.dilate(threshold, el, iterations=iterations)

        contours, _ = cv2.findContours(
            threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        angles = []
        widths = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x1, x2 = x, x + w
            y1, y2 = y, y + h
            angle = PdfImageProcessor.get_line_angle(contours=c)

            if direction == "vertical":
                lines.append(((x1 + x2) // 2, y2, (x1 + x2) // 2, y1))
                widths.append(w)
                left = Point(x=(x1 + x2) // 2, y=y1)
                right = Point(x=left.x, y=y2)
                one_line = Line(left=left, right=right,
                                direction=LineDirectionType.VERTICAL,
                                width=w, height=h)
                lines_v2.append(one_line)

                if h > diff_angle:
                    angles.append(angle)

            elif direction == "horizontal":
                lines.append((x1, (y1 + y2) // 2, x2, (y1 + y2) // 2))
                widths.append(h)

                left = Point(x=x1, y=(y1 + y2) // 2)
                right = Point(x=x2, y=left.y)
                one_line = Line(left=left, right=right,
                                direction=LineDirectionType.HORIZONTAL,
                                width=w, height=h)
                lines_v2.append(one_line)

                if w > diff_angle:
                    angles.append(angle)

        avg_angle = PdfImageProcessor.average_angle(angles)
        avg_width = np.average(widths) if len(widths) > 0 else None
        # logger.info(f"avg angle - {direction} : {avg_angle} - {len(angles)} - {angles}")
        return dmask, lines, avg_angle, avg_width, lines_v2

    @staticmethod
    def average_angle(angles, direction="horizontal"):
        """

        :param angles:
        :param direction:
        :return:
        """
        filter_angles = [angle for angle in angles if angle != 0 and angle != 90]
        avg_angle = np.average(filter_angles) if len(filter_angles) > 0 else None
        return avg_angle

    @staticmethod
    def calculate_angle(point1, point2):
        """
        计算一条直线的角度

        :param point1:
        :param point2:
        :return:
        """
        x1 = float(point1[0])
        y1 = float(point1[1])
        x2 = float(point2[0])
        y2 = float(point2[1])

        if x2 - x1 == 0:
            result = 90  # 直线是竖直的
        elif y2 - y1 == 0:
            result = 0  # 直线是水平的
        else:
            # 计算斜率
            k = (y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result = np.arctan(k) * 57.29577
        # if result < 0:
        #     result = 180 + result
        return result

    @staticmethod
    def get_line_angle(contours):
        """
        提取一条线的倾斜角度

        :param contours:
        :param diff:
        :return:
        """
        points = np.squeeze(contours, axis=None)
        min_x = min(points[:, 0])
        min_y = min(points[:, 1])
        max_x = max(points[:, 0])
        max_y = max(points[:, 1])

        width = max_x - min_x
        height = max_y - min_y

        horizontal = False
        points_list = points.tolist()
        if width > height:
            horizontal = True
            points_list = sorted(points_list, key=lambda x: x[0])
        else:
            points_list = sorted(points_list, key=lambda x: x[1])

        lb = points_list[0]
        rt = points_list[-1]
        angle = PdfImageProcessor.calculate_angle(lb, rt)
        return angle

    @staticmethod
    def find_contours(vertical, horizontal):
        """Finds table boundaries using OpenCV's findContours.

        Parameters
        ----------
        vertical : object
            numpy.ndarray representing pixels where vertical lines lie.
        horizontal : object
            numpy.ndarray representing pixels where horizontal lines lie.

        Returns
        -------
        cont : list
            List of tuples representing table boundaries. Each tuple is of
            the form (x, y, w, h) where (x, y) -> left-top, w -> width and
            h -> height in image coordinate space.

        """
        mask = vertical + horizontal

        try:
            # for opencv backward compatibility
            contours, __ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"cv2.findContours error: {e}")
        # sort in reverse based on contour area and use first 10 contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        cont = []
        for c in contours:
            c_poly = cv2.approxPolyDP(c, 3, True)
            x, y, w, h = cv2.boundingRect(c_poly)
            cont.append((x, y, w, h))
        return cont

    @staticmethod
    def find_joints(contours, vertical, horizontal, diff=2):
        """Finds joints/intersections present inside each table boundary.

        Parameters
        ----------
        contours : list
            List of tuples representing table boundaries. Each tuple is of
            the form (x, y, w, h) where (x, y) -> left-top, w -> width and
            h -> height in image coordinate space.
        vertical : object
            numpy.ndarray representing pixels where vertical lines lie.
        horizontal : object
            numpy.ndarray representing pixels where horizontal lines lie.

        Returns
        -------
        tables : dict
            Dict with table boundaries as keys and list of intersections
            in that boundary as their value.
            Keys are of the form (x1, y1, x2, y2) where (x1, y1) -> lb
            and (x2, y2) -> rt in image coordinate space.

        """
        joints = np.multiply(vertical, horizontal)
        tables = {}
        for c in contours:
            x, y, w, h = c
            roi = joints[y: y + h, x: x + w]
            if h < diff * 10 or w < 200 or w * h < 500:
                continue
            try:
                # for opencv backward compatibility
                jc, __ = cv2.findContours(
                    roi.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"cv2.findContours error: {e}")

            if len(jc) < 4:  # remove contours with less than 4 joints
                continue
            joint_coords = []
            for j in jc:
                jx, jy, jw, jh = cv2.boundingRect(j)
                c1, c2 = x + (2 * jx + jw) // 2, y + (2 * jy + jh) // 2
                joint_coords.append((c1, c2))
            tables[(x, y + h, x + w, y)] = joint_coords

        return tables

    @staticmethod
    def find_close_norm_x(x, norm_list, atol=2):
        """
        查找最近的点

        :param x:
        :param norm_list:
        :param atol:
        :return:
        """
        new_x = x
        for norm_x in norm_list:
            if np.isclose(x, norm_x, atol=atol):
                new_x = norm_x
                break

        return new_x

    @staticmethod
    def rotate_image_angle(image, angle, save_image_file=None):
        """
        旋转图片指定角度

            - 提取旋转矩阵
        :param image:
        :param angle:
        :param save_image_file:
        :return:
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        # 提取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if save_image_file is not None:
            FileUtils.check_file_exists(save_image_file)
            cv2.imwrite(save_image_file, rotated_image)
            logger.info(f"保存旋转图片：{angle} - {save_image_file}")
        return rotated_image

    @staticmethod
    def rotate_image_angle_v2(image, angle=90, save_image_file=None):
        """
        旋转图片指定角度

            - 提取旋转矩阵
        :param image:
        :param angle:
        :param save_image_file:
        :return:
        """
        cv_rotate_code = {
            '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
            '180': cv2.ROTATE_180,
            '270': cv2.ROTATE_90_CLOCKWISE
        }
        angle = str(angle)

        if angle in cv_rotate_code:
            rotated_image = cv2.rotate(image, cv_rotate_code[angle])
        else:
            logger.info(f"没有旋转图片：{angle} 不是[90,180,270]度。")
            return image

        if save_image_file is not None:
            FileUtils.check_file_exists(save_image_file)
            cv2.imwrite(save_image_file, rotated_image)
            logger.info(f"保存旋转图片：{angle} - {save_image_file}")
        return rotated_image

    @staticmethod
    def rotate_image(image_name, save_image_file=None,
                     regions=None,
                     threshold_block_size=15,
                     threshold_constant=-2,
                     line_scale_horizontal=40,
                     line_scale_vertical=50,
                     iterations=0,
                     diff_angle=100,
                     angle_threshold=0.1
                     ):
        """
        旋转图片

        :param image_name:
        :param save_image_file:
        :param regions:
        :param threshold_block_size:
        :param threshold_constant:
        :param line_scale_horizontal:
        :param line_scale_vertical:
        :param iterations:
        :param diff_angle:
        :param angle_threshold:
        :return:
        """
        image, threshold, line_mask, angle = PdfImageProcessor.get_image_rotate_angle_v2(image_name,
                                                                                         regions=regions,
                                                                                         threshold_block_size=threshold_block_size,
                                                                                         threshold_constant=threshold_constant,
                                                                                         line_scale_horizontal=line_scale_horizontal,
                                                                                         line_scale_vertical=line_scale_vertical,
                                                                                         iterations=iterations,
                                                                                         diff_angle=diff_angle
                                                                                         )

        if abs(angle) < angle_threshold:
            logger.info(f"旋转角度较小，不需要旋转图片: {angle}")
            return image
        rotated_image = PdfImageProcessor.rotate_image_angle(image=image,
                                                             angle=angle,
                                                             save_image_file=save_image_file)
        logger.info(f"旋转图片: {angle} - {image_name}")
        return rotated_image

    @staticmethod
    def get_image_rotate_angle_v2(imagename,
                                  regions=None,
                                  threshold_block_size=15,
                                  threshold_constant=-2,
                                  line_scale_horizontal=15,
                                  line_scale_vertical=15,
                                  iterations=0,
                                  diff_angle=100):
        """
        图像旋转
            - 计算包含了旋转文本的最小边框
        :param imagename:
        :param regions:
        :param threshold_block_size:
        :param threshold_constant:
        :param line_scale_horizontal:
        :param line_scale_vertical:
        :param iterations:
        :param diff_angle:
        :return:
        """

        image, threshold = PdfImageProcessor.adaptive_threshold(imagename=imagename,
                                                                process_background=False,
                                                                blocksize=threshold_block_size,
                                                                c=threshold_constant,
                                                                )

        vertical_mask, vertical_segments, vertical_angle, _, _ = PdfImageProcessor.find_lines_angle(threshold,
                                                                                                    regions=regions,
                                                                                                    direction="vertical",
                                                                                                    line_scale=line_scale_vertical,
                                                                                                    iterations=iterations,
                                                                                                    diff_angle=diff_angle
                                                                                                    )
        horizontal_mask, horizontal_segments, horizontal_angle, _, _ = PdfImageProcessor.find_lines_angle(threshold,
                                                                                                          regions=regions,
                                                                                                          direction="horizontal",
                                                                                                          line_scale=line_scale_horizontal,
                                                                                                          iterations=iterations,
                                                                                                          diff_angle=diff_angle
                                                                                                          )

        line_mask = vertical_mask + horizontal_mask
        angle = 0
        if horizontal_angle is not None:
            logger.info(f"提取的旋转角度：{horizontal_angle} - {vertical_angle}")
            angle = horizontal_angle

        # 调整角度
        # if angle < -45:
        #     angle = -(90 + angle)
        # elif angle > 45:
        #     angle = -(90 - angle)
        # else:
        #     angle = -angle

        return image, threshold, line_mask, angle

    @staticmethod
    def find_cell_line_exist(image,
                             regions=None,
                             is_horizontal=True,
                             threshold_block_size=5,
                             threshold_constant=-2,
                             line_scale=10,
                             iterations=0,
                             line_radio=0.7):
        """
        判断一个cell 的边是否存在

        :param image:
        :param regions:
        :param is_horizontal:
        :param threshold_block_size:
        :param threshold_constant:
        :param line_scale:
        :param iterations:
        :param line_radio:
        :return:
        """
        image, threshold = PdfImageProcessor.adaptive_threshold(imagename=image,
                                                                process_background=False,
                                                                blocksize=threshold_block_size,
                                                                c=threshold_constant,
                                                                )

        direction = "horizontal" if is_horizontal else "vertical"

        height, width = threshold.shape
        line_length = width if is_horizontal else height
        # logger.info(f"判断一个cell 的边是否存在: {direction} - {threshold.shape}")
        mask, _, line_angle, _, line_segments = PdfImageProcessor.find_lines_angle(threshold,
                                                                                   regions=regions,
                                                                                   direction=direction,
                                                                                   line_scale=line_scale,
                                                                                   iterations=iterations,
                                                                                   )

        is_line = PdfImageProcessor.check_line_exist(lines=line_segments,
                                                     line_length=line_length,
                                                     radio=line_radio)

        return is_line

    @staticmethod
    def check_line_exist(lines: List[Line], line_length=1000000, radio=0.7):
        """
        判断线段是否存在

        :param lines:
        :param line_length:
        :param radio:
        :return:
        """
        flag = False

        if len(lines) == 0:
            return False

        total_length = 0

        mask = np.zeros(line_length)
        for line in lines:
            if line.direction == LineDirectionType.HORIZONTAL:
                length = line.width
                mask[line.min_x:line.max_x] = 1
            elif line.direction == LineDirectionType.VERTICAL:
                length = line.height
                mask[line.min_y:line.max_y] = 1
            else:
                length = 0

            # total_length += length

        total_length = sum(mask)
        if total_length / line_length >= radio:
            flag = True

        return flag

    @staticmethod
    def get_point_from_pdf_image(images: List[LTImage], area_threshold=10):
        """
        通过图片识别线段

        :param images:
        :param area_threshold:
        :return:
        """
        points = []
        for index, image in enumerate(images):
            w, h = image.srcsize
            area = w * h
            if area >= area_threshold:
                continue
            (x0, y0, x1, y1) = image.bbox
            x = round((x0 + x1) / 2)
            y = round((y0 + y1) / 2)
            points.append((x, y))

        return points

    @staticmethod
    def add_pdf_point_to_image(threshold, images: List[LTImage],
                               image_scalers, point_size=4, thickness=-2):
        """
        添加pdf点 到 image

        :param threshold:
        :param image_scalers:
        :param point_size:
        :param thickness:
        :return:
        """
        points = PdfImageProcessor.get_point_from_pdf_image(images, area_threshold=10)

        logger.info(f"添加pdf点到image: {len(points)} 个点")
        for point in points:
            new_point = MathUtils.scale_point(point, image_scalers)
            cv2.circle(threshold, new_point, point_size, (255, 255, 255), thickness)

        return threshold


    @staticmethod
    def convert_png_to_jpg(image_file, remove_src=True):
        """
        转换图片文件格式

        :param image_file:
        :param remove_src:
        :return:
        """
        if not os.path.splitext(image_file)[1] == '.png':
            return image_file

        image = cv2.imread(image_file)
        file_name = image_file.replace(".png", ".jpg")
        cv2.imwrite(file_name, image)

        if remove_src:
            FileUtils.delete_file(image_file)

        return file_name

    @staticmethod
    def convert_pdf_to_image(file_name, image_name=None, force_convert=False) -> Tuple[str, bool]:
        """
        转换PDF 为图片

        :param file_name:
        :param image_name:
        :param force_convert:
        :return:
        """
        pdf_converter = GhostscriptBackend()
        do_convert = False
        if image_name is None:
            image_name = FileUtils.get_pdf_to_image_file_name(file_name)

        is_pdf = FileUtils.is_pdf_file(file_name)
        if is_pdf and FileUtils.check_file_exists(file_name):
            if force_convert or not FileUtils.check_file_exists(image_name):
                pdf_converter.convert(file_name, image_name)
                do_convert = True

        return image_name, do_convert
