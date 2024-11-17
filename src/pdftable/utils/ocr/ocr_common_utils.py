#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_common_utils.py
# @Author  ：cycloneboy
# @Date    ：20xx/10/28 11:09

import math
import os
from typing import List

import PIL
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

from pdftable.utils import BaseUtil, logger, FileUtils

"""
OCR 基础工具
"""

__all__ = [
    'OcrCommonUtils'
]


class OcrCommonUtils(BaseUtil):

    def init(self):
        pass

    @staticmethod
    def get_rotate_crop_image(img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    @staticmethod
    def get_minarea_rect_crop(img, points):
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = OcrCommonUtils.get_rotate_crop_image(img, np.array(box))
        return crop_img

    @staticmethod
    def load_vqa_bio_label_maps(label_map_path):
        with open(label_map_path, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
        old_lines = [line.strip() for line in lines]
        lines = ["O"]
        for line in old_lines:
            # "O" has already been in lines
            if line.upper() in ["OTHER", "OTHERS", "IGNORE"]:
                continue
            lines.append(line)
        labels = ["O"]
        for line in lines[1:]:
            labels.append("B-" + line)
            labels.append("I-" + line)
        label2id_map = {label.upper(): idx for idx, label in enumerate(labels)}
        id2label_map = {idx: label.upper() for idx, label in enumerate(labels)}
        return label2id_map, id2label_map

    @staticmethod
    def print_dict(d, logger, delimiter=0):
        """
        Recursively visualize a dict and
        indenting acrrording by the relationship of keys.
        """
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                logger.info("{}{} : ".format(delimiter * " ", str(k)))
                OcrCommonUtils.print_dict(v, logger, delimiter + 4)
            elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
                logger.info("{}{} : ".format(delimiter * " ", str(k)))
                for value in v:
                    OcrCommonUtils.print_dict(value, logger, delimiter + 4)
            else:
                logger.info("{}{} : {}".format(delimiter * " ", k, v))

    @staticmethod
    def get_check_global_params(mode):
        check_params = ['use_gpu', 'max_text_length', 'image_shape', \
                        'image_shape', 'character_type', 'loss_type']
        if mode == "train_eval":
            check_params = check_params + [ \
                'train_batch_size_per_card', 'test_batch_size_per_card']
        elif mode == "test":
            check_params = check_params + ['test_batch_size_per_card']
        return check_params

    @staticmethod
    def check_image_file(path):
        img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
        return any([path.lower().endswith(e) for e in img_end])

    @staticmethod
    def get_image_file_list(img_file):
        imgs_lists = []
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
        if os.path.isfile(img_file) and OcrCommonUtils.check_image_file(img_file):
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for single_file in os.listdir(img_file):
                file_path = os.path.join(img_file, single_file)
                if os.path.isfile(file_path) and OcrCommonUtils.check_image_file(file_path):
                    imgs_lists.append(file_path)
        if len(imgs_lists) == 0:
            raise Exception("not found any img file in {}".format(img_file))
        imgs_lists = sorted(imgs_lists)
        return imgs_lists

    @staticmethod
    def check_and_read(img_path):
        """
        检查和读取 图片

        :param img_path:
        :return:
        """
        if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
            gif = cv2.VideoCapture(img_path)
            ret, frame = gif.read()
            if not ret:
                logger.info("Cannot read {}. This gif image maybe corrupted.")
                return None, False
            if len(frame.shape) == 2 or frame.shape[-1] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            imgvalue = frame[:, :, ::-1]
            return imgvalue, True, False
        elif os.path.basename(img_path)[-3:] in ['pdf']:
            import fitz
            from PIL import Image
            imgs = []
            with fitz.open(img_path) as pdf:
                if hasattr(pdf, 'pageCount'):
                    page_count = pdf.pageCount
                elif hasattr(pdf, 'page_count'):
                    page_count = pdf.page_count
                else:
                    page_count = len(pdf)
                for pg in range(0, page_count):
                    page = pdf[pg]
                    mat = fitz.Matrix(2, 2)
                    if hasattr(page, "getPixmap"):
                        pm = page.getPixmap(matrix=mat, alpha=False)
                    else:
                        pm = page.get_pixmap(matrix=mat, alpha=False)

                    # if width or height > 2000 pixels, don't enlarge the image
                    if pm.width > 2000 or pm.height > 2000:
                        if hasattr(page, "getPixmap"):
                            pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                        else:
                            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    imgs.append(img)
                return imgs, False, True
        return None, False, False

    @staticmethod
    def crop_image(img, position):
        """
        scope ocr 透视变换
            https://modelscope.cn/studios/damo/cv_ocr-text-spotting/file/view/master/app.py

        :param img:
        :param position:
        :return:
        """

        def distance(x1, y1, x2, y2):
            return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

        position = position.tolist()
        for i in range(4):
            for j in range(i + 1, 4):
                if (position[i][0] > position[j][0]):
                    tmp = position[j]
                    position[j] = position[i]
                    position[i] = tmp
        if position[0][1] > position[1][1]:
            tmp = position[0]
            position[0] = position[1]
            position[1] = tmp

        if position[2][1] > position[3][1]:
            tmp = position[2]
            position[2] = position[3]
            position[3] = tmp

        x1, y1 = position[0][0], position[0][1]
        x2, y2 = position[2][0], position[2][1]
        x3, y3 = position[3][0], position[3][1]
        x4, y4 = position[1][0], position[1][1]

        corners = np.zeros((4, 2), np.float32)
        corners[0] = [x1, y1]
        corners[1] = [x2, y2]
        corners[2] = [x4, y4]
        corners[3] = [x3, y3]

        img_width = distance((x1 + x4) / 2, (y1 + y4) / 2, (x2 + x3) / 2, (y2 + y3) / 2)
        img_height = distance((x1 + x2) / 2, (y1 + y2) / 2, (x4 + x3) / 2, (y4 + y3) / 2)

        corners_trans = np.zeros((4, 2), np.float32)
        corners_trans[0] = [0, 0]
        corners_trans[1] = [img_width - 1, 0]
        corners_trans[2] = [0, img_height - 1]
        corners_trans[3] = [img_width - 1, img_height - 1]

        transform = cv2.getPerspectiveTransform(corners, corners_trans)
        dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
        return dst

    @staticmethod
    def crop_image_by_box(img, bbox, save_name=None, diff=0):
        """
        根据矩形框剪切图片

        :param img:
        :param bbox:
        :param save_name:
        :param diff:
        :return:
        """
        x1, y1, x2, y2 = bbox
        cropped = img[round(y1 - diff):round(y2 + diff), round(x1 - diff):round(x2 + diff)]  # 裁剪坐标为[y0:y1, x0:x1]

        if save_name:
            cv2.imwrite(save_name, cropped)
        return cropped

    @staticmethod
    def order_point(coor):
        """
        scope ocr
            https://modelscope.cn/studios/damo/cv_ocr-text-spotting/file/view/master/app.py

        :param coor:
        :return:
        """
        arr = np.array(coor).reshape([4, 2])
        sum_ = np.sum(arr, 0)
        centroid = sum_ / arr.shape[0]
        theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
        sort_points = arr[np.argsort(theta)]
        sort_points = sort_points.reshape([4, -1])
        if sort_points[0][0] > centroid[0]:
            sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
        sort_points = sort_points.reshape([4, 2]).astype('float32')
        return sort_points

    @staticmethod
    def draw_boxes(image_full, det_result, color="green", width=5, return_numpy=True):
        """
        scope ocr
            https://modelscope.cn/studios/damo/cv_ocr-text-spotting/file/view/master/app.py


        :param image_full:
        :param det_result:
        :param color:
        :param width:
        :param return_numpy:
        :return:
        """
        if isinstance(image_full, np.ndarray):
            image_full = Image.fromarray(image_full)

        draw = ImageDraw.Draw(image_full)
        for i in range(det_result.shape[0]):
            # import pdb; pdb.set_trace()
            p0, p1, p2, p3 = OcrCommonUtils.order_point(det_result[i])
            draw.text((p0[0] + 5, p0[1] + 5), str(i + 1), fill='blue', align='center')
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        if return_numpy:
            image_full = np.array(image_full)
        return image_full

    @staticmethod
    def draw_table_cell_boxes(image_full, cell_result, color="green", width=5, return_numpy=True):
        """
        绘制表格cell

        :param image_full:
        :param cell_result:
        :param color:
        :param width:
        :param return_numpy:
        :return:
        """
        det_result = []

        for table in cell_result:
            table_cells = table["table_cells"]

            one_table_result = []
            if table_cells:
                for index, columns in table_cells.items():
                    for cell in columns:
                        one_table_result.append(cell.point_array())

            det_result.extend(one_table_result)

        image_draw = OcrCommonUtils.draw_boxes(image_full=image_full,
                                               det_result=np.array(det_result),
                                               color=color, width=width,
                                               return_numpy=return_numpy)

        return image_draw

    @staticmethod
    def read_image(image_path_or_url, return_nparray=False):
        image_content = image_path_or_url
        if isinstance(image_content, str):
            if image_path_or_url.startswith("http"):
                image_content = requests.get(image_path_or_url, stream=True).raw

            image = Image.open(image_content)
            image = image.convert("RGB")
        elif isinstance(image_content, PIL.Image.Image):
            image = image_content.convert("RGB")
        elif isinstance(image_content, np.ndarray):
            if len(image_content.shape) == 2:
                image = cv2.cvtColor(image_content, cv2.COLOR_GRAY2RGB)
            else:
                image = image_content

        if return_nparray:
            image = np.array(image)

        return image

    @staticmethod
    def parse_lang_ppocr(lang):
        """
        解析语言
            --  中文	chinese and english	ch		保加利亚文	Bulgarian	bg
                英文	english	en		乌克兰文	Ukranian	uk
                法文	french	fr		白俄罗斯文	Belarusian	be
                德文	german	german		泰卢固文	Telugu	te
                日文	japan	japan		阿巴扎文	Abaza	abq
                韩文	korean	korean		泰米尔文	Tamil	ta
                中文繁体	chinese traditional	chinese_cht		南非荷兰文	Afrikaans	af
                意大利文	Italian	it		阿塞拜疆文	Azerbaijani	az
                西班牙文	Spanish	es		波斯尼亚文	Bosnian	bs
                葡萄牙文	Portuguese	pt		捷克文	Czech	cs
                俄罗斯文	Russia	ru		威尔士文	Welsh	cy
                阿拉伯文	Arabic	ar		丹麦文	Danish	da
                印地文	Hindi	hi		爱沙尼亚文	Estonian	et
                维吾尔	Uyghur	ug		爱尔兰文	Irish	ga
                波斯文	Persian	fa		克罗地亚文	Croatian	hr
                乌尔都文	Urdu	ur		匈牙利文	Hungarian	hu
                塞尔维亚文（latin)	Serbian(latin)	rs_latin		印尼文	Indonesian	id
                欧西坦文	Occitan	oc		冰岛文	Icelandic	is
                马拉地文	Marathi	mr		库尔德文	Kurdish	ku
                尼泊尔文	Nepali	ne		立陶宛文	Lithuanian	lt
                塞尔维亚文（cyrillic)	Serbian(cyrillic)	rs_cyrillic		拉脱维亚文	Latvian	lv
                毛利文	Maori	mi		达尔瓦文	Dargwa	dar
                马来文	Malay	ms		因古什文	Ingush	inh
                马耳他文	Maltese	mt		拉克文	Lak	lbe
                荷兰文	Dutch	nl		莱兹甘文	Lezghian	lez
                挪威文	Norwegian	no		塔巴萨兰文	Tabassaran	tab
                波兰文	Polish	pl		比尔哈文	Bihari	bh
                罗马尼亚文	Romanian	ro		迈蒂利文	Maithili	mai
                斯洛伐克文	Slovak	sk		昂加文	Angika	ang
                斯洛文尼亚文	Slovenian	sl		孟加拉文	Bhojpuri	bho
                阿尔巴尼亚文	Albanian	sq		摩揭陀文	Magahi	mah
                瑞典文	Swedish	sv		那格浦尔文	Nagpur	sck
                西瓦希里文	Swahili	sw		尼瓦尔文	Newari	new
                塔加洛文	Tagalog	tl		保加利亚文	Goan Konkani	gom
                土耳其文	Turkish	tr		沙特阿拉伯文	Saudi Arabia	sa
                乌兹别克文	Uzbek	uz		阿瓦尔文	Avar	ava
                越南文	Vietnamese	vi		阿瓦尔文	Avar	ava
                蒙古文	Mongolian	mn		阿迪赫文	Adyghe	ady
        :param lang:
        :return:
        """
        supported_languages = [
            "ch", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka",
            "latin", "arabic", "cyrillic", "devanagari"
        ]

        latin_lang = [
            'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
            'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
            'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
            'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
        ]
        arabic_lang = ['ar', 'fa', 'ug', 'ur']
        cyrillic_lang = [
            'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
            'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
        ]
        devanagari_lang = [
            'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
            'sa', 'bgc'
        ]
        if lang in latin_lang:
            lang = "latin"
        elif lang in arabic_lang:
            lang = "arabic"
        elif lang in cyrillic_lang:
            lang = "cyrillic"
        elif lang in devanagari_lang:
            lang = "devanagari"
        assert lang in supported_languages, 'param lang must in {}, but got {}'.format(supported_languages, lang)

        if lang == "ch":
            det_lang = "ch"
        elif lang == 'structure':
            det_lang = 'structure'
        elif lang in ["en", "latin"]:
            det_lang = "en"
        else:
            det_lang = "ml"
        return lang, det_lang

    @staticmethod
    def draw_lore_one_cell(image, bbox, logi=None, line_color=(0, 0, 255), thickness=1, txt_color=(0, 255, 255)):
        """
        绘制 lore cell
        :param image:
        :param bbox:
        :param logi:
        :param line_color:
        :param thickness:
        :param txt_color:
        :return:
        """
        bbox = np.array(bbox, dtype=np.int32)

        cv2.line(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), line_color, thickness)
        cv2.line(image, (bbox[2], bbox[3]), (bbox[4], bbox[5]), line_color, thickness)
        cv2.line(image, (bbox[4], bbox[5]), (bbox[6], bbox[7]), line_color, thickness)
        cv2.line(image, (bbox[6], bbox[7]), (bbox[0], bbox[1]), line_color, thickness)

        if logi is not None:
            txt = '{:.0f},{:.0f},{:.0f},{:.0f}'.format(logi[0], logi[1], logi[2], logi[3])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, 0.2, 2)[0]
            cv2.rectangle(image,
                          (bbox[0], bbox[1] - cat_size[1] - 2),
                          (bbox[0] + cat_size[0], bbox[1] - 2), txt_color, -1)
            cv2.putText(image, txt, (bbox[0], bbox[1] - 2),
                        font, 0.2, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)  # 1 - 5 # 0.20 _ 0.60

    @staticmethod
    def draw_lore_bboxes(image_name, boxes: List, logits: List, save_name=None):
        """
        绘制 LORE  预测结果

        :param image_name:
        :param boxes:
        :param logits:
        :param save_name:
        :return:
        """
        if isinstance(image_name, str):
            image = cv2.imread(image_name)
        else:
            image = image_name

        for index, bbox in enumerate(boxes):
            OcrCommonUtils.draw_lore_one_cell(image=image, bbox=bbox, logi=logits[index])

        if save_name is not None:
            FileUtils.check_file_exists(save_name)
            cv2.imwrite(save_name, image)

    @staticmethod
    def box_transform(bbox_list):
        """
        box 格式转换： 两点转四点

        :param bbox_list:
        :return:
        """
        if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
            new_bboxes = []
            for (x1, y1, x2, y2) in bbox_list:
                box = [x1, y1, x2, y1, x2, y2, x1, y2]
                new_bboxes.append(box)
            bbox_list = np.array(new_bboxes)
        return bbox_list

    @staticmethod
    def box_move_point(bbox, x=0, y=0):
        """
        矩形框移动

        :param bbox:
        :param x:
        :param y:
        :return:
        """
        new_bbox = []
        for index in range(0, len(bbox)):
            temp = x if index % 2 == 0 else y
            new_bbox.append(bbox[index] + temp)
        return new_bbox

    @staticmethod
    def box_list_move_point(bbox, x=0, y=0):
        is_np = isinstance(bbox, np.ndarray)
        if is_np:
            bbox = bbox.tolist()
        polygons = [OcrCommonUtils.box_move_point(item, x=x, y=y)
                    for item in bbox]

        if is_np:
            polygons = np.array(polygons)
        return polygons

    @staticmethod
    def box_list_two_point_to_four_point(bbox_list):
        if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
            new_bboxes = []
            for item in bbox_list:
                if len(item) == 0:
                    continue
                (x1, y1, x2, y2) = item
                box = [x1, y1, x2, y1, x2, y2, x1, y2]
                new_bboxes.append(box)
            bbox_list = np.array(new_bboxes)
        return bbox_list
