#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_infer_utils.py
# @Author  ：cycloneboy
# @Date    ：20xx/10/28 14:56
import argparse
import ast
import copy
import os
import random
from typing import List, Union

import PIL
import cv2
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pdftable.entity import OcrCell
from pdftable.utils import BaseUtil, FileUtils, Constants, CommonUtils
from pdftable.utils.colormap import colormap
from pdftable.utils.ocr import OcrCommonUtils

"""
OCR 推理工具
"""


class OcrInferUtils(BaseUtil):

    def init(self):
        pass

    @staticmethod
    def init_args():
        def str2bool(v):
            return v.lower() in ("true", "t", "1")

        parser = argparse.ArgumentParser()
        # params for prediction engine
        parser.add_argument("--use_gpu", type=str2bool, default=True)
        parser.add_argument("--use_onnx", type=str2bool, default=False)
        # parser.add_argument("--ir_optim", type=str2bool, default=True)
        # parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--fp16", type=str2bool, default=True)
        parser.add_argument("--gpu_mem", type=int, default=500)

        # params for text detector
        parser.add_argument("--image_dir", type=str)
        parser.add_argument("--page_num", type=int, default=0)
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_path", type=str)
        parser.add_argument("--det_limit_side_len", type=float, default=960)
        parser.add_argument("--det_limit_type", type=str, default='max')
        parser.add_argument("--det_box_type", type=str, default='quad')

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
        parser.add_argument("--max_batch_size", type=int, default=10)
        parser.add_argument("--use_dilation", type=str2bool, default=False)
        parser.add_argument("--det_db_score_mode", type=str, default="fast")

        # EAST parmas
        parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
        parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
        parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

        # SAST parmas
        parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
        parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
        parser.add_argument("--det_sast_polygon", type=str2bool, default=False)

        # PSE parmas
        parser.add_argument("--det_pse_thresh", type=float, default=0)
        parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
        parser.add_argument("--det_pse_min_area", type=float, default=16)
        parser.add_argument("--det_pse_box_type", type=str, default='box')
        parser.add_argument("--det_pse_scale", type=int, default=1)

        # FCE parmas
        parser.add_argument("--scales", type=list, default=[8, 16, 32])
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--fourier_degree", type=int, default=5)
        parser.add_argument("--det_fce_box_type", type=str, default='poly')

        # params for text recognizer
        parser.add_argument("--rec_algorithm", type=str, default='CRNN')
        parser.add_argument("--rec_model_path", type=str)
        parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
        parser.add_argument("--rec_char_type", type=str, default='ch')
        parser.add_argument("--rec_batch_num", type=int, default=6)
        parser.add_argument("--max_text_length", type=int, default=25)

        parser.add_argument("--use_space_char", type=str2bool, default=True)
        parser.add_argument("--drop_score", type=float, default=0.5)
        parser.add_argument("--limited_max_width", type=int, default=1280)
        parser.add_argument("--limited_min_width", type=int, default=16)

        parser.add_argument(
            "--vis_font_path", type=str,
            default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 'doc/fonts/simfang.ttf'))
        parser.add_argument(
            "--rec_char_dict_path",
            type=str,
            default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 'utils/ocr/other_dict/ppocr_keys_v1.txt'))

        # params for text classifier
        parser.add_argument("--use_angle_cls", type=str2bool, default=False)
        parser.add_argument("--cls_model_path", type=str)
        parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
        parser.add_argument("--label_list", type=list, default=['0', '180'])
        parser.add_argument("--cls_batch_num", type=int, default=6)
        parser.add_argument("--cls_thresh", type=float, default=0.9)

        parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
        parser.add_argument("--use_pdserving", type=str2bool, default=False)

        # params for e2e
        parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
        parser.add_argument("--e2e_model_path", type=str)
        parser.add_argument("--e2e_limit_side_len", type=float, default=768)
        parser.add_argument("--e2e_limit_type", type=str, default='max')

        # PGNet parmas
        parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
        parser.add_argument(
            "--e2e_char_dict_path", type=str,
            default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 'pytorchocr/utils/ic15_dict.txt'))
        parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
        parser.add_argument("--e2e_pgnet_polygon", type=bool, default=True)
        parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

        # params .yaml
        parser.add_argument("--det_yaml_path", type=str, default=None)
        parser.add_argument("--rec_yaml_path", type=str, default=None)
        parser.add_argument("--cls_yaml_path", type=str, default=None)
        parser.add_argument("--e2e_yaml_path", type=str, default=None)
        parser.add_argument("--structure_yaml_path", type=str, default=None)
        parser.add_argument("--layout_yaml_path", type=str, default=None)
        parser.add_argument("--kie_ser_yaml_path", type=str, default=None)
        parser.add_argument("--kie_re_yaml_path", type=str, default=None)

        # multi-process
        parser.add_argument("--use_mp", type=str2bool, default=False)
        parser.add_argument("--total_process_num", type=int, default=1)
        parser.add_argument("--process_id", type=int, default=0)

        parser.add_argument("--benchmark", type=str2bool, default=False)
        parser.add_argument("--save_log_path", type=str, default="./log_output/")

        parser.add_argument("--show_log", type=str2bool, default=True)
        parser.add_argument("--output_dir", type=str, default="./outputs")
        parser.add_argument("--do_transform",
                            type=str,
                            default=True,
                            help="Whether to transform paddle model to pytorch .")

        # params for table structure
        parser.add_argument("--table_max_len", type=int, default=488)
        parser.add_argument("--table_algorithm", type=str, default='TableAttn')
        parser.add_argument("--table_model_path", type=str)
        parser.add_argument(
            "--merge_no_span_structure", type=str2bool, default=True)
        parser.add_argument(
            "--table_char_dict_path",
            type=str,
            default="../ppocr/utils/dict/table_structure_dict_ch.txt")
        # params for layout
        parser.add_argument("--layout_model_dir", type=str)
        parser.add_argument("--layout_algorithm", type=str, default='picodet_lcnet_x1_0_layout')
        parser.add_argument(
            "--layout_dict_path",
            type=str,
            default="../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt")
        parser.add_argument(
            "--layout_score_threshold",
            type=float,
            default=0.5,
            help="Threshold of score.")
        parser.add_argument(
            "--layout_nms_threshold",
            type=float,
            default=0.5,
            help="Threshold of nms.")
        # params for kie
        parser.add_argument("--kie_algorithm", type=str, default='LayoutXLM')
        parser.add_argument("--ser_model_dir", type=str)
        parser.add_argument("--re_model_dir", type=str)
        parser.add_argument("--use_visual_backbone", type=str2bool, default=True)
        parser.add_argument(
            "--ser_dict_path",
            type=str,
            default="../train_data/XFUND/class_list_xfun.txt")
        # need to be None or tb-yx
        parser.add_argument("--ocr_order_method", type=str, default=None)
        # params for inference
        parser.add_argument(
            "--mode",
            type=str,
            choices=['structure', 'kie'],
            default='structure',
            help='structure and kie is supported')
        parser.add_argument(
            "--image_orientation",
            type=bool,
            default=False,
            help='Whether to enable image orientation recognition')
        parser.add_argument(
            "--layout",
            type=str2bool,
            default=True,
            help='Whether to enable layout analysis')
        parser.add_argument(
            "--table",
            type=str2bool,
            default=True,
            help='In the forward, whether the table area uses table recognition')
        parser.add_argument(
            "--ocr",
            type=str2bool,
            default=True,
            help='In the forward, whether the non-table area is recognition by ocr')
        # param for recovery
        parser.add_argument(
            "--recovery",
            type=str2bool,
            default=False,
            help='Whether to enable layout of recovery')
        parser.add_argument(
            "--use_pdf2docx_api",
            type=str2bool,
            default=False,
            help='Whether to use pdf2docx api')
        # detection
        parser.add_argument(
            "--draw_threshold",
            type=float,
            default=0.5,
            help="Threshold to reserve the result for visualization.")

        parser.add_argument(
            "--slim_config",
            default=None,
            type=str,
            help="Configuration file of slim method.")
        parser.add_argument(
            "--use_vdl",
            type=bool,
            default=False,
            help="Whether to record the data to VisualDL.")
        parser.add_argument(
            '--vdl_log_dir',
            type=str,
            default="vdl_log_dir/image",
            help='VisualDL logging directory for image.')
        parser.add_argument(
            "--save_results",
            type=bool,
            default=False,
            help="Whether to save inference results to output_dir.")
        parser.add_argument(
            "--overlap_ratio",
            nargs='+',
            type=float,
            default=[0.25, 0.25],
            help="Overlap height ratio of the sliced image.")
        parser.add_argument(
            "--combine_method",
            type=str,
            default='nms',
            help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
        )
        parser.add_argument(
            "--match_threshold",
            type=float,
            default=0.6,
            help="Combine method matching threshold.")
        parser.add_argument(
            "--match_metric",
            type=str,
            default='ios',
            help="Combine method matching metric, choose in ['iou', 'ios'].")
        parser.add_argument(
            "--slice_infer",
            action='store_true',
            help="Whether to slice the image and merge the inference results for small object detection."
        )
        parser.add_argument(
            "--visualize",
            type=ast.literal_eval,
            default=True,
            help="Whether to save visualize results to output_dir.")

        parser.add_argument(
            "--predict_labels",
            type=str,
            default=None,
            help="predict labels file")

        # OCR CMD
        parser.add_argument("--lang", type=str, default='ch')
        parser.add_argument("--det", type=str2bool, default=True)
        parser.add_argument("--rec", type=str2bool, default=True)
        parser.add_argument("--type", type=str, default='ocr')
        parser.add_argument(
            "--ocr_version",
            type=str,
            choices=['PP-OCR', 'PP-OCRv2', 'PP-OCRv3'],
            default='PP-OCRv3',
            help='OCR Model version, the current model support list is as follows: '
                 '1. PP-OCRv3 Support Chinese and English detection and recognition model, and direction classifier model'
                 '2. PP-OCRv2 Support Chinese detection and recognition model. '
                 '3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
        )
        parser.add_argument(
            "--structure_version",
            type=str,
            choices=['PP-Structure', 'PP-StructureV2'],
            default='PP-StructureV2',
            help='Model version, the current model support list is as follows:'
                 ' 1. PP-Structure Support en table structure model.'
                 ' 2. PP-StructureV2 Support ch and en table structure model.')

        return parser

    @staticmethod
    def parse_args():
        parser = OcrInferUtils.init_args()
        return parser.parse_args()

    @staticmethod
    def get_default_config(args):
        return vars(args)

    @staticmethod
    def read_network_config_from_yaml(yaml_path):
        res = OcrInferUtils.read_ocr_config_from_yaml(yaml_path)
        return res['Architecture']

    @staticmethod
    def read_ocr_config_from_yaml(yaml_path):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        import yaml
        with open(yaml_path, encoding='utf-8') as f:
            res = yaml.safe_load(f)
        if res.get('Architecture') is None:
            raise ValueError('{} has no Architecture'.format(yaml_path))
        return res

    @staticmethod
    def draw_e2e_res(dt_boxes, strs, img_path):
        src_im = cv2.imread(img_path)
        for box, str in zip(dt_boxes, strs):
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            cv2.putText(
                src_im,
                str,
                org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=1)
        return src_im

    @staticmethod
    def draw_text_det_res(dt_boxes, img_path):
        src_im = cv2.imread(img_path)
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        return src_im

    @staticmethod
    def resize_img(img, input_size=600):
        """
        resize img and limit the longest side of the image to input_size
        """
        img = np.array(img)
        im_shape = img.shape
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(input_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
        return img

    @staticmethod
    def draw_ocr_box_txt(image,
                         boxes,
                         txts,
                         scores=None,
                         drop_score=0.5,
                         font_path="./doc/simfang.ttf",
                         table_structure_result: Union[List, np.ndarray] = None,
                         table_cell_result: Union[List, np.ndarray] = None,
                         layout_result: List = None, ):
        h, w = image.height, image.width
        img_left = image.copy()
        img_right = Image.new('RGB', (w, h), (255, 255, 255))

        # 绘制表格结构识别
        if table_structure_result is not None:
            img_right = OcrCommonUtils.draw_boxes(image_full=img_right,
                                                  det_result=table_structure_result,
                                                  color="red", width=3, return_numpy=False)

        # 绘制表格识别
        if table_cell_result is not None and len(table_cell_result) > 0 and table_cell_result[0]["table_cells"] is not None:
            img_right = OcrCommonUtils.draw_table_cell_boxes(image_full=img_right,
                                                             cell_result=table_cell_result,
                                                             color="red", width=3, return_numpy=False)

        # 绘制版面分析结果
        if layout_result is not None:
            img_right = OcrInferUtils.draw_text_layout_res(img=img_right, layout_res=layout_result)

        import random

        random.seed(0)
        draw_left = ImageDraw.Draw(img_left)
        draw_right = ImageDraw.Draw(img_right)
        for idx, (box, txt) in enumerate(zip(boxes, txts)):
            if scores is not None and scores[idx] < drop_score:
                continue
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw_left.polygon(box, fill=color)
            draw_right.polygon(
                [
                    box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                    box[2][1], box[3][0], box[3][1]
                ],
                outline=color)
            box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
            box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
            if box_height > 2 * box_width:
                font_size = max(int(box_width * 0.9), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                cur_y = box[0][1]
                for c in txt:
                    if hasattr(font, "getsize"):
                        char_size = font.getsize(c)
                    else:
                        char_size = font.getbbox(c)[2:]
                    draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                    cur_y += char_size[1]
            else:
                font_size = max(int(box_height * 0.8), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)

        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(img_right, (w, 0, w * 2, h))
        return np.array(img_show)

    @staticmethod
    def str_count(s):
        """
        Count the number of Chinese characters,
        a single English character and a single number
        equal to half the length of Chinese characters.
        args:
            s(string): the input of string
        return(int):
            the number of Chinese characters
        """
        import string
        count_zh = count_pu = 0
        s_len = len(s)
        en_dg_count = 0
        for c in s:
            if c in string.ascii_letters or c.isdigit() or c.isspace():
                en_dg_count += 1
            elif c.isalpha():
                count_zh += 1
            else:
                count_pu += 1
        return s_len - math.ceil(en_dg_count / 2)

    @staticmethod
    def text_visual(texts,
                    scores,
                    img_h=400,
                    img_w=600,
                    threshold=0.,
                    font_path="./doc/simfang.ttf"):
        """
        create new blank img and draw txt on it
        args:
            texts(list): the text will be draw
            scores(list|None): corresponding score of each txt
            img_h(int): the height of blank img
            img_w(int): the width of blank img
            font_path: the path of font which is used to draw text
        return(array):
        """
        if scores is not None:
            assert len(texts) == len(
                scores), "The number of txts and corresponding scores must match"

        def create_blank_img():
            blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
            blank_img[:, img_w - 1:] = 0
            blank_img = Image.fromarray(blank_img).convert("RGB")
            draw_txt = ImageDraw.Draw(blank_img)
            return blank_img, draw_txt

        blank_img, draw_txt = create_blank_img()

        font_size = 20
        txt_color = (0, 0, 0)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

        gap = font_size + 5
        txt_img_list = []
        count, index = 1, 0
        for idx, txt in enumerate(texts):
            index += 1
            if scores[idx] < threshold or math.isnan(scores[idx]):
                index -= 1
                continue
            first_line = True
            while OcrInferUtils.str_count(txt) >= img_w // font_size - 4:
                tmp = txt
                txt = tmp[:img_w // font_size - 4]
                if first_line:
                    new_txt = str(index) + ': ' + txt
                    first_line = False
                else:
                    new_txt = '    ' + txt
                draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
                txt = tmp[img_w // font_size - 4:]
                if count >= img_h // gap - 1:
                    txt_img_list.append(np.array(blank_img))
                    blank_img, draw_txt = create_blank_img()
                    count = 0
                count += 1
            if first_line:
                new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
            else:
                new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            # whether add new blank img or not
            if count >= img_h // gap - 1 and idx + 1 < len(texts):
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        txt_img_list.append(np.array(blank_img))
        if len(txt_img_list) == 1:
            blank_img = np.array(txt_img_list[0])
        else:
            blank_img = np.concatenate(txt_img_list, axis=1)
        return np.array(blank_img)

    @staticmethod
    def base64_to_cv2(b64str):
        import base64
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.fromstring(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data

    @staticmethod
    def draw_boxes(image, boxes, scores=None, drop_score=0.5):
        image = cv2.imread(image) if isinstance(image, str) else image
        if scores is None:
            scores = [1] * len(boxes)
        for (box, score) in zip(boxes, scores):
            if score < drop_score:
                continue
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            cv2.polylines(image, [box], True, (0, 0, 255), 4)
        return image

    @staticmethod
    def draw_rectangle(img_path, boxes):
        boxes = np.array(boxes)
        img = cv2.imread(img_path) if isinstance(img_path, str) else img_path
        img_show = img.copy()
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 0, 255), 4)
        return img_show

    @staticmethod
    def sorted_boxes(dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    @staticmethod
    def draw_structure_result(image, result, font_path):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        boxes, txts, scores = [], [], []

        img_layout = image.copy()
        draw_layout = ImageDraw.Draw(img_layout)
        text_color = (255, 255, 255)
        text_background_color = (80, 127, 255)
        catid2color = {}
        font_size = 15
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

        for region in result:
            if region['type'] not in catid2color:
                box_color = (random.randint(0, 255), random.randint(0, 255),
                             random.randint(0, 255))
                catid2color[region['type']] = box_color
            else:
                box_color = catid2color[region['type']]
            box_layout = region['bbox']
            draw_layout.rectangle(
                [(box_layout[0], box_layout[1]), (box_layout[2], box_layout[3])],
                outline=box_color,
                width=3)
            text_w, text_h = font.getsize(region['type'])
            draw_layout.rectangle(
                [(box_layout[0], box_layout[1]),
                 (box_layout[0] + text_w, box_layout[1] + text_h)],
                fill=text_background_color)
            draw_layout.text(
                (box_layout[0], box_layout[1]),
                region['type'],
                fill=text_color,
                font=font)

            if region['type'] == 'table':
                pass
            else:
                for text_result in region['res']:
                    boxes.append(np.array(text_result['text_region']))
                    txts.append(text_result['text'])
                    scores.append(text_result['confidence'])

        im_show = OcrInferUtils.draw_ocr_box_txt(
            img_layout, boxes, txts, scores, font_path=font_path, drop_score=0)
        return im_show

    @staticmethod
    def kie_ser_re_make_input(ser_inputs, ser_results):
        """
        kie_ser_re

        :param ser_inputs:
        :param ser_results:
        :return:
        """
        entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}
        batch_size, max_seq_len = ser_inputs[0].shape[:2]
        entities = ser_inputs[8][0]
        ser_results = ser_results[0]
        assert len(entities) == len(ser_results)

        # entities
        start = []
        end = []
        label = []
        entity_idx_dict = {}
        for i, (res, entity) in enumerate(zip(ser_results, entities)):
            if res['pred'] == 'O':
                continue
            entity_idx_dict[len(start)] = i
            start.append(entity['start'])
            end.append(entity['end'])
            label.append(entities_labels[res['pred']])

        entities = np.full([max_seq_len + 1, 3], fill_value=-1, dtype=np.int64)
        entities[0, 0] = len(start)
        entities[1:len(start) + 1, 0] = start
        entities[0, 1] = len(end)
        entities[1:len(end) + 1, 1] = end
        entities[0, 2] = len(label)
        entities[1:len(label) + 1, 2] = label

        # relations
        head = []
        tail = []
        for i in range(len(label)):
            for j in range(len(label)):
                if label[i] == 1 and label[j] == 2:
                    head.append(i)
                    tail.append(j)

        relations = np.full([len(head) + 1, 2], fill_value=-1, dtype=np.int64)
        relations[0, 0] = len(head)
        relations[1:len(head) + 1, 0] = head
        relations[0, 1] = len(tail)
        relations[1:len(tail) + 1, 1] = tail

        entities = np.expand_dims(entities, axis=0)
        entities = np.repeat(entities, batch_size, axis=0)
        relations = np.expand_dims(relations, axis=0)
        relations = np.repeat(relations, batch_size, axis=0)

        # remove ocr_info segment_offset_id and label in ser input
        if isinstance(ser_inputs[0], torch.Tensor):
            entities = torch.LongTensor(entities)
            relations = torch.LongTensor(relations)
        ser_inputs = ser_inputs[:5] + [entities, relations]

        entity_idx_dict_batch = []
        for b in range(batch_size):
            entity_idx_dict_batch.append(entity_idx_dict)
        return ser_inputs, entity_idx_dict_batch

    @staticmethod
    def kie_ser_make_input(data):
        import numbers
        from collections import defaultdict
        data_dict = defaultdict(list)
        to_tensor_idxs = []

        for idx, v in enumerate(data):
            if isinstance(v, (np.ndarray, torch.Tensor, numbers.Number)):
                if idx not in to_tensor_idxs:
                    to_tensor_idxs.append(idx)
            data_dict[idx].append(v)
        for idx in to_tensor_idxs:
            if idx in [0, 1, 2, 3, ]:
                data_dict[idx] = torch.LongTensor(data_dict[idx])
            else:
                data_dict[idx] = torch.Tensor(data_dict[idx])
        return list(data_dict.values())

    @staticmethod
    def show_compare_result(image_full, ocr_result: List[OcrCell],
                            table_structure_result: Union[List, np.ndarray] = None,
                            table_cell_result: Union[List, np.ndarray] = None,
                            layout_result: List = None,
                            lang: str = "ch"
                            ):
        """
        显示OCR 识别结果 左右对比

        :param image_full:
        :param ocr_result:
        :param table_structure_result:
        :param table_cell_result:
        :param layout_result:
        :return:
        """
        image = Image.fromarray(image_full) if isinstance(image_full, np.ndarray) else image_full

        boxes = []
        txts = []
        scores = []

        for item in ocr_result:
            boxes.append(item.bbox)
            txts.append(item.text)
            scores.append(0.9)

        font_name = Constants.FONT_CONFIG.get(lang, "chinese_cht.ttf")
        model_dir = CommonUtils.download_model_from_hub("cycloneboy/pdftable_config")

        font_path = os.path.join(model_dir,"fonts", font_name)
        draw_img = OcrInferUtils.draw_ocr_box_txt(image,
                                                  boxes,
                                                  txts,
                                                  scores,
                                                  font_path=font_path,
                                                  table_structure_result=table_structure_result,
                                                  table_cell_result=table_cell_result,
                                                  layout_result=layout_result)

        return draw_img

    @staticmethod
    def draw_text_layout_res(img, layout_res: list, save_path: str = None, return_new_image=True):
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.5
        font_color = (0, 0, 255)
        font_thickness = 1

        catid2color = {}
        color_list = colormap(rgb=True)[:40]

        tmp_img = copy.deepcopy(img) if return_new_image else img
        if isinstance(tmp_img, PIL.Image.Image):
            tmp_img = np.array(tmp_img)
        for v in layout_res:
            bbox = np.round(v['bbox']).astype(np.int32)
            label = v['label']
            score = v['score']
            catid = v['category_id']

            if catid not in catid2color:
                idx = np.random.randint(len(color_list))
                catid2color[catid] = color_list[idx]
            color = tuple([round(item) for item in catid2color[catid]])

            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])

            # print(f"color: {color}")
            cv2.rectangle(tmp_img, start_point, end_point, color[0], 2)

            text = "{} {:.2f}".format(label, score)

            (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            put_point = start_point[0], start_point[1]

            text_start_point = (bbox[0], bbox[1] - h)
            text_end_point = (bbox[0] + w, bbox[1])
            cv2.rectangle(tmp_img, text_start_point, text_end_point, color[0], 2)
            cv2.putText(tmp_img, text, put_point, font, font_scale,
                        font_color, font_thickness)
        if save_path is not None:
            FileUtils.check_file_exists(save_path)
            cv2.imwrite(save_path, tmp_img)
            # print(f'The infer result has saved in {save_path}')

        if isinstance(img, PIL.Image.Image) and isinstance(tmp_img, np.ndarray):
            tmp_img = Image.fromarray(tmp_img)
        return tmp_img
