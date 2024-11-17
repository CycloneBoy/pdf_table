#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：math_utils.py
# @Author  ：cycloneboy
# @Date    ：20xx/1/11 16:01
import traceback
import uuid

import hashlib
from typing import List

from . import logger, BaseUtil, Constants

"""
数学相关工具类
"""


class MathUtils(BaseUtil):
    """
    数学相关工具类
    """

    def init(self):
        pass

    @staticmethod
    def edit_distance_word(word, char_set):
        """
        all edits that are one edit away from 'word'
        :param word:
        :param char_set:
        :return:
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
        return set(transposes + replaces)

    @staticmethod
    def get_sub_array(nums):
        """
        取所有连续子串，
        [0, 1, 2, 5, 7, 8]
        => [[0, 3], 5, [7, 9]]
        :param nums: sorted(list)
        :return:
        """
        ret = []
        ii = 0
        for i, c in enumerate(nums):
            if i == 0:
                pass
            elif i <= ii:
                continue
            elif i == len(nums) - 1:
                ret.append([c])
                break
            ii = i
            cc = c
            # get continuity Substring
            while ii < len(nums) - 1 and nums[ii + 1] == cc + 1:
                ii = ii + 1
                cc = cc + 1
            if ii > i:
                ret.append([c, nums[ii] + 1])
            else:
                ret.append([c])
        return ret

    @staticmethod
    def get_sub_array2(nums):
        """
        取所有连续子串，
        [0, 1, 2, 5, 7, 8]
        => [[0, 3], 5, [7, 9]]
        :param nums: sorted(list)
        :return:
        """
        res = MathUtils.get_sub_array(nums)
        sub_res = [[row[0], row[1] - 1] for row in res if len(row) == 2 and row[1] - row[0] < 4]
        return sub_res

    @staticmethod
    def find_all_idx2(lst, item):
        """
        取列表中指定元素的所有下标
        :param lst: 列表或字符串
        :param item: 指定元素
        :return: 下标列表
        """
        ids = []
        for i in range(len(lst)):
            if item == lst[i]:
                ids.append(i)
        return ids

    @staticmethod
    def find_all_idx(lst, item):
        """
        取列表中指定元素的所有下标
        :param lst: 列表或字符串
        :param item: 指定元素
        :return: 下标列表
        """
        ids = []
        pos = -1
        for i in range(lst.count(item)):
            pos = lst.index(item, pos + 1)
            if pos > -1:
                ids.append(pos)
        return ids

    @staticmethod
    def get_edit_distance(src, target):
        """
        获取编辑距离 Distance with Any Object
         Above libraries only support strings. But Sometimes other type of objects such as list of strings(words).
         I support any iterable, only requires hashable object of it:
        :param src:
        :param target:
        :return:
        """
        import editdistance
        distance = editdistance.eval(src, target)
        return distance

    @staticmethod
    def chinese2digits(uchars_chinese):
        """
            中文数字转换为阿拉伯数字

        :param uchars_chinese:
        :return:
        """
        total = 0
        try:
            if str(uchars_chinese).isdigit():
                return int(str(uchars_chinese).strip())
            r = 1  # 表示单位：个十百千...
            for i in range(len(uchars_chinese) - 1, -1, -1):
                val = Constants.NUMERALS_ZH_DICT.get(uchars_chinese[i])
                if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
                    if val > r:
                        r = val
                        total = total + val
                    else:
                        r = r * val
                        # total =total + r * x
                elif val >= 10:
                    if val > r:
                        r = val
                    else:
                        r = r * val
                else:
                    total = total + r * val
        except Exception as e:
            traceback.print_exc()
            logger.error(f"中文数字转换为阿拉伯数字: {uchars_chinese} -  {e}")
        return total

    @staticmethod
    def digits2chinese(number):
        """
            阿拉伯数字 => 中文数字
        :param number:
        :return:
        """
        import cn2an
        output = cn2an.an2cn(str(number))
        return output

    @staticmethod
    def md5(text_input):
        """
        md5
        :param text_input:
        :return:
        """
        hl = hashlib.md5()
        hl.update(text_input.encode(encoding='utf-8'))
        result = hl.hexdigest()
        return result

    @staticmethod
    def uuid():
        return str(uuid.uuid1()).replace('-', '').replace('_', '')

    @staticmethod
    def translate(x1, x2):
        """Translates x2 by x1.

        Parameters
        ----------
        x1 : float
        x2 : float

        Returns
        -------
        x2 : float

        """
        x2 += x1
        return x2

    @staticmethod
    def scale(x, s):
        """Scales x by scaling factor s.

        Parameters
        ----------
        x : float
        s : float

        Returns
        -------
        x : float

        """
        x *= s
        return x

    @staticmethod
    def scale_pdf(k, factors):
        """Translates and scales pdf coordinate space to image
        coordinate space.

        Parameters
        ----------
        k : tuple
            Tuple (x1, y1, x2, y2) representing table bounding box where
            (x1, y1) -> lt and (x2, y2) -> rb in PDFMiner coordinate
            space.
        factors : tuple
            Tuple (scaling_factor_x, scaling_factor_y, pdf_y) where the
            first two elements are scaling factors and pdf_y is height of
            pdf.

        Returns
        -------
        knew : tuple
            Tuple (x1, y1, x2, y2) representing table bounding box where
            (x1, y1) -> lt and (x2, y2) -> rb in OpenCV coordinate
            space.

        """
        x1, y1, x2, y2 = k
        scaling_factor_x, scaling_factor_y, pdf_y = factors
        x1 = MathUtils.scale(x1, scaling_factor_x)
        y1 = MathUtils.scale(abs(MathUtils.translate(-pdf_y, y1)), scaling_factor_y)
        x2 = MathUtils.scale(x2, scaling_factor_x)
        y2 = MathUtils.scale(abs(MathUtils.translate(-pdf_y, y2)), scaling_factor_y)
        knew = (round(x1), round(y1), round(x2), round(y2))
        return knew

    @staticmethod
    def scale_point(k, factors):
        """Translates and scales pdf coordinate space to image
        coordinate space.

        """
        x1, y1 = k
        scaling_factor_x, scaling_factor_y, pdf_y = factors
        x1 = MathUtils.scale(x1, scaling_factor_x)
        y1 = MathUtils.scale(abs(MathUtils.translate(-pdf_y, y1)), scaling_factor_y)
        knew = (round(x1), round(y1))
        return knew

    @staticmethod
    def scale_image(tables, v_segments, h_segments, factors):
        """Translates and scales image coordinate space to pdf
        coordinate space.

        Parameters
        ----------
        tables : dict
            Dict with table boundaries as keys and list of intersections
            in that boundary as value.
        v_segments : list
            List of vertical line segments.
        h_segments : list
            List of horizontal line segments.
        factors : tuple
            Tuple (scaling_factor_x, scaling_factor_y, img_y) where the
            first two elements are scaling factors and img_y is height of
            image.

        Returns
        -------
        tables_new : dict
        v_segments_new : dict
        h_segments_new : dict

        """
        scaling_factor_x, scaling_factor_y, img_y = factors
        tables_new = {}
        for k in tables.keys():
            x1, y1, x2, y2 = k
            x1 = MathUtils.scale(x1, scaling_factor_x)
            y1 = MathUtils.scale(abs(MathUtils.translate(-img_y, y1)), scaling_factor_y)
            x2 = MathUtils.scale(x2, scaling_factor_x)
            y2 = MathUtils.scale(abs(MathUtils.translate(-img_y, y2)), scaling_factor_y)

            joints = []
            for x, y in tables[k]:
                j_x = MathUtils.scale(x, scaling_factor_x)
                j_y = MathUtils.scale(abs(MathUtils.translate(-img_y, y)), scaling_factor_y)
                joints.append((j_x, j_y))
            # j_x, j_y = zip(*tables[k])
            # j_x = [MathUtils.scale(j, scaling_factor_x) for j in j_x]
            # j_y = [MathUtils.scale(abs(MathUtils.translate(-img_y, j)), scaling_factor_y) for j in j_y]
            # joints = zip(j_x, j_y)
            tables_new[(x1, y1, x2, y2)] = joints

        v_segments_new = []
        for v in v_segments:
            x1, x2 = MathUtils.scale(v[0], scaling_factor_x), MathUtils.scale(v[2], scaling_factor_x)
            y1, y2 = (
                MathUtils.scale(abs(MathUtils.translate(-img_y, v[1])), scaling_factor_y),
                MathUtils.scale(abs(MathUtils.translate(-img_y, v[3])), scaling_factor_y),
            )
            v_segments_new.append((x1, y1, x2, y2))

        h_segments_new = []
        for h in h_segments:
            x1, x2 = MathUtils.scale(h[0], scaling_factor_x), MathUtils.scale(h[2], scaling_factor_x)
            y1, y2 = (
                MathUtils.scale(abs(MathUtils.translate(-img_y, h[1])), scaling_factor_y),
                MathUtils.scale(abs(MathUtils.translate(-img_y, h[3])), scaling_factor_y),
            )
            h_segments_new.append((x1, y1, x2, y2))

        return tables_new, v_segments_new, h_segments_new

    @staticmethod
    def scale_image_bbox(bbox: List, factors):
        """Translates and scales image coordinate space to pdf
        coordinate space.

        Parameters
        ----------
        bbox :
            Dict with table boundaries as keys and list of intersections
            in that boundary as value.
        factors : tuple
            Tuple (scaling_factor_x, scaling_factor_y, img_y) where the
            first two elements are scaling factors and img_y is height of
            image.

        """
        scaling_factor_x, scaling_factor_y, img_y = factors

        x1, y1, x2, y2 = bbox
        x1 = MathUtils.scale(x1, scaling_factor_x)
        y1 = MathUtils.scale(abs(MathUtils.translate(-img_y, y1)), scaling_factor_y)
        x2 = MathUtils.scale(x2, scaling_factor_x)
        y2 = MathUtils.scale(abs(MathUtils.translate(-img_y, y2)), scaling_factor_y)

        new_bbox = [x1, y1, x2, y2]
        return new_bbox

    @staticmethod
    def bbox_intersection_area(ba, bb) -> float:
        """Returns area of the intersection of the bounding boxes of two PDFMiner objects.

        Parameters
        ----------
        ba : PDFMiner text object
        bb : PDFMiner text object

        Returns
        -------
        intersection_area : float
            Area of the intersection of the bounding boxes of both objects

        """
        x_left = max(ba.x0, bb.x0)
        y_top = min(ba.y1, bb.y1)
        x_right = min(ba.x1, bb.x1)
        y_bottom = max(ba.y0, bb.y0)

        if x_right < x_left or y_bottom > y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_top - y_bottom)
        return intersection_area

    @staticmethod
    def bbox_area(bb) -> float:
        """Returns area of the bounding box of a PDFMiner object.

        Parameters
        ----------
        bb : PDFMiner text object

        Returns
        -------
        area : float
            Area of the bounding box of the object

        """
        return (bb.x1 - bb.x0) * (bb.y1 - bb.y0)

    @staticmethod
    def bbox_intersect(ba, bb) -> bool:
        """Returns True if the bounding boxes of two PDFMiner objects intersect.

        Parameters
        ----------
        ba : PDFMiner text object
        bb : PDFMiner text object

        Returns
        -------
        overlaps : bool
            True if the bounding boxes intersect

        """
        return ba.x1 >= bb.x0 and bb.x1 >= ba.x0 and ba.y1 >= bb.y0 and bb.y1 >= ba.y0

    @staticmethod
    def bbox_longer(ba, bb) -> bool:
        """Returns True if the bounding box of the first PDFMiner object is longer or equal to the second.

        Parameters
        ----------
        ba : PDFMiner text object
        bb : PDFMiner text object

        Returns
        -------
        longer : bool
            True if the bounding box of the first object is longer or equal

        """
        return (ba.x1 - ba.x0) >= (bb.x1 - bb.x0)


if __name__ == '__main__':
    pass
    print(MathUtils.get_edit_distance(src="hellp", target="hello"))
    print(MathUtils.get_edit_distance(src="hllp", target="hello"))
    print(MathUtils.get_edit_distance(src="你好", target="您好"))
    print(MathUtils.get_edit_distance(src="你好吗", target="您好"))
    print(MathUtils.get_edit_distance(src=["你好吗", "测试"], target=["您好", "测试"]))
    print(MathUtils.get_edit_distance(src=["你好吗", "测试"], target=["您好", "考试"]))

    pos_list = [0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 15, 18, 20]
    print(MathUtils.get_sub_array(pos_list))
    print(MathUtils.get_sub_array2(pos_list))
