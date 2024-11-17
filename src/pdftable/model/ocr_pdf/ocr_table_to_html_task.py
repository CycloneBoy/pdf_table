#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_table_to_html_task
# @Author  ：cycloneboy
# @Date    ：20xx/7/25 18:11
import time
import traceback
from typing import List

import numpy as np

from pdftable.entity.ie_entity import PdfTextCellV2
from pdftable.entity.table_entity import OcrCell
from pdftable.model import Cell, box_in_other_box, distance, compute_iou_v2, TableProcessUtils
from pdftable.model.ocr_pdf.ocr_output import OcrSystemModelOutput
from pdftable.model.ocr_pdf.table import TableMatch
from pdftable.model.ocr_pdf.table.table_master_match import TableMasterMatcher
from pdftable.utils import FileUtils, logger, PdfUtils
from pdftable.utils.table.image_processing import PdfImageProcessor

"""
OCR table 结果转html 
"""

__all__ = [
    'OcrTableToHtmlTask'
]


class OcrTableToHtmlTask(object):

    def __init__(self, config=None, debug=True,
                 output_dir=None, **kwargs):
        super().__init__()
        self.config = config
        self.debug = debug
        self.output_dir = output_dir
        self.src_id = kwargs.get('src_id', None)

        self.diff = 2
        self.match = TableMatch(filter_ocr_result=True)
        self.match_master = TableMasterMatcher()

    def convert_to_html(self, ocr_system_output: OcrSystemModelOutput):
        pass

    def find_top1_mach_box(self, text_box, table_bboxs: List[Cell]):
        """
        查找对应的 top1 距离最近的box

        :param text_box:
        :param table_bboxs:
        :return:
        """
        distances = []
        find_cell_in_index = -1
        for index, table_cell in enumerate(table_bboxs):
            pred_box = table_cell.to_bbox()
            # 文本框在table 的cell 内
            if box_in_other_box(pred_box, text_box):
                dis = (0, 0)
                find_cell_in_index = index
                break
            else:
                # compute iou and l1 distance
                dis = (distance(text_box, pred_box), 1. - compute_iou_v2(text_box, pred_box))
            distances.append(dis)

        if find_cell_in_index > -1:
            return find_cell_in_index

        sorted_distances = distances.copy()
        # select det box by iou and l1 distance
        sorted_distances = sorted(sorted_distances, key=lambda item: (item[1], item[0]))
        top1_index = distances.index(sorted_distances[0])
        return top1_index

    def convert_table_to_html(self, ocr_system_output: OcrSystemModelOutput):

        run_time = FileUtils.get_file_name(f"{self.output_dir}.txt")

        metric = {
            "run_time": run_time,
            "table_metric": []
        }

        ocr_results = ocr_system_output.ocr_result
        layout_result = ocr_system_output.layout_result

        ocr_post_process = False if ocr_system_output.is_pdf else True
        use_master = True if ocr_system_output.use_master else False

        all_table_cell = []
        all_table_html = []
        all_db_table_html = []
        all_table_width = [abs(table["bbox"][0] - table["bbox"][2])
                           for table in ocr_system_output.table_cell_result]
        max_table_width = max(all_table_width) if len(all_table_width) > 0 else 100
        for table_idx, table in enumerate(ocr_system_output.table_cell_result):
            try:
                one_metric = {}
                table_bbox = table["bbox"]
                raw_table_cells = table["table_cells"]
                is_image = table["is_image"]
                if is_image:
                    logger.info(f"当前表格是图片误识别【PDF图片】：{table_idx} - {table_bbox}")
                    continue

                match_table_flag, match_table = TableProcessUtils.filter_layout_figure(layout_result=layout_result,
                                                                                       table_bbox=table_bbox,
                                                                                       label="table", )
                # 没有匹配到表格，再进行匹配图片
                match_flag = False
                match_figure = None

                x1, y1, x2, y2 = table_bbox
                table_width = abs(x2 - x1)
                radio = table_width / abs(y1 - y2)
                if not match_table_flag and (table_width / max_table_width < 0.7 or radio < 3):
                    match_flag, match_figure = TableProcessUtils.filter_layout_figure(layout_result=layout_result,
                                                                                      table_bbox=table_bbox,
                                                                                      label="figure", )

                table["is_layout_figure"] = match_flag
                table["match_figure"] = match_figure
                if match_flag:
                    logger.info(f"当前表格是图片误识别【Layout图片】：{table_idx} - {table_bbox} ->layout: {match_figure}")
                    continue

                table_cells = []
                for key, cells in raw_table_cells.items():
                    table_cells.extend(cells)

                # 表格中的text cell
                text_bboxs, remain_cells = TableProcessUtils.get_text_in_table_bbox(bbox=table_bbox,
                                                                                    ocr_results=ocr_results,
                                                                                    diff=self.diff)
                if "not_spit_text" in table:
                    structure_res = table["html"], table["table_bbox"]
                    table_html, match_metric = self.match_table_structure_and_text_cell(
                        table_idx=table_idx,
                        structure_res=structure_res,
                        text_bboxs=text_bboxs,
                        raw_filename=ocr_system_output.raw_filename,
                        ocr_post_process=ocr_post_process,
                        use_master=use_master)
                    db_table_html = table_html
                    table_row_dict_sorted = None
                else:
                    table_cells, table_html, match_metric, db_table_html = self.match_table_cell_and_text_cell(
                        table_idx=table_idx,
                        table_cells=table_cells,
                        text_bboxs=text_bboxs,
                        raw_filename=ocr_system_output.raw_filename,
                        ocr_post_process=ocr_post_process)

                    table_row_dict_sorted = TableProcessUtils.convert_table_cell_to_dict(table_cells)

                all_table_cell.append(table_cells)
                all_table_html.append(table_html)
                all_db_table_html.append(db_table_html)

                table["table_html"] = table_html
                table["db_table_html"] = db_table_html
                table["table_cells"] = table_row_dict_sorted
                table["text_bboxs"] = text_bboxs

                one_metric.update(match_metric)
                metric["table_metric"].append(one_metric)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"提取表格异常：{self.src_id} - {table_idx}")

        return metric

    def match_table_cell_and_text_cell(self, table_idx, table_cells: List[Cell],
                                       text_bboxs: List[OcrCell],
                                       raw_filename=None,
                                       ocr_post_process=False):
        """
        匹配 table cell 和 text cell

        :param table_idx:
        :param table_cells:
        :param text_bboxs:
        :param raw_filename:
        :param ocr_post_process:
        :return:
        """
        # text cell match to table cell
        matched = {}
        for index, text_ocr_box in enumerate(text_bboxs):
            text_box = text_ocr_box.to_bbox()
            top1_index = self.find_top1_mach_box(text_box=text_box, table_bboxs=table_cells)
            top1_bbox = table_cells[top1_index]
            # msg = f"{}"
            if top1_index not in matched.keys():
                matched[top1_index] = [index]
            else:
                matched[top1_index].append(index)
        logger.info(f"cell and text matched: {matched}")

        results = []
        for key, val in matched.items():
            table_cell = table_cells[key]
            match_text_cells = [text_bboxs[index] for index in val]
            text_show, match_text_cells = self.get_one_cell_text(match_text_cells)
            table_cell.text_item = match_text_cells

            text = ''.join(text_show)
            if ocr_post_process:
                text = TableProcessUtils.ocr_post_process(text)

            table_cell.text = text
            results.append(table_cell)
        not_matched_cells = [item for index, item in enumerate(table_cells) if index not in matched.keys()]
        results.extend(not_matched_cells)

        logger.info(f"cell and text matched matched end: match total: {len(matched)} "
                    f"-not match total {len(not_matched_cells)}")
        logger.info(f"match text cell results: {len(results)} -> ")

        results.sort(key=lambda x: (x.row_index, x.col_index))

        table_html, db_table_html = TableProcessUtils.cell_to_html(table_cells=results)
        table_html_str = "\n".join(table_html) + "\n"
        save_html_file = f"{self.output_dir}/{raw_filename}_table_{table_idx + 1}.html"
        FileUtils.save_to_text(save_html_file, table_html_str)

        save_db_html_file = save_html_file.replace(".html", "_db.html")
        FileUtils.save_to_text(save_db_html_file, "\n".join(db_table_html) + "\n")
        logger.info(f"保存table cell html：{save_html_file}")

        metric = {
            "table_idx": table_idx,
            # "cell_text_image_file": cell_text_image_file,
            "save_html_file": save_html_file,
            "save_db_html_file": save_db_html_file
        }

        return results, table_html, metric, db_table_html

    def match_table_structure_and_text_cell(self, table_idx, structure_res,
                                            text_bboxs: List[OcrCell],
                                            raw_filename=None,
                                            ocr_post_process=False,
                                            use_master=None):
        """
        匹配 table cell 和 text cell

        :param table_idx:
        :param structure_res:
        :param text_bboxs:
        :param raw_filename:
        :param ocr_post_process:
        :return:
        """

        dt_boxes = []
        rec_res = []
        for cell in text_bboxs:
            dt_boxes.append(cell.to_bbox())
            rec_res.append((cell.text, 1))

        dt_boxes = np.array(dt_boxes)

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res, use_master=use_master)

        pred_html = [pred_html.replace("<html><body>", "")
                     .replace("</body></html>", "")
                     .replace("<tbody>", "")
                     .replace("</tbody>", "")
                     .replace("</tr>", "</tr>\n")
                     .replace("<table>", "<table class='pdf-table' border='1'>")]

        toc = time.time()

        save_html_file = f"{self.output_dir}/{raw_filename}_table_{table_idx + 1}.html"
        FileUtils.save_to_text(save_html_file, "".join(pred_html))

        save_db_html_file = save_html_file.replace(".html", "_db.html")
        FileUtils.save_to_text(save_db_html_file, "".join(pred_html))
        logger.info(f"保存table cell html：{save_html_file}")

        metric = {
            "table_idx": table_idx,
            # "cell_text_image_file": cell_text_image_file,
            "save_html_file": save_html_file,
            "save_db_html_file": save_db_html_file
        }

        return pred_html, metric

    def get_one_cell_text(self, match_text_cells: List[OcrCell]):
        """
            获取一个table cell 中的文字
                - 按照阅读顺序排序，合并一行

        :param match_text_cells:
        :return:
        """

        match_text_cells_height = [item.height for item in match_text_cells]
        mean_height = sum(match_text_cells_height) / len(match_text_cells_height) * 1.0
        line_tol = mean_height / 3

        match_text_cells_y1 = [round(item.left_top.y) for item in match_text_cells]

        norm_y1_list = PdfUtils.merge_close_lines(sorted(match_text_cells_y1, reverse=True),
                                                  line_tol=line_tol)

        pdf_text_cells = []
        for item in match_text_cells:
            new_y1 = PdfImageProcessor.find_close_norm_x(item.left_top.y, norm_list=norm_y1_list, atol=line_tol)
            one_cell = PdfTextCellV2(text_cell=item, y_index=new_y1)
            pdf_text_cells.append(one_cell)

        # 文字排序
        pdf_text_cells.sort(key=lambda x: (x.y_index, x.x1))

        text_show = []
        for item in pdf_text_cells:
            raw_text = item.get_text().strip("\n")
            text_show.append(raw_text)

        sort_match_text_cells = [item.text_cell for item in pdf_text_cells]
        return text_show, sort_match_text_cells

    def __call__(self, ocr_system_output: OcrSystemModelOutput):
        start = time.time()
        logger.info(f"开始表格结构和OCR识别合并提取TABLE。")
        metric = self.convert_table_to_html(ocr_system_output=ocr_system_output)

        use_time = time.time() - start
        logger.info(f"结束表格结构和OCR识别合并提取TABLE, 耗时：{use_time} s, "
                    f"提取表格数量：{len(ocr_system_output.table_cell_result)} 个。")
        return ocr_system_output
