#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_table_task
# @Author  ：cycloneboy
# @Date    ：20xx/9/5 9:53
import os
import time

from typing import List, Union, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pdftable.model.ocr_pdf import OCRDocumentConfig, OcrTableStructureTask
from pdftable.model.ocr_pdf.ocr_output import OcrSystemModelOutput
from pdftable.model.ocr_pdf.ocr_text_task import OcrTextTask
from pdftable.model.ocr_pdf.table import TableMatch
from pdftable.model.ocr_pdf.table.table_metric import TEDS
from pdftable.utils import FileUtils, logger, TimeUtils

"""
OCR TABLE SLANET 
"""

def to_excel(html_table, excel_path):
    from tablepyxl import tablepyxl

    tablepyxl.document_to_xl(html_table, excel_path)



class OcrTableTask(object):

    def __init__(self, config: OCRDocumentConfig,
                 output_dir=None, predictor_type="onnx",
                 delete_check_success=True, **kwargs):
        super().__init__()
        self.config = config
        self.debug = config.debug
        self.output_dir = output_dir
        self.predictor_type = predictor_type
        self.delete_check_success = delete_check_success

        # 延迟加载模型
        self.text_ocr = None
        self.text_recognizer = None
        self.table_structure_recognizer = None

        self.match = TableMatch(filter_ocr_result=True)

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        if self.text_ocr is not None:
            self.text_ocr.set_output_dir(output_dir)

        if self.text_recognizer is not None:
            self.text_recognizer.set_output_dir(output_dir)

        FileUtils.check_file_exists(f"{output_dir}/demo.txt")

    def init_ocr_model(self):
        start = time.time()
        logger.info(f"开始加载OCR模型")
        init_model = []
        if self.text_ocr is None:
            self.text_ocr = OcrTextTask(config=self.config,
                                        debug=self.debug,
                                        output_dir=self.output_dir, )
            init_model.append(self.text_ocr.task)

        if self.table_structure_recognizer is None:
            self.table_structure_recognizer = OcrTableStructureTask(model=self.config.table_structure_model,
                                                                    predictor_type=self.predictor_type,
                                                                    output_dir=self.output_dir,
                                                                    lang=self.config.lang,
                                                                    task_type=self.config.table_structure_task_type)
            init_model.append(self.table_structure_recognizer.task)

        use_time = time.time() - start
        logger.info(f"加载OCR模型完毕： {len(init_model)} - {init_model}, 耗时：{use_time:3f} s.")

    def ocr_extract_text(self, inputs, **kwargs) -> Tuple[OcrSystemModelOutput, Dict]:
        start = time.time()

        if self.text_ocr is None:
            self.init_ocr_model()

        ocr_system_output: OcrSystemModelOutput = self.text_ocr(inputs, **kwargs)

        use_time = time.time() - start
        logger.info(f"OCR提取文字耗时：{use_time:3f} s. det_result: {len(ocr_system_output.ocr_result)}")

        metric = {
            "use_time": use_time
        }
        return ocr_system_output, metric

    def table_structure_detection(self, image):
        start = time.time()

        if self.table_structure_recognizer is None:
            self.init_ocr_model()

        result = self.table_structure_recognizer(image)
        table_structure_result = result[0]
        use_time = time.time() - start
        logger.info(f"表格结构检测耗时：{use_time:3f} s. table_structure_result: {len(table_structure_result)}")

        metric = {
            "use_time": use_time
        }
        return result, metric

    def __call__(self, inputs, save_result=True, src_id=None, page=None, **kwargs):
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()
        run_time = TimeUtils.now_str_short()

        ocr_system_output, ocr_metric = self.ocr_extract_text(inputs,
                                                              save_result=True, src_id=None, page=None,
                                                              **kwargs)

        if not ocr_system_output.is_pdf:
            table_structure_result, table_structure_metric = self.table_structure_detection(inputs, )
            ocr_system_output.table_structure_result = table_structure_result

        pred_res, time_dict = self.extract_table(ocr_system_output)

        if self.output_dir is not None and self.debug:
            self.show_results(pred_res, inputs)

        pdf_html = [pred_res]
        ocr_system_output.pdf_html = pdf_html

        return ocr_system_output

    def extract_table(self, ocr_system_output: OcrSystemModelOutput,
                      return_ocr_result_in_table=True):
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()

        table_structure_result = ocr_system_output.table_structure_result[0]

        result['cell_bbox'] = table_structure_result["polygons"].tolist()

        structure_res = table_structure_result["structure_str_list"], table_structure_result["polygons"]
        det_result = ocr_system_output.det_result
        ocr_result = ocr_system_output.ocr_result

        result['structure_str_list'] = structure_res[0]

        dt_boxes = []
        rec_res = []
        for cell in ocr_result:
            dt_boxes.append(cell.to_bbox())
            rec_res.append((cell.text, 1))

        dt_boxes = np.array(dt_boxes)

        if return_ocr_result_in_table:
            result['boxes'] = dt_boxes  # [x.tolist() for x in dt_boxes]
            result['rec_res'] = rec_res

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()
        time_dict['match'] = toc - tic
        result['html'] = pred_html
        end = time.time()
        time_dict['all'] = end - start

        return result, time_dict

    def show_results(self, predict, image_file):
        pred_html = predict['html']
        pred_cell_bbox = predict['cell_bbox']

        base_name = os.path.basename(image_file)
        excel_path = os.path.join(self.output_dir, base_name.split('.')[0] + '.xlsx')
        to_excel(pred_html, excel_path)

        name_end = ".jpg"
        if name_end not in base_name:
            name_end = ".png"
        text_cell_file = base_name.replace(name_end, "_text_cell.jpg")
        table_res_file = f"table_res_{base_name}"

        res_html_file = os.path.join(self.output_dir, f'table_{FileUtils.get_file_name(image_file)}.html')

        f_html = open(res_html_file, mode='w', encoding='utf-8')
        f_html.write('<html>\n<body>\n')
        f_html.write('<table border="1">\n')
        f_html.write(
            "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />"
        )
        f_html.write("<tr>\n")
        f_html.write('<td>img name\n')
        f_html.write('<td>ori image</td>')
        f_html.write('<td>table html</td>')
        f_html.write('<td>cell box</td>')
        f_html.write("</tr>\n")

        f_html.write("<tr>\n")
        f_html.write(f'<td> {base_name} <br/>\n')
        f_html.write(f'<td><img src="{text_cell_file}" width=640></td>\n')
        f_html.write('<td><table  border="1">' + pred_html.replace(
            '<html><body><table>', '').replace('</table></body></html>', '') +
                     '</table></td>\n')
        f_html.write(
            f'<td><img src="{table_res_file}" width=640></td>\n')
        f_html.write("</tr>\n")

        f_html.write("</table>\n")
        f_html.write('</body>\n</html>\n')
        f_html.close()

    def eval_table(self, file_dir,
                   label_file,
                   output_dir,
                   run_name="_2023_09_05",
                   ):
        begin_time = time.time()
        begin_time_str = TimeUtils.now_str()
        run_time = TimeUtils.now_str_short()

        # file_name_list = FileUtils.list_file_prefix(file_dir, sort=True, end_with=".png", )

        gt_html_dict = FileUtils.load_table_label_txt(label_file)

        ocr_result_file = f"{output_dir}/ocr{run_name}.pickle"
        structure_result_file = f"{output_dir}/structure{run_name}.pickle"
        table_html_result_file = f"{output_dir}/table{run_name}.pickle"

        ocr_result = FileUtils.load_to_model(ocr_result_file)
        structure_result = FileUtils.load_to_model(structure_result_file)
        table_html_result = FileUtils.load_to_model(table_html_result_file)

        logger.info(f"开始进行评估：{len(gt_html_dict)}")
        logger.info(f"已经评估数量：{len(ocr_result)}")

        pred_htmls = []
        gt_htmls = []
        eval_total = 0
        for img_name, gt_html in tqdm(gt_html_dict.items()):
            if img_name in ocr_result and img_name in table_html_result:
                pred_html = table_html_result[img_name]
                pred_htmls.append(pred_html)
                gt_htmls.append(gt_html)
                continue

            image_file = f"{file_dir}/{img_name}"
            if not FileUtils.check_file_exists(image_file):
                continue

            ocr_system_output = self.__call__(image_file)
            table_result = ocr_system_output.pdf_html[0]

            dt_boxes = table_result['boxes']
            rec_res = table_result['rec_res']
            structure_res = table_result['structure_str_list']
            pred_html = table_result["html"]

            ocr_result[img_name] = [dt_boxes, rec_res]
            structure_result[img_name] = structure_res
            table_html_result[img_name] = pred_html

            FileUtils.save_to_pickle(ocr_result, ocr_result_file)
            FileUtils.save_to_pickle(structure_result, structure_result_file)
            FileUtils.save_to_pickle(table_html_result, table_html_result_file)

            eval_total += 1

            pred_htmls.append(pred_html)
            gt_htmls.append(gt_html)

        logger.info(f"开始进行计算TEDS: {len(pred_htmls)}")
        # compute teds
        teds = TEDS(n_jobs=16)
        scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
        teds_metric = sum(scores) / len(scores)

        use_time = time.time() - begin_time

        end_time_str = TimeUtils.now_str()
        use_time2 = TimeUtils.calc_diff_time(begin_time_str, end_time_str)
        metric = {
            "begin_time": begin_time_str,
            "end_time": end_time_str,
            "use_time": use_time,
            "use_time2": use_time2,
            "total_label": len(gt_html_dict),
            "eval_total": eval_total,
            "teds_metric": teds_metric,
        }

        metric_file = f"{output_dir}/metric{run_name}.json"
        metric_append_file = metric_file.replace('.json', ".txt")
        FileUtils.dump_json(metric_file, metric)

        FileUtils.save_to_text(metric_append_file, content=f"{str(metric)}\n", mode="a")

        logger.info(f'teds: {teds_metric}')
        logger.info(f"评估文件：{metric_append_file}")
