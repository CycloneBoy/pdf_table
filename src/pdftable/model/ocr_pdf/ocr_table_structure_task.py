#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_table_structure_task
# @Author  ：cycloneboy
# @Date    ：20xx/7/17 18:11
import json
import os
import time
from copy import deepcopy
from typing import Dict, List

import cv2
import numpy as np
import torch
from pdfminer.layout import LTRect

from .base_infer_task import BaseInferTask
from .. import LoreModel, TableLorePreProcessor, TableLorePostProcessor, TableProcessUtils, Cell
from ..center_net import OCRTableCenterNetPreProcessor, OCRTableCenterNetPostProcessor, TableStructureRec, \
    TableCenterNetConfig
from ..lgpma.configuration_lgpma import LgpmaConfig
from ..lgpma.processor_lgpma import LgpmaPreProcessor, LgpmaPostProcessor
from ..line_cell import LineCellConfig, LineCellPreProcessor, LineCellPostProcessor
from ..line_cell_pdf import LineCellPdfConfig, LineCellPdfPreProcessor, LineCellPdfPostProcessor
from ..lore.configuration_lore import LoreConfig
from ..mtl_tabnet import MtlTabnetConfig, MtlTabNetPreProcessor, MtlTabNetPostProcessor
from ..pdf_table.table_cell_extract import TableCellExtract
from ..slanet import SLANetConfig, SLANetPreprocessor, SLANetPostProcessor
from ..table.lgpma.base_utils import build
from ..table.lgpma.checkpoint import load_checkpoint
from ..table.lgpma.model_lgpma import LGPMA
from ..table.line_cell.table_cell_extract_from_pdf import TableCellExtractFromPdf
from ..table.mtl_tabnet.table_master import MtlTabNet, TableMaster
from ...utils import logger, FileUtils, PdfUtils, CommonUtils
from ...utils.ocr import OcrCommonUtils, OcrInferUtils
from ...utils.table.image_processing import PdfImageProcessor

"""
ocr_table_structure_task 
"""

__all__ = [
    'OcrTableStructureTask'
]


class OcrTableStructureTask(BaseInferTask):

    def __init__(self, task="ocr_table_structure", model="CenterNet", **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        assert model in ["CenterNet", "SLANet", "Lore", "Lgpma", "MtlTabNet",
                         "TableMaster", "LineCell", "LineCellPdf"]

        self.table_structure_merge = kwargs.get("table_structure_merge", False)

        if model == "CenterNet":
            self._config = TableCenterNetConfig(debug=self.debug)
        elif model == "SLANet":
            lang = self.lang
            if lang not in ["ch","en"]:
                lang = "en"
            self._config = SLANetConfig(lang=lang, debug=self.debug)
        elif model == "Lore":
            if self.task_type == 'PubTabNet':
                self.task_type = 'ptn'
            self._config = LoreConfig(task_type=self.task_type, debug=self.debug)
        elif model == "Lgpma":
            self._config = LgpmaConfig(debug=self.debug)
        elif model in ["MtlTabNet", "TableMaster"]:
            self._config = MtlTabnetConfig(model_name=model, task_type=self.task_type, debug=self.debug)
        elif model == "LineCell":
            self._config = LineCellConfig(output_dir=self.output_dir, debug=self.debug)
        elif model == "LineCellPdf":
            self._config = LineCellPdfConfig(output_dir=self.output_dir, debug=self.debug)

        self.model_provider = self._config.model_provider
        self._predictor_type = self._config.predictor_type

        model_name_or_path = self.get_model_name_or_path()
        self._config.model_path = model_name_or_path

        self._get_inference_model()

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        FileUtils.check_file_exists(f"{output_dir}/demo.txt")

    def _construct_model(self, model):
        if model == "CenterNet":
            run_model = TableStructureRec(self._config)
        elif model == "Lore":
            run_model = LoreModel(self._config)

            # model_path_v2 = f"./best_model/pytorch_model.bin"
            # if FileUtils.check_file_exists(model_path_v2):
            #     checkpoint = torch.load(model_path_v2, map_location='cpu')
            #     new_state = OrderedDict()
            #     for k, v in checkpoint.items():
            #         new_k = k.replace("_orig_mod.", "")
            #         new_state[new_k] = v
            #     run_model.load_state_dict(new_state, strict=True)
            #     logger.info(f"加载LORE训练好的模型：{model_path_v2}")

        elif model in ["Lgpma", "MtlTabNet", "TableMaster"]:
            DETECTORS = {
                "LGPMA": LGPMA,
                "MtlTabNet": MtlTabNet,
                "TableMaster": TableMaster
            }
            run_model = build(self._config.get_model_config(), DETECTORS, self._config.get_test_config())
            run_model.cfg = self._config.config
            load_checkpoint(run_model, filename=self._config.model_path)

        elif model == "LineCell":
            run_model = TableCellExtract(**self._config.__dict__)
        elif model == "LineCellPdf":
            run_model = TableCellExtractFromPdf(**self._config.__dict__)
        else:
            run_model = None

        self._model = run_model
        if run_model is not None and hasattr(run_model, "parameters"):
            # CommonUtils.calc_model_parameter_number(run_model, model_name=model)
            if "." in self._config.model_path:
                model_dir = FileUtils.get_dir_file_name(self._config.model_path)
            else:
                model_dir = self._config.model_path
            save_best_model_path = f"{model_dir}/pytorch_model.bin"
            if not FileUtils.check_file_exists(save_best_model_path):
                logger.info(f"model path: {save_best_model_path}")
                torch.save(run_model.state_dict(), save_best_model_path)

    def _build_processor(self):
        if self.model == "CenterNet":
            self._pre_processor = OCRTableCenterNetPreProcessor(self._config)
            self._post_processor = OCRTableCenterNetPostProcessor(self._config)
        elif self.model == "SLANet":
            self._pre_processor = SLANetPreprocessor(self._config)
            self._post_processor = SLANetPostProcessor(self._config)
        elif self.model == "Lore":
            self._pre_processor = TableLorePreProcessor(self._config)
            self._post_processor = TableLorePostProcessor(self._config, output_dir=self.output_dir)
        elif self.model == "Lgpma":
            self._pre_processor = LgpmaPreProcessor(self._config)
            self._post_processor = LgpmaPostProcessor(self._config, output_dir=self.output_dir)
        elif self.model in ["MtlTabNet", "TableMaster"]:
            self._pre_processor = MtlTabNetPreProcessor(self._config)
            self._post_processor = MtlTabNetPostProcessor(self._config, output_dir=self.output_dir)
        elif self.model == "LineCell":
            self._pre_processor = LineCellPreProcessor(self._config)
            self._post_processor = LineCellPostProcessor(self._config, output_dir=self.output_dir)
        elif self.model == "LineCellPdf":
            self._pre_processor = LineCellPdfPreProcessor(self._config)
            self._post_processor = LineCellPdfPostProcessor(self._config, output_dir=self.output_dir)

    def _preprocess(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = []
        for item in inputs:
            one_item = self._pre_processor(item) if self._pre_processor else item
            if isinstance(one_item, list):
                one_item = one_item[0]
            batch.append(one_item)

        results = {"inputs": batch}
        return results

    def build_batch_lore(self, batch: Dict):
        one_batch = {}
        for k, v in batch.items():
            if k in ["image"]:
                v = v.to(self.device)
                if self.eval_fp16:
                    v = v.half()
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                new_v = []
                for item in v:
                    item = item.to(self.device)
                    if self.eval_fp16:
                        item = item.half()
                    new_v.append(item)
                v = new_v
            one_batch[k] = v
        return one_batch

    def _run_model(self, inputs, **kwargs):
        infer_data_loader = inputs["inputs"]

        results = []
        begin = time.time()
        for index, batch in enumerate(infer_data_loader):
            run_batch = self.build_run_batch(batch)

            pred_res, elapse = self.infer(run_batch)

            pred_res = self.extract_predict_result(pred_res)

            # logger.info(f'pred_res: {pred_res.shape}')

            infer_result = {
                "results": pred_res,
                "elapse": elapse,
            }
            raw_batch = deepcopy(batch)
            for image_key in ["image", "img", "pixel_values"]:
                if image_key in raw_batch:
                    raw_batch.pop(image_key)
            infer_result.update(raw_batch)

            results.append(infer_result)

        use_time = time.time() - begin
        inputs["results"] = results
        inputs["use_time"] = use_time
        return inputs

    def build_run_batch(self, batch: Dict):
        if self.model in ["Lore", "Lgpma", "MtlTabNet", "TableMaster"]:
            one_batch = self.build_batch_lore(batch)
            one_batch.pop("inputs")
            run_batch = one_batch
        elif self.model in ["LineCell", "LineCellPdf"]:
            run_batch = batch
        else:
            image = batch.get("image", None)
            run_batch = {}
            if self._predictor_type == "onnx":
                input_name = self.get_input_name()
            else:
                input_name = "image"
            run_batch[input_name] = image
        return run_batch

    def extract_predict_result(self, pred_res):
        if self._predictor_type == "onnx":
            if self.model == "CenterNet":
                pred_res = self.get_onnx_output_dict(pred_res)
                pred_res = self.build_torch_infer_batch(pred_res)
            elif self.model == "SLANet":
                pred_res = {
                    "loc_preds": pred_res[0],
                    "structure_probs": pred_res[1],
                }
        return pred_res

    def _postprocess(self, inputs, **kwargs):
        results = []

        use_time = inputs["use_time"]

        predict_result = inputs["results"]
        for index, batch in enumerate(predict_result):
            predict = self._post_processor(batch) if self._post_processor else batch["results"]

            # table_results = predict['polygons']
            if self.output_dir is not None and "inputs" in predict:
                try:
                    # image_height = int(batch['meta'][0][2]) if "meta" in batch else None
                    table_cells, table_cell_metric = self.show_results(predict)
                    predict['table_cells'] = table_cells
                    predict['table_cell_metric'] = table_cell_metric
                except Exception as e:
                    pass
            results.append(predict)

        # logger.info(f"infer finished：{len(predict_result)} - use time：{use_time:.3f} s / {use_time / 60:.3f} min.")
        return results

    def is_line_cell_model(self):
        return self.model in ["LineCell", "LineCellPdf"]

    def show_results(self, predict, ):
        """
        显示结果

        :param predict:
        :return:
        """
        bbox_list = predict['polygons']
        structure_str_list = predict.get('structure_str_list', "")
        logi_list = predict.get('logi', "")
        html_context = predict.get('html_context', "")

        image_file = FileUtils.get_pdf_to_image_file_name(predict['inputs'])
        file_name = FileUtils.get_file_name(image_file)

        save_html_file = f"{self.output_dir}/{file_name}"
        table_cells_v2 = None
        table_cell_metric = None
        if len(bbox_list) > 0 and len(logi_list) > 0 and not self.is_line_cell_model():
            table_cells = TableProcessUtils.get_table_cell_from_table_logit(table_bboxs=bbox_list,
                                                                            logits=logi_list,
                                                                            save_html_file=save_html_file)
            if len(structure_str_list) == 0:
                table_html, db_table_html = TableProcessUtils.cell_to_html(table_cells=table_cells,
                                                                           first_header=False,
                                                                           add_width=False,
                                                                           add_text=False)
                table_html = "".join(table_html)
                table_html = f"<html><body>{table_html}</body></html>" \
                    .replace("<td >", "<td>") \
                    .replace("<tbody>", "") \
                    .replace("</tbody>", "")
                structure_str_list = [table_html]

            if self.model in ["Lore"] and self.task_type in ["wtw"]:
                post_table_cells = self.post_process_bbox_and_logits(table_bboxs=bbox_list,
                                                                     logits=logi_list,
                                                                     image_file=image_file)
                TableProcessUtils.table_cell_to_html_show(table_cells=post_table_cells,
                                                          save_html_file=save_html_file,
                                                          add_end="_table_structure_post.html")

                if not self.table_structure_merge:
                    logger.info(f"没有采用TSR merge模型，生成table_cell")
                    parser = TableCellExtractFromPdf(output_dir=self.output_dir,
                                                     line_tol=10,
                                                     debug=self.debug)
                    table_cells_v2, table_cell_metric = parser.extract_cell(file_name=image_file,
                                                                            table_bboxs=bbox_list)
        # LineCell
        if "table_cells" in predict:
            table_cells_v2 = predict["table_cells"]
            table_cell_metric = predict["table_cell_metric"]

        if self.model in ["SLANet"]:
            structure_str_list = ["".join(structure_str_list)]

        if len(structure_str_list) > 0:
            all_tsr_result = []
            if not isinstance(structure_str_list, list):
                structure_str_list = [structure_str_list]
            for structure_list in structure_str_list:
                tsr_result = "".join(structure_list) \
                    .replace("></td>", ">test_text</td>") \
                    .replace("</tr>", "</tr>\n") \
                    .replace("<table>", '<table border="1">')

                all_tsr_result.append(tsr_result)

            FileUtils.save_to_text(f"{save_html_file}_table_structure.html", "\n".join(all_tsr_result))

        bbox_list_str = CommonUtils.box_list_to_json_str(bbox_list)
        bbox_list_sep = "" if "polygons_sep" not in predict else CommonUtils.box_list_to_json_str(
            predict["polygons_sep"])
        logi_list_sep = "" if "logi_sep" not in predict else CommonUtils.box_list_to_json_str(predict["logi_sep"])

        result_file_name = os.path.join(self.output_dir, f'table_infer_{file_name}.json')
        result = {
            "file_name": predict['inputs'],
            "structure_str_list": structure_str_list,
            "bbox_list": bbox_list_str,
            "logi_list": json.dumps(logi_list.tolist()) if len(logi_list) > 0 else "",
            "bbox_list_sep": bbox_list_sep,
            "logi_list_sep": logi_list_sep,
        }
        if len(html_context) > 0:
            result["html_context"] = html_context

        FileUtils.dump_json(result_file_name, result)

        if self.debug:
            src_img = cv2.imread(image_file)
            img_save_path = os.path.join(self.output_dir, f"table_res_{file_name}.jpg")
            FileUtils.check_file_exists(img_save_path)

            if self.model in ["CenterNet", "SLANet"]:
                image_draw = OcrCommonUtils.draw_boxes(image_full=src_img,
                                                       det_result=bbox_list,
                                                       color="red", width=3,
                                                       return_numpy=False)
                image_draw.save(img_save_path)
            elif self.model in ["Lore"]:
                OcrCommonUtils.draw_lore_bboxes(image_name=src_img,
                                                boxes=bbox_list,
                                                logits=logi_list,
                                                save_name=img_save_path)

            else:
                if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
                    img = OcrInferUtils.draw_rectangle(src_img, bbox_list)
                else:
                    img = OcrInferUtils.draw_boxes(src_img, bbox_list)
                cv2.imwrite(img_save_path, img)

            logger.info("save vis result to {}".format(img_save_path))
        return table_cells_v2, table_cell_metric

    def post_process_bbox_and_logits(self, table_bboxs, logits, image_file, line_tol=10):
        """
        对识别的表格cell 和 logit坐标进行后处理修正

        :param table_bboxs:
        :param logits:
        :param image_file:
        :param line_tol:
        :return:
        """
        joint_point = []

        for box in table_bboxs:
            joint_point.extend(box.reshape(4, 2))

        logger.info(f"total : {len(joint_point)}")

        cols = []
        rows = []

        for x, y in joint_point:
            cols.append(x)
            rows.append(y)

        # sort horizontal and vertical segments
        cols = PdfUtils.merge_close_lines(sorted(cols), line_tol=line_tol, last_merge_threold=20)
        rows = PdfUtils.merge_close_lines(sorted(rows, reverse=False), line_tol=line_tol, last_merge_threold=20)

        # 归一化 相交点
        new_joint_point = []
        exits_points = set()
        for x, y in joint_point:
            new_x = PdfImageProcessor.find_close_norm_x(x, norm_list=cols, atol=line_tol)
            new_y = PdfImageProcessor.find_close_norm_x(y, norm_list=rows, atol=line_tol)
            key = f"{int(new_x)}_{int(new_y)}"
            if key not in exits_points:
                new_joint_point.append((new_x, new_y))
                exits_points.add(key)

        if self.debug:
            src_img = cv2.imread(image_file)
            file_name = FileUtils.get_file_name(image_file)

            thickness = 2
            color2 = (255, 0, 255)
            for index, point in enumerate(new_joint_point):
                cv2.circle(src_img, (int(point[0]), int(point[1])), 10, color2, thickness)

            save_image_file = f"{self.output_dir}/{file_name}_tsr_joint_point.jpg"
            FileUtils.check_file_exists(save_image_file)
            cv2.imwrite(save_image_file, src_img)

            logger.info(f"保存表格交点图像：{save_image_file}")

        min_col = min(cols)
        max_col = max(cols)
        min_row = min(rows)
        max_row = max(rows)
        one_col_width = abs(max_col - min_col)
        one_row_height = abs(max_row - min_row)

        cells = []
        for index, bbox in enumerate(table_bboxs):
            bboxs = bbox.reshape(4, 2)
            logit_axis = logits[index].astype(np.int32) if len(logits) > 0 else None
            new_bboxs = []
            for x, y in bboxs:
                new_x = PdfImageProcessor.find_close_norm_x(x, norm_list=cols, atol=line_tol)
                new_y = PdfImageProcessor.find_close_norm_x(y, norm_list=rows, atol=line_tol)
                new_bboxs.append((new_x, new_y))
            cell = Cell(x1=new_bboxs[0][0], y1=new_bboxs[0][1],
                        x2=new_bboxs[2][0], y2=new_bboxs[2][1],
                        logit_axis=logit_axis, )
            cell.text = "test_text"
            cells.append(cell)

        all_cell_results = TableProcessUtils.modify_cell_info(cells,
                                                              cols=cols,
                                                              rows=rows,
                                                              one_col_width=one_col_width,
                                                              one_row_height=one_row_height,
                                                              is_pdf=True)
        logit_diff_total = 0
        for index, cell in enumerate(all_cell_results):
            if not cell.check_pred_logit():
                logit_diff_total += 1
                # logger.info(f"cell logit axis not match: {index} - {cell}")
        logger.info(f"cell logit axis not match: {logit_diff_total} -总共：{len(all_cell_results)}")

        if self.debug:
            self.show_table_cell(table_cells=all_cell_results,
                                 image_file=image_file, )
        return all_cell_results

    def show_table_cell(self, table_cells: List[Cell],
                        image_file,
                        sep_file_name="_table_cell.jpg"):
        """
        显示表格的网格图片

        :param table_cells:
        :param image_file:
        :param sep_file_name:
        :return:
        """
        color_list = CommonUtils.get_color_list()
        src_img = cv2.imread(image_file)
        file_name = FileUtils.get_file_name(image_file)

        thickness = 2

        for index, cell in enumerate(table_cells):
            color = color_list[cell.row_index % len(color_list)]
            cv2.rectangle(src_img, (cell.x1_round, cell.y1_round), (cell.x2_round, cell.y2_round), color, thickness)

        save_image_file = f"{self.output_dir}/{file_name}{sep_file_name}"
        FileUtils.check_file_exists(save_image_file)
        cv2.imwrite(save_image_file, src_img)

        logger.info(f"保存表格cell图像：{len(table_cells)} - {save_image_file}")
        return save_image_file

    def post_process_bbox_and_logits_v2(self, table_bboxs, logits, image_file, line_tol=10, line_diff=0.5):
        """
        对识别的表格cell 和 logit坐标进行后处理修正

        :param table_bboxs:
        :param logits:
        :param image_file:
        :param line_tol:
        :return:
        """
        linewidth = line_diff * 2
        line_rects = []
        joint_point = []
        for box in table_bboxs:
            points = box.reshape(4, 2)
            joint_point.extend(points)

            (x1, y1) = points[0]
            (x2, y2) = points[1]
            (x3, y3) = points[2]
            (x4, y4) = points[3]
            rect1 = LTRect(linewidth=linewidth, bbox=(x1, y1 - line_diff, x2, y1 + line_diff), )
            rect2 = LTRect(linewidth=linewidth, bbox=(x2 - line_diff, y2, x3 + line_diff, y3), )
            rect3 = LTRect(linewidth=linewidth, bbox=(x4, y4 - line_diff, x3, y4 + line_diff), )
            rect4 = LTRect(linewidth=linewidth, bbox=(x1 - line_diff, y1, x4 + line_diff, y4), )
            line_rects.extend([rect1, rect2, rect3, rect4])

        logger.info(f"total : {len(joint_point)}")

        joint_point_rects = []
        for x, y in joint_point:
            bbox = (x - line_diff, y, x + line_diff, y)
            point = LTRect(linewidth=linewidth, bbox=bbox, )
            joint_point_rects.append(point)

        return line_rects, joint_point_rects
