#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_ocr_document
# @Author  ：cycloneboy
# @Date    ：20xx/7/14 16:58
import time
import traceback
from collections import defaultdict
from pprint import pprint

from tqdm import tqdm

from pdftable.model import TableProcessUtils
from pdftable.model.ocr_pdf import OCRDocumentConfig, OcrDetectionTask, \
    OcrTableStructureTask, OcrLayoutTask
from pdftable.model.ocr_pdf.ocr_recognition_task import OcrRecognitionTask
from pdftable.model.ocr_pdf.ocr_system_task import OcrSystemTask
from pdftable.model.ocr_pdf.ocr_table_cell_task import OcrTableCellTask
from pdftable.model.ocr_pdf.ocr_table_task import OcrTableTask
from pdftable.model.ocr_pdf.ocr_text_task import OcrTextTask
from pdftable.model.ocr_pdf.table.table_metric import TEDS
from pdftable.utils import logger, Constants, TimeUtils, FileUtils, CommonUtils
from pdftable.utils.benchmark_utils import print_timings


class RunOcrDocument(object):

    def __init__(self):
        self.base_dir = f'f"{Constants.DATA_DIR}/pdf/table_file/temp_file'
        self.html_base_dir = Constants.HTML_BASE_DIR

        self.image_dir = Constants.SRC_IMAGE_DIR
        self.recognition_img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'

        # self.img_url = f"{self.image_dir}/table_01.jpg"
        self.img_url = f"{self.image_dir}/page01.png"

        self.predictor_type = "pytorch"
        # self.predictor_type = "onnx"
        self.output_dir = FileUtils.get_output_dir_with_time()

        self.pdf_image_dir = f"{self.html_base_dir}/pdf_image/2023-07-25"
        self.pdf_image_output_dir = f"{self.html_base_dir}/pdf_debug/{TimeUtils.get_time()}"
        # self.pdf_image_output_dir = f"{self.html_base_dir}/pdf_debug/2024-01-12"
        self.src_id = 1
        # self.debug_image = f"{self.pdf_image_dir}/{self.src_id}/1205240445_page_003_rotate.png"
        self.debug_image = f"{Constants.DATA_DIR}/pdf/ocr_file/page-27.pdf"
        # self.debug_image = f"{Constants.DATA_DIR}/pdf/ocr_file/page-27.png"
        self.pubtabnet_dir = f"{Constants.DATA_DIR}/table/pubtabnet"

    def build_ocr(self):
        task_type = "document"

        lang = "ch"
        # lang = "en"

        # table_task_type = "wireless"
        # table_task_type = "ptn"
        table_task_type = "wtw"

        # table_structure_model = "CenterNet"
        # table_structure_model = "SLANet"
        table_structure_model = "Lore"

        ocr_config = OCRDocumentConfig(task_type=task_type,
                                       lang=lang,
                                       table_task_type=table_task_type,
                                       table_structure_model=table_structure_model)

        # output_dir = f"{self.base_dir}/ocr_document/{TimeUtils.get_time()}"
        # output_dir = self.output_dir
        output_dir = f"{self.pdf_image_output_dir}/{self.src_id}"

        # ocr_document = OcrDocument(ocr_config, debug=True, output_dir=output_dir)

        ocr_document = OcrSystemTask(ocr_config,
                                     debug=True,
                                     output_dir=output_dir,
                                     predictor_type=self.predictor_type, )
        return ocr_document

    def run(self):
        ocr_document = self.build_ocr()

        # image = self.img_url
        # image = f'{self.base_dir}/test_image/ocr_table2.jpg'
        # image = f"{self.pdf_image_dir}/{self.src_id}/1208981112_page_3.png"
        image = self.debug_image
        image = "https://disc.static.szse.cn/disc/disk03/finalpage/2023-06-09/52a65d41-637d-4e66-879b-5a5744092d2e.PDF"
        det_result, ocr_result, metric = ocr_document(image, src_id=self.src_id)

    def batch_run(self):
        file_list = FileUtils.list_file_prefix(file_dir=f"{self.pdf_image_dir}/{self.src_id}", add_parent=True,
                                               sort=True, )

        logger.info(f"file list: {len(file_list)}")
        ocr_document = self.build_ocr()
        begin = time.time()

        all_use_time = defaultdict(list)
        for index, image in enumerate(file_list):
            det_result, ocr_result, metric = ocr_document(image, src_id=self.src_id)

            for k, v in metric.items():
                if k in ["result"]:
                    continue
                if isinstance(v, dict) and "use_time" in v:
                    all_use_time[k].append(v["use_time"])

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

    def run_ocr_detection_task(self):
        # ocr_detection_task = OcrDetectionTask(task="ocr_detection", model="db")
        ocr_detection_task = OcrDetectionTask(task="ocr_detection",
                                              model="db_pp",
                                              backbone="PP-OCRv4",
                                              predictor_type="onnx",
                                              output_dir=self.output_dir,
                                              debug=True,
                                              lang="ch")

        image = self.img_url
        image = f"{self.image_dir}/table_01.jpg"
        image = f"{self.image_dir}/page01.png"

        result = ocr_detection_task(image)
        logger.info(f"result: {result}")

        TableProcessUtils.check_pdf_text_need_rotate90(result[0])

    def run_ocr_recognition_task(self):
        lang = "ch"
        # lang = "en"

        test_image_config = {
            "ch": [
                "word_2.jpg"
            ],
            "en": [
                "word_1.png"
            ]
        }

        model = "PP-OCRv4"
        # model = "ConvNextViT"
        model = "CRNN"
        # model = "LightweightEdge"

        ocr_recognition_task = OcrRecognitionTask(task="ocr_recognition",
                                                  model=model,
                                                  task_type="general",
                                                  output_dir=self.output_dir,
                                                  debug=True,
                                                  lang=lang
                                                  )

        image = self.recognition_img_url
        test_image = test_image_config.get(lang, ["word_1.png"])
        image = f"{Constants.SRC_IMAGE_DIR}/imgs_words/{lang}/{test_image[0]}"

        result = ocr_recognition_task(image)
        logger.info(f"result: {result}")

    def run_ocr_text_task(self):

        output_dir = self.output_dir

        lang = "ch"
        # lang = "en"
        # lang = "korean"
        # lang = "japan"

        image_dir_mapping = {
            "ch": "imgs/11.jpg",
            # "en": "imgs/ger_1.jpg",
            "en": "imgs/en_img_12.jpg",
            # "en": "imgs_words_en/word_19.png",
            "korean": "imgs/korean_1.jpg",
            "japan": "imgs/japan_2.jpg",
        }

        image_name = image_dir_mapping.get(lang, "")

        config = OCRDocumentConfig(detect_model="db_pp",
                                   detector="PP-OCRv4",
                                   recognizer="PP-OCRv4",
                                   lang=lang)
        ocr_text = OcrTextTask(config=config,
                               debug=True,
                               output_dir=output_dir, )

        # image = self.ocr_image
        image = f"{Constants.SRC_IMAGE_DIR}/{image_name}"
        res = ocr_text(image)
        logger.info(f"res: {res}")

    def run_ocr_table_structure_task(self):
        debug = True
        # debug = False

        # lang = "ch"
        lang = "en"

        # task_type = "wireless"
        task_type = "ptn"
        # task_type = "PubTabNet"
        # task_type = "wtw"
        # task_type = "FinTabNet"

        # model = "CenterNet"
        # model = "SLANet"
        # model = "Lore"
        # model = "Lgpma"
        # model = "MtlTabNet"
        model = "TableMaster"
        # model = "LineCell"
        # model = "LineCellPdf"

        output_dir = f"{self.output_dir}/{model}"

        # task = OcrTableStructureTask(predictor_type=self.predictor_type, )
        task = OcrTableStructureTask(model=model,
                                     predictor_type=self.predictor_type,
                                     output_dir=output_dir,
                                     lang=lang,
                                     task_type=task_type,
                                     debug=debug)

        # image = self.img_url
        if lang == "ch":
            image = f"{Constants.SRC_IMAGE_DIR}/layout_demo3.jpg"
            image = f"/nlp_data/pdftable/outputs/pdf/inference_results/pdf_debug/2024-11-09/1/20241109_191445/page-2_0.jpg"
        else:
            image = f"{Constants.SRC_IMAGE_DIR}/table_01.jpg"
            image = f"/root/autodl-tmp/pdftable/outputs/pdf/inference_results/pdf_debug/2024-11-16/20241116_132532/page-6.png"

        result = task(image)
        logger.info(f"result: {result}")

    def run_ocr_table_task(self):
        # output_dir = f"{self.pdf_image_output_dir}/{self.src_id}"
        output_dir = self.output_dir

        lang = "ch"
        # lang = "en"

        # table_structure_model = "CenterNet"
        table_structure_model = "SLANet"

        ocr_model = "PP-OCRv4"
        # ocr_model = "PP-Table"

        config = OCRDocumentConfig(detector=ocr_model,
                                   recognizer=ocr_model,
                                   table_structure_model=table_structure_model,
                                   lang=lang,
                                   debug=True, )

        task = OcrTableTask(config=config,
                            output_dir=output_dir, )

        # image = self.img_url
        if lang == "ch":
            image = f"{Constants.SRC_IMAGE_DIR}/layout_demo3.jpg"
        else:
            image = f"{Constants.SRC_IMAGE_DIR}/table_01.jpg"

        result = task(image)
        logger.info(f"result: {result}")

        # run eval table task
        # file_base_dir = self.pubtabnet_dir
        # task.eval_table(
        #     file_dir=f"{file_base_dir}/val",
        #     label_file=f"{file_base_dir}/PubTabNet_val.txt",
        #     output_dir=self.pdf_image_output_dir,
        #     run_name=f"_{TimeUtils.get_time()}",
        # )

    def run_ocr_layout_task(self):
        new_output_dir = self.output_dir
        model = "picodet"
        # model = "DocXLayout"

        # task_type = "ch"
        # task_type = "en"
        task_type = "table"

        task = OcrLayoutTask(model=model,
                             task_type=task_type,
                             output_dir=new_output_dir)
        image = self.img_url
        result = task(image)
        logger.info(f"result: {result}")

    def run_ocr_table_cell_task(self):
        output_dir = self.output_dir

        lang = "ch"
        # lang = "en"

        task = OcrTableCellTask(output_dir=output_dir)
        # image = f"{self.pdf_image_dir}/{self.src_id}/1208981112_page_3.png"
        # image = f"{self.pdf_image_dir}/{self.src_id}/1205240445_page_003_rotate.png"
        image = self.img_url

        if lang == "ch":
            image = f"{Constants.SRC_IMAGE_DIR}/layout_demo3.jpg"
        else:
            image = f"{Constants.SRC_IMAGE_DIR}/table_01.jpg"

        result = task(image)
        logger.info(f"result: {result}")

    def generate_cls_result_html(self, results):
        demo_list = {
            "now_str": TimeUtils.now_str(),
            "metric": {
                "total": len(results),
            },
            "headers": [
                'id', 'page', 'show_url'
            ],
            "check_list": results
        }

        template_name = f"image_cls_show.html"
        result = CommonUtils.render_html_template(template_name, demo_list)

        file_name = f"image_cls_show_{TimeUtils.now_str_short()}.html"
        save_file = f"{self.html_base_dir}/check_result/{file_name}"
        FileUtils.save_to_text(save_file, result)
        logger.info(f"save_file:{save_file}")
        show_url = f"http://localhost:9100/pdf_html/check_result/{file_name}"
        logger.info(f"show_url:{show_url}")

    def batch_run_tsr(self):
        # lang = "ch"
        lang = "en"

        # task_type = "wireless"
        task_type = "ptn"
        # task_type = "wtw"

        model = "SLANet"
        # model = "Lore"
        # model = "Lgpma"
        # model = "MtlTabNet"
        # model = "TableMaster"
        model_list = [
            "SLANet",
            "Lore",
            "Lgpma",
            "MtlTabNet",
            "TableMaster"
        ]

        image = f"{self.pubtabnet_dir}/val/PMC5755158_010_01.png"
        for model in model_list:
            output_dir = f"{self.pdf_image_output_dir}/{model}"
            task = OcrTableStructureTask(model=model,
                                         predictor_type=self.predictor_type,
                                         output_dir=output_dir,
                                         lang=lang,
                                         task_type=task_type)
            result = task(image)

    def eval_batch(self, model, file_dir, label_file):
        # lang = "ch"
        lang = "en"

        # task_type = "wireless"
        task_type = "ptn"
        # task_type = "wtw"

        output_dir = f"{self.pdf_image_output_dir}/{model}"
        task = OcrTableStructureTask(model=model,
                                     predictor_type=self.predictor_type,
                                     output_dir=output_dir,
                                     lang=lang,
                                     task_type=task_type)

        gt_html_dict = FileUtils.load_table_label_txt(label_file)
        logger.info(f"开始进行评估：{len(gt_html_dict)}")

        error_total = 0
        for img_name, gt_html in tqdm(gt_html_dict.items()):
            image_file = f"{file_dir}/{img_name}"
            if not FileUtils.check_file_exists(image_file):
                continue

            json_file = f"{output_dir}/table_infer_{FileUtils.get_file_name(img_name)}.json"
            if FileUtils.check_file_exists(json_file):
                continue

            try:
                result = task(image_file)
            except Exception as e:
                traceback.print_exc()
                error_total += 1
                pass
        logger.info(f"处理完毕：失败数量: {error_total} - {error_total / len(gt_html_dict):.3f}")
        return error_total

    def eval_table(self):
        """


        :return:
        """
        model_list = [
            # "SLANet",
            # "Lore",
            "Lgpma",
            # "TableMaster",
            # "MtlTabNet"
        ]

        metric = {}
        for model in model_list:
            begin_time = time.time()
            begin = TimeUtils.now_str()
            error_total = self.eval_batch(model=model, file_dir=f"{self.pubtabnet_dir}/val",
                                          label_file=f"{self.pubtabnet_dir}/PubTabNet_val.txt")

            use_time = time.time() - begin_time
            end = TimeUtils.now_str()
            metric[model] = {
                "begin": begin,
                "end": end,
                "use": TimeUtils.calc_diff_time(begin_time=begin, end_time=end),
                "use_time": use_time,
                "error_total": error_total
            }

        pprint(metric)
        FileUtils.save_to_json(f"{self.pdf_image_output_dir}/metric_{TimeUtils.now_str_short()}.json", metric)

    def clean_tsr(self, html):
        # html = html.replace("<thead>", "</thead>")
        # html = html.replace("<tbody>", "</tbody>")
        return html

    def eval_tsr_result(self):
        model_list = [
            # "SLANet",
            # "Lore",
            # "Lgpma",
            # "TableMaster",
            "MtlTabNet"
        ]

        all_metric = {}
        label_file = f"{self.pubtabnet_dir}/PubTabNet_val.txt"
        gt_html_dict = FileUtils.load_table_label_txt(label_file)

        run_time = TimeUtils.now_str_short()

        metric_all_file = f"{self.pdf_image_output_dir}/metric_all_{run_time}.json"
        metric_append_file = metric_all_file.replace('.json', ".txt")
        for model in model_list:
            begin_time = time.time()
            begin = TimeUtils.now_str()
            output_dir = f"{self.pdf_image_output_dir}/{model}"

            pred_htmls = []
            gt_htmls = []
            eval_total = 0
            for img_name, gt_html in tqdm(gt_html_dict.items()):
                image_file = f"{output_dir}/{FileUtils.get_file_name(img_name)}_table_structure.html"
                if not FileUtils.check_file_exists(image_file):
                    continue

                eval_total += 1

                json_file = f"{output_dir}/table_infer_{FileUtils.get_file_name(img_name)}.json"
                result = FileUtils.load_json(json_file)
                pred_html = result["structure_str_list"][0]
                if len(result.get("html_context", "")) > 2:
                    pred_html = result["html_context"]
                pred_htmls.append(self.clean_tsr(pred_html))
                gt_htmls.append(self.clean_tsr(gt_html))

            logger.info(f"开始进行计算TEDS: {len(pred_htmls)} - {eval_total}")
            begin_time = time.time()
            begin_time_str = TimeUtils.now_str()
            run_time = TimeUtils.now_str_short()

            # compute teds
            teds = TEDS(n_jobs=16, structure_only=True)
            scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
            teds_metric = sum(scores) / len(scores)
            tp = [item for item in scores if item == 1.0]
            acc = len(tp) / len(scores)

            use_time = time.time() - begin_time

            end_time_str = TimeUtils.now_str()
            use_time2 = TimeUtils.calc_diff_time(begin_time_str, end_time_str)
            metric = {
                "model": model,
                "begin_time": begin_time_str,
                "end_time": end_time_str,
                "use_time": use_time,
                "use_time2": use_time2,
                "total_label": len(gt_html_dict),
                "eval_total": eval_total,
                "teds_metric": teds_metric,
                "acc": acc,
                "tp": len(tp),
            }
            all_metric[model] = metric
            pprint(metric)

            metric_file = f"{self.pdf_image_output_dir}/metric_{model}_{run_time}.json"
            FileUtils.dump_json(metric_file, metric)

            FileUtils.save_to_text(metric_append_file, content=f"{str(metric)}\n", mode="a")

            logger.info(f'teds: {model} - {teds_metric:.3f}')
            logger.info(f"评估文件：{metric_append_file}")

        FileUtils.dump_json(metric_all_file, all_metric)


def main():
    runner = RunOcrDocument()
    # runner.run()
    # runner.batch_run()
    # runner.run_ocr_detection_task()
    # runner.run_ocr_recognition_task()
    # runner.run_ocr_text_task()
    runner.run_ocr_table_structure_task()

    # runner.run_ocr_table_task()
    # runner.run_ocr_layout_task()
    # runner.run_ocr_table_cell_task()

    # runner.batch_run_tsr()
    # runner.eval_table()
    # runner.eval_tsr_result()


if __name__ == '__main__':
    main()
