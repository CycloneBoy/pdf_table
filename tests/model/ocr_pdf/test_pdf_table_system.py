#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : test_pdf_table_system.py
# @Author   : sl
# @Date     : 2024/11/9 - 14:50
import os
import unittest
from typing import List

from pdftable.entity import PdfTaskParams
from pdftable.model import TableProcessUtils
from pdftable.model.ocr_pdf import OCRDocumentConfig
from pdftable.model.ocr_pdf.ocr_system_task import OcrSystemTask
from pdftable.utils import FileUtils, TimeUtils, PdfUtils, Constants, logger, PdfTableExtractUtils, CommonUtils


class TestPdfTableSystem(unittest.TestCase):
    run_params: PdfTaskParams = None
    predictor_type = "pytorch"
    pdf_check_dir = FileUtils.get_output_dir_with_time(add_now_end=False)
    base_dir = f"{Constants.PDF_CACHE_BASE}/table_file"
    pdf_dir = f"{base_dir}/pdf_file"
    # debug = True
    debug = False

    def build_ocr(self, output_dir=None) -> OcrSystemTask:
        ocr_task_type = "document"

        # lang = "ch"
        lang = "en"

        # table_task_type = "wireless"
        table_task_type = "ptn"
        # table_task_type = "wtw"

        # table_structure_model = "CenterNet"
        # table_structure_model = "SLANet"
        # table_structure_model = "Lore"

        # table_structure_model = "Lgpma"
        table_structure_model = "MtlTabNet"
        # table_structure_model = "TableMaster"
        # table_structure_model = "LineCell"
        # table_structure_model = "LoreAndLineCell"

        layout_model = "picodet"
        # layout_model="DocXLayout"
        layout_model_task_type = lang

        if self.run_params is not None:
            if self.run_params.tsr_model_name is not None:
                table_structure_model = self.run_params.tsr_model_name
            if self.run_params.table_task_type is not None:
                table_task_type = self.run_params.table_task_type
            if self.run_params.lang is not None:
                lang = self.run_params.lang
            if self.run_params.ocr_task_type is not None:
                ocr_task_type = self.run_params.ocr_task_type

        ocr_config = OCRDocumentConfig(task_type=ocr_task_type,
                                       lang=lang,
                                       table_task_type=table_task_type,
                                       table_structure_model=table_structure_model,
                                       layout_model=layout_model,
                                       layout_model_task_type=layout_model_task_type)
        if output_dir is None:
            output_dir = self.pdf_check_dir
        ocr_document = OcrSystemTask(ocr_config,
                                     debug=self.debug,
                                     output_dir=output_dir,
                                     predictor_type=self.predictor_type)
        return ocr_document

    def ocr(self, file_name, page=None, src_id=None, output_dir=None):
        ocr_document: OcrSystemTask = self.build_ocr()

        if output_dir is not None:
            ocr_document.set_output_dir(output_dir)
        outputs, metric = ocr_document(file_name, src_id=src_id, page=page)
        return outputs, metric

    def run_extract_pdf_table(self, file_url, pages="1", temp_dir=None,
                              password=None, src_id=None,
                              delete_check_success=False,
                              need_check_imaged_pdf=True,
                              is_image_pdf=False):
        """
        提取一个PDF

        :param file_url:
        :param pages:
        :param temp_dir:
        :param password:
        :param src_id:
        :param delete_check_success:
        :param need_check_imaged_pdf:
        :param is_image_pdf:
        :return:
        """
        run_time = TimeUtils.now_str_short()
        output_dir = f"{self.pdf_check_dir}/{run_time}"
        self.output_dir = output_dir
        ocr_document: OcrSystemTask = self.build_ocr(output_dir=output_dir)

        file_name = PdfUtils.download_pdf_before_check(file_url=file_url, pdf_dir=self.pdf_dir)
        is_pdf_file = FileUtils.is_pdf_file(file_name)
        if is_pdf_file:
            try:
                if need_check_imaged_pdf:
                    is_image_pdf = PdfUtils.check_is_imaged_pdf_v2(file_name=file_name,
                                                                   pdf_dir=self.pdf_dir)
                    # is_image_pdf = PdfUtils.check_is_imaged_pdf(file_name)
            except Exception as e:
                logger.info(f"判断文件是否是图片型PDF出现异常，删除文件：{e} - {file_name}")

            page_list = PdfUtils.build_pdf_pages_list(file_name=file_name, password=password, pages=pages)

            if temp_dir is None:
                temp_dir = output_dir
                logger.info(f"parse pdf output dir: {temp_dir}")

            save_file_name = PdfTableExtractUtils.save_pdf_page(filepath=file_name, pages=page_list,
                                                                temp_dir=temp_dir, password=password)
        else:
            logger.info(f"当前文件不是PDF文件：{file_name}")
            save_file_name = [file_name]

        run_save_file_name = []
        run_metric = []
        for index, run_page_file in enumerate(save_file_name):
            raw_file_name = FileUtils.get_file_name(run_page_file)
            if is_pdf_file:
                page = int(raw_file_name[5:])
            else:
                page = 1

            if is_image_pdf:
                pdf_page, other_page = PdfUtils.extract_pdf_image(file_name=run_page_file,
                                                                  output_dir=output_dir,
                                                                  page_file_name=raw_file_name,
                                                                  dim_limit=0, rel_size=0, abs_size=10,
                                                                  do_rotate=False)

                # save_page_image_file = PdfUtils.extract_pdf_image_page(file_name=run_page_file,
                #                                                        output_dir=output_dir,
                #                                                        page_file_name=raw_file_name,
                #                                                        dim_limit=0, )
                if len(pdf_page) > 0:
                    run_page_file = pdf_page[0]
                else:
                    logger.info(f"图片型PDF没有提取出图片,采用PDF提取：{run_page_file}")

            logger.info(f"开始提取：{index} - {page} - {run_page_file} ")
            outputs, metric = ocr_document(run_page_file, src_id=src_id, page=page)

            run_save_file_name.append(run_page_file)
            run_metric.append(metric)

        # 输出全部页面
        show_url = self.make_pdf_output_html(save_file_name=run_save_file_name,
                                             output_dir=output_dir,
                                             run_metric=run_metric)
        return show_url

    def make_pdf_output_html(self, save_file_name: List, output_dir, run_metric: List):
        """
        构造显示结果

        :param save_file_name:
        :param output_dir:
        :param run_metric:
        :return:
        """
        raw_filename = f"a_pdf_{TimeUtils.now_str_short()}"
        page_show_filename = f"{raw_filename}_show.html"
        html_filename = f"{raw_filename}.html"
        html_show_file = os.path.join(output_dir, page_show_filename)
        html_file = os.path.join(output_dir, html_filename)
        logger.info(f"page html show file : {html_show_file}")

        FileUtils.delete_file(html_show_file)

        TableProcessUtils.write_html_result_header(html_show_file)

        all_htmls = []
        for index, run_page_file in enumerate(save_file_name):
            raw_file_name = FileUtils.get_file_name(run_page_file)
            page = raw_file_name

            predict_file = f"{output_dir}/{raw_file_name}.html"
            if FileUtils.check_file_exists(predict_file):
                pdf_pred_html = FileUtils.read_to_text(predict_file)
            else:
                pdf_pred_html = str(run_metric[index])
            pdf_image = FileUtils.get_file_name(run_page_file, add_end=True)
            pdf_image = FileUtils.get_pdf_to_image_file_name(pdf_image)
            pdf_image_src = pdf_image.replace(".jpg", ".png")
            html_content = [
                "<tr>",
                f'<td colspan=4 align=center><p style="color:red;">'
                f'页面解析结果{run_page_file}</p></td>',
                "</tr>",

                "<tr>",
                f'<td> {page} <br/>',
                f'<td><img src="{pdf_image_src}" width=640></td>',
                f'<td><img src="{pdf_image}" width=640></td>',
                f'<td>{pdf_pred_html}</td>',
                "</tr>",
            ]
            FileUtils.append_to_text(html_show_file, html_content)

            all_htmls.append(pdf_pred_html)

        TableProcessUtils.write_html_result_footer(html_show_file)

        FileUtils.delete_file(html_file)
        FileUtils.save_to_text(html_file, content="@@@\n".join(all_htmls))

        result_dir_url = CommonUtils.get_result_http_server(output_dir=html_file)

        show_url = f'{result_dir_url}/{page_show_filename}'
        logger.info(f"解析结果目录链接：{result_dir_url}")
        logger.info(f"解析对比显示链接：{show_url}")
        logger.info(f"解析最终显示链接：{f'{result_dir_url}/{html_filename}'}")

        return show_url

    def test_do_one_pdf(self):
        # 中文数字型PDF
        # file_url = "https://disc.static.szse.cn/disc/disk03/finalpage/2023-06-09/52a65d41-637d-4e66-879b-5a5744092d2e.PDF"

        # 中文图片型PDF
        # file_url = "http://static.cninfo.com.cn/finalpage/2020-12-25/1208981112.PDF"
        # file_url = "http://static.cninfo.com.cn/finalpage/2018-07-31/1205240445.PDF"

        # 中文图片
        # file_url = "/nlp_data/pdftable/outputs/pdf/inference_results/pdf_debug/2024-11-10/20241110_161330/page-3.jpg"

        # english pdf
        file_url = "https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf"
        # file_url = f"/nlp_data/pdftable/outputs/pdf/inference_results/pdf_debug/2024-11-16/20241116_132532/page-6.png"
        # file_url = f"{Constants.SRC_IMAGE_DIR}/table_en_01.png"

        self.run_extract_pdf_table(file_url, pages="all", )


    def test_pdf_table(self):
        pass
