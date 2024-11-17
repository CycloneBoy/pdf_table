#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : pdf_table
# @File     : main.py
# @Author   : sl
# @Date     : 2024/11/17 - 13:03
import os
import traceback
from typing import List

from transformers import HfArgumentParser

from pdftable.entity.common_entity import PdfTableCliArguments
from pdftable.model import TableProcessUtils
from pdftable.model.ocr_pdf import OCRDocumentConfig
from pdftable.model.ocr_pdf.ocr_system_task import OcrSystemTask
from pdftable.utils import TimeUtils, PdfUtils, FileUtils, logger, PdfTableExtractUtils, Constants, CommonUtils


class PdfTableCli(object):

    def __init__(self, cli_args: PdfTableCliArguments):
        self.cli_args = cli_args
        self.ocr_document: OcrSystemTask = None

    def build_ocr(self, output_dir=None) -> OcrSystemTask:
        ocr_config = OCRDocumentConfig(
            detector=self.cli_args.detect_model,
            thresh=self.cli_args.detect_db_thresh,
            recognizer=self.cli_args.recognizer_model,
            task_type=self.cli_args.recognizer_task_type,
            lang=self.cli_args.lang,
            debug=self.cli_args.debug,
            table_structure_model=self.cli_args.table_structure_model,
            table_task_type=self.cli_args.table_structure_task_type,
            layout_model=self.cli_args.layout_model,
            layout_model_task_type=self.cli_args.lang)

        if output_dir is None:
            output_dir = self.cli_args.output_dir

        ocr_document = OcrSystemTask(ocr_config,
                                     debug=self.cli_args.debug,
                                     output_dir=output_dir)
        return ocr_document

    def ocr(self, file_name, page=None, src_id=None, output_dir=None):
        ocr_document: OcrSystemTask = self.build_ocr()

        if output_dir is not None:
            ocr_document.set_output_dir(output_dir)
        outputs, metric = ocr_document(file_name, src_id=src_id, page=page)
        return outputs, metric

    def run_extract_pdf_table(self, file_url, pages="all", temp_dir=None,
                              password=None, src_id=None,
                              need_check_imaged_pdf=True,
                              is_image_pdf=False,
                              output_dir=None,
                              merge_sep=None):
        """
        提取一个PDF

        :param file_url:
        :param pages:
        :param temp_dir:
        :param password:
        :param src_id:
        :param need_check_imaged_pdf:
        :param is_image_pdf:
        :return:
        """
        run_time = TimeUtils.now_str_short()
        if output_dir is None:
            output_dir = self.cli_args.output_dir

        if output_dir is None:
            output_dir = f"{Constants.OUTPUT_DIR}/{run_time}"

        self.cli_args.output_dir = output_dir

        pdf_dir = output_dir
        if self.ocr_document is None:
            ocr_document: OcrSystemTask = self.build_ocr(output_dir=output_dir)
            self.ocr_document = ocr_document
        else:
            ocr_document = self.ocr_document
            if output_dir is not None:
                ocr_document.set_output_dir(output_dir)

        file_name = PdfUtils.download_pdf_before_check(file_url=file_url, pdf_dir=pdf_dir)
        is_pdf_file = FileUtils.is_pdf_file(file_name)
        if is_pdf_file:
            try:
                if need_check_imaged_pdf:
                    is_image_pdf = PdfUtils.check_is_imaged_pdf_v2(file_name=file_name,
                                                                   pdf_dir=pdf_dir)
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
        all_run_result = []
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

                if len(pdf_page) > 0:
                    run_page_file = pdf_page[0]
                else:
                    logger.info(f"图片型PDF没有提取出图片,采用PDF提取：{run_page_file}")

            logger.info(f"开始提取：{index} - {page} - {run_page_file} ")
            try:
                outputs, metric = ocr_document(run_page_file, src_id=src_id, page=page)

                run_save_file_name.append(run_page_file)
                run_metric.append(metric)
                all_run_result.append(outputs)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"提取出错 {index} - {page} - {run_page_file} ：{e}")

        # 输出全部页面
        html_result = self.make_pdf_output_html(save_file_name=run_save_file_name,
                                                output_dir=output_dir,
                                                run_metric=run_metric,
                                                merge_sep=merge_sep)
        html_result["all_run_result"] = all_run_result
        return html_result

    def make_pdf_output_html(self, save_file_name: List, output_dir, run_metric: List,
                             merge_sep=None):
        """
        构造显示结果

        :param save_file_name:
        :param output_dir:
        :param run_metric:
        :return:
        """
        if merge_sep is None:
            merge_sep = "@@@@@@"

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
        pdf_html = f"{merge_sep}\n".join(all_htmls)
        FileUtils.save_to_text(html_file, pdf_html)

        result_dir_url = CommonUtils.get_result_http_server(output_dir=html_file)
        show_url = f'{result_dir_url}/{page_show_filename}'
        logger.info(f"解析结果目录链接：{result_dir_url}")
        logger.info(f"解析对比显示链接：{show_url}")
        logger.info(f"解析最终显示链接：{f'{result_dir_url}/{html_filename}'}")

        logger.info(f"pdf to html result file name：{html_file}")
        result = {
            "html_file": html_file,
            "html": pdf_html,
            "page_html": all_htmls
        }
        return result


def main():
    parser = HfArgumentParser(PdfTableCliArguments)
    args = parser.parse_args_into_dataclasses()

    cli_args = args[0]
    cli_task = PdfTableCli(cli_args=cli_args)
    cli_task.run_extract_pdf_table(file_url=cli_args.file_path_or_url,
                                   pages=cli_args.pages,
                                   output_dir=cli_args.output_dir,
                                   merge_sep=cli_args.html_page_merge_sep,
                                   )


if __name__ == '__main__':
    main()
