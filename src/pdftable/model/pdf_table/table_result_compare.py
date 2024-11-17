#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_result_compare
# @Author  ：cycloneboy
# @Date    ：20xx/6/26 13:40
import difflib
import re
from typing import List

from pdftable.entity import HtmlTableCompareType
from ...utils import (
    logger,
    PdfUtils,
    CommonUtils,
    MatchUtils
)

__all__ = [
    'TableResultCompare'
]

"""
TABLE HTML result compare 
"""


class TableResultCompare(object):

    def __init__(self):
        pass

    def check_pred_table_html(self, pred_html, label_html):
        """
        校验 解析的结果

        :param pred_html:
        :param label_html:
        :return:
        """
        check_type = ""
        flag = False
        pred_html_clean = self.norm_html(pred_html)
        label_html_clean = self.norm_html(label_html)
        if pred_html_clean == label_html_clean:
            flag = True
            check_type = "same"

        logger.info(f"pred_html_clean : {pred_html_clean}")
        logger.info(f"label_html_clean: {label_html_clean}")
        logger.info(f"check result: {flag} - {check_type}")

        #  移除width
        diff_result2 = ""
        if not flag:
            pred_html_clean2 = self.norm_html_remove_width(pred_html_clean)
            label_html_clean2 = self.norm_html_remove_width(label_html_clean)
            if pred_html_clean2 == label_html_clean2:
                flag = True
                check_type = "remove_width_same"

            diff_result2, raw_diff_result = self.compare_diff(pred_html_clean2, label_html_clean2)

            diff_flag, diff_check_type = self.analysis_diff_result(diff_result=raw_diff_result)
            if diff_flag:
                flag = True
                check_type = diff_check_type.name.lower()
                logger.info(f"check diff result : {diff_check_type.desc}")

            logger.info(f"pred_html_clean2 : {pred_html_clean2}")
            logger.info(f"label_html_clean2: {label_html_clean2}")
            logger.info(f"check result: {flag} - {check_type} - \n{diff_result2}")

        pred_html_show = self.process_show_html(pred_html)
        label_html_show = self.process_show_html(label_html)

        diff_result = ""
        diff_html_show = ""
        # diff = difflib.Differ()
        # diff_result = diff.compare(pred_html_clean, label_html_clean)
        # diff_result = CommonUtils.clean_sentence_remove_space(''.join(list(diff_result)))
        # diff_html_show = self.process_show_html(diff_result)
        # logger.info(f"check diff_result: {diff_result}")

        # diff_html = difflib.HtmlDiff()
        # diff_result_html = diff_html.make_table(pred_html_clean.split("</td>"), label_html_clean.split("</td>"))
        # logger.info(f"check diff_result_html: {diff_result_html}")
        diff_result_html = ""

        check_metric = {
            "pred_html": pred_html,
            "label_html": label_html,
            "pred_html_clean": pred_html_clean,
            "label_html_clean": label_html_clean,

            "pred_html_show": pred_html_show,
            "label_html_show": label_html_show,
            "diff_result": diff_result,
            "diff_html_show": diff_html_show,
            "diff_html_show2": diff_result2,
            "diff_result_html": diff_result_html,
            "flag": flag,
            "check_type": check_type,
        }

        return flag, check_metric

    def norm_html(self, html):
        """
        HTML 对比规范化

        :param html:
        :return:
        """
        html_clean = CommonUtils.clean_sentence_remove_space(html)

        replace_dict = {
            "'": '"',
            ".": '。',
            "％": '%',
            "“": '"',
            "<br/>": "",
            "pdf-borderless-table": "",
            "pdf-table": "pdf-ocr-table",
        }

        for k, v in replace_dict.items():
            html_clean = str(html_clean).replace(k, v)
        return html_clean

    def norm_html_text(self, html):
        """
        HTML 对比规范化

        :param html:
        :return:
        """
        html_clean = CommonUtils.clean_sentence_remove_space(html)

        replace_dict = {
            "'": '"',
            "<br/>": "",
            "pdf-borderless-table": "",
            "pdf-table": "pdf-ocr-table",
            "％": "%",
            "“": '"',
            ".": '。',
            'border="1"': '',
        }

        for k, v in replace_dict.items():
            html_clean = str(html_clean).replace(k, v)
        return html_clean

    def norm_html_remove_width(self, html):
        """
        HTML 对比规范化

        :param html:
        :return:
        """
        html_clean = CommonUtils.clean_sentence_remove_space(html)
        html_res = MatchUtils.clean_html_table_width(html_clean)
        res = str(html_res).replace("'", '"').replace("<br/>", "")
        return res

    def norm_html_remove_tag_and_space(self, html):
        text_content = PdfUtils.html_to_text_pure(src_html="\n".join(html))
        clean_text_content = CommonUtils.clean_sentence_remove_space(''.join(text_content))
        return clean_text_content

    def process_show_html(self, html, show_length=50):
        results = []
        for index in range(0, len(html), show_length):
            results.append(html[index:index + show_length])
        new_html = "\n".join(results)
        new_html = f"<xmp>{new_html}</xmp>"
        return new_html

    def compare_diff(self, src, dest):
        diff = difflib.Differ()

        # src_list = src.split("</td>")
        # dest_list = dest.split("</td>")

        # diff_result = diff.compare([src], [dest])
        # diff_result_show = ''.join(diff_result)
        # logger.info(f"diff: {diff_result_show}")

        a = src
        b = dest
        s = difflib.SequenceMatcher(None, a, b)
        diff_result_show = []
        show_length = 50
        diff_result = []
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            a_span = a[i1:i2] if i2 - i1 < show_length else a[i1:i1 + show_length]
            b_span = b[j1:j2] if j2 - j1 < show_length else b[j1:j1 + show_length]
            msg = '{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, a_span, b_span)
            diff_result_show.append(msg)
            diff_result.append([tag, i1, i2, j1, j2, a[i1:i2], b[j1:j2]])

        show_msg = '\n'.join(diff_result_show)
        new_html = f"<xmp>\n{show_msg}\n</xmp>"
        return new_html, diff_result

    def analysis_diff_result(self, diff_result: List):
        """
        分析diff result

        :param diff_result:
        :return:
        """
        check_type = HtmlTableCompareType.DIFF
        flag = False

        if len(diff_result) == 3:
            item1 = diff_result[0][0]
            item2 = diff_result[1][0]
            item3 = diff_result[2][0]

            a_span = diff_result[1][2] - diff_result[1][1]
            b_span = diff_result[1][4] - diff_result[1][3]

            # 标签少一个字
            if item1 == "equal" and item2 == "delete" and item3 == "equal" and a_span == 1 and b_span == 0:
                flag = True
                check_type = HtmlTableCompareType.SAME_LABEL_MISSING_ONE_CHARACTER

            # 标签一个字乱码
            if item1 == "equal" and item2 == "replace" and item3 == "equal" and a_span == 1 and b_span == 1:
                flag = True
                check_type = HtmlTableCompareType.SAME_LABEL_MISSING_ONE_CHARACTER

        return flag, check_type

    def compare_text_diff(self, src_html, dest_html, limit_len=10):
        """
        判断文字相似性

        :param src_html:
        :param dest_html:
        :param limit_len:
        :return:
        """
        src_content = self.norm_html_remove_tag_and_space(src_html)
        dest_content = self.norm_html_remove_tag_and_space(dest_html)

        # 计算不同地方
        diff_content = CommonUtils.calc_pair_sentences_diff(input_text1=src_content,
                                                            input_text2=dest_content)

        src_len = len(src_content)
        dest_len = len(dest_content)

        diff_len = sum(diff_content.values())
        diff_char = len(diff_content.keys())

        src_diff_radio = diff_len / src_len
        dest_diff_radio = diff_len / dest_len

        flag = False
        if src_len >= limit_len and dest_len >= limit_len \
                and abs(src_len - dest_len) < 10 \
                and diff_len < 50 \
                and diff_char < 10:
            flag = True

        logger.info(
            f"判断两段文本是否相似：flag: {flag} - src_len: {src_len} dest_len: {dest_len}  diff_len: {diff_len} "
            f" diff_char: {diff_char}  src_diff_radio: {src_diff_radio:.3f}  dest_diff_radio: {dest_diff_radio:.3f}")

        check_metric = {
            "src_content": src_content,
            "dest_content": dest_content,
            "diff_content": diff_content,
            "src_len": src_len,
            "dest_len": dest_len,
            "diff_len": diff_len,
            "diff_char": diff_char,
            "src_diff_radio": src_diff_radio,
            "dest_diff_radio": dest_diff_radio,
            "flag": flag,
        }

        return flag, check_metric

    def run(self):
        pass

    def check_table_diff_error(self, pred_html, label_html):
        """
        判断两段html 表格内容

        :param pred_html:
        :param label_html:
        :return:
        """
        pred_html_clean = self.norm_html_text(pred_html)
        label_html_clean = self.norm_html_text(label_html)
        pred_table_structure, pred_structure_cells = self.extract_table_structure(pred_html)
        label_table_structure, label_structure_cells = self.extract_table_structure(label_html)

        pred_table_cells = self.extract_table_cell(pred_html_clean)
        label_table_cells = self.extract_table_cell(label_html_clean)

        structure_flag = False
        if pred_table_structure == label_table_structure:
            logger.info(f"表格结构相同")
            structure_flag = True
        else:
            logger.info(f"pred_table_structure : {pred_table_structure}")
            logger.info(f"label_table_structure: {label_table_structure}")
            # return structure_flag, []

        diff_cells = self.get_table_text_cell_diff(pred_table_cells, label_table_cells)
        diff_structures = self.get_table_structure_cell_diff(pred_structure_cells, label_structure_cells)

        return structure_flag, diff_cells, diff_structures,pred_structure_cells,label_structure_cells

    def get_table_text_cell_diff(self, pred_table_cells, label_table_cells):
        """
        判断text cell diff

        :param pred_table_cells:
        :param label_table_cells:
        :return:
        """
        diff_cells = []
        # 结构一致，判断内容是否一致
        for row_index, (pred_rows, label_rows) in enumerate(zip(pred_table_cells, label_table_cells)):
            for column_index, (pred_column, label_column) in enumerate(zip(pred_rows, label_rows)):
                if pred_column == label_column:
                    continue
                # 计算不同地方
                diff_content = CommonUtils.calc_pair_sentences_diff(input_text1=pred_column,
                                                                    input_text2=label_column)

                src_len = len(pred_column)
                dest_len = len(label_column)

                diff_len = sum(diff_content.values())
                diff_char = len(diff_content.keys())

                if diff_len == 0:
                    compare_type = HtmlTableCompareType.DIFF_TEXT_ORDER
                else:
                    if src_len > dest_len:
                        compare_type = HtmlTableCompareType.DIFF_TEXT_LABEL_LESS_WORDS
                    elif src_len == dest_len:
                        compare_type = HtmlTableCompareType.DIFF_TEXT_INCONSISTENT
                    else:
                        compare_type = HtmlTableCompareType.DIFF_TEXT_PREDICT_LESS_WORDS

                diff_item = {
                    "compare_type": compare_type.desc,
                    "check_type": compare_type.get_check_type(),
                    "row_index": row_index + 1,
                    "column_index": column_index + 1,
                    "pred_text": pred_column,
                    "label_text": label_column,
                    "pred_len": src_len,
                    "label_len": dest_len,
                    "diff_len": diff_len,
                    "diff_char": diff_char,
                    "diff_content": dict(diff_content),
                }

                diff_cells.append(diff_item)
        return diff_cells

    def get_table_structure_cell_diff(self, pred_structure_cells, label_structure_cells):
        """
        判断table structure cell diff

        :param pred_structure_cells:
        :param label_structure_cells:
        :return:
        """
        diff_cells = []
        pred_row_total = len(pred_structure_cells)
        label_row_total = len(label_structure_cells)

        pred_cell_total = sum([len(item) for item in pred_structure_cells])
        label_cell_total = sum([len(item) for item in label_structure_cells])

        diff_item = {
            "pred_row_total": pred_row_total,
            "label_row_total": label_row_total,
            "diff_row_total": pred_row_total - label_row_total,
            "pred_cell_total": pred_cell_total,
            "label_cell_total": label_cell_total,
            "diff_cell_total": pred_cell_total - label_cell_total,
        }
        if pred_row_total != label_row_total:
            compare_type = HtmlTableCompareType.DIFF_CELL_DIFF_ROW
            diff_item["compare_type"] = compare_type.desc

            diff_cells.append(diff_item)
            return diff_cells
        # 结构一致，判断内容是否一致
        for row_index, (pred_rows, label_rows) in enumerate(zip(pred_structure_cells, label_structure_cells)):
            max_row = max(len(pred_rows), len(label_rows))
            for column_index, idx in enumerate(range(0,max_row)):
                pred_column = pred_rows[column_index] if column_index < len(pred_rows) else None
                label_column = label_rows[column_index] if column_index < len(label_rows) else None

                if pred_column == label_column or label_column is None or pred_column is None:
                    continue
                # 计算不同地方
                diff_span_dict = CommonUtils.calc_pair_structure_diff(input_cell1=pred_column,
                                                                      input_cell2=label_column)
                diff_row = diff_span_dict["diff_row"]
                diff_col = diff_span_dict["diff_col"]
                if diff_row == 0 and diff_col != 0:
                    compare_type = HtmlTableCompareType.DIFF_CELL_COL_SPAN
                elif diff_row != 0 and diff_col == 0:
                    compare_type = HtmlTableCompareType.DIFF_CELL_ROW_SPAN
                elif diff_row == 0 and diff_col == 0:
                    compare_type = HtmlTableCompareType.DIFF_CELL_SPAN_SAME
                else:
                    compare_type = HtmlTableCompareType.DIFF_CELL_ROW_COL_SPAN

                diff_item = {
                    "compare_type": compare_type.desc,
                    "check_type": compare_type.get_check_type(),
                    "row_index": row_index + 1,
                    "column_index": column_index + 1,
                    "pred_column": pred_column,
                    "label_column": label_column,
                    "diff_row": diff_row,
                    "diff_col": diff_col,
                    "diff_content": diff_span_dict,
                }

                diff_cells.append(diff_item)

        if len(diff_cells) ==0 and pred_cell_total != label_cell_total:
            compare_type = HtmlTableCompareType.DIFF_CELL_ROW_COL_SPAN
            diff_item["compare_type"] = compare_type.desc

            diff_cells.append(diff_item)

        return diff_cells

    def extract_table_structure(self, table_html):
        """
        提取表格的结构：移除文字

        :param table_html:
        :return:
        """

        # table_html_clean = MatchUtils.clean_html_table_width(table_html)
        # html_clean = MatchUtils.remove_html_table_text(table_html_clean)
        #

        table_html = table_html.replace("'", '"')
        row_list = table_html.split('</tr>')

        tabel_rows = []
        tabel_cells = []
        for idx, row in enumerate(row_list):
            columns = row.split('</td>')

            column_list = []
            clean_column_list = []
            for column in columns:
                if "</table>" in column:
                    column_list.append(column)
                    continue
                index = -1
                skip = 2
                if column.count('<td>') > 0:
                    index = column.rindex('<td>')
                    skip = 4
                elif column.count('">') > 0:
                    index = column.rindex('">')
                elif column.count("'>") > 0:
                    index = column.rindex("'>")
                span = f"{column[:index + skip]}</td>"
                column_list.append(span)

                new_span = MatchUtils.clean_html_table_width(span)
                new_span = CommonUtils.clean_sentence_remove_space(new_span)
                if len(new_span) > 5:
                    # tr_index = new_span.index('<tr>') if "<tr>" in new_span else 0
                    # new_span = new_span[tr_index+4:]
                    td_index = new_span.index('<td>') if "<td>" in new_span else 0
                    new_span_clean = new_span[td_index:]
                    clean_column_list.append(new_span_clean)

            one_row = f"{''.join(column_list)}</tr>" if idx + 1 != len(row_list) else ''.join(column_list)
            tabel_rows.append(one_row)
            if len(clean_column_list) > 0:
                tabel_cells.append(clean_column_list)

        table_structure = "".join(tabel_rows)

        pred_html_clean = self.norm_html_text(table_structure)
        table_html_clean = MatchUtils.clean_html_table_width(pred_html_clean)
        clean_table_content = CommonUtils.clean_sentence_remove_space(table_html_clean)
        clean_table_content = clean_table_content.replace('class="pdf-ocr-table"', "")
        return clean_table_content, tabel_cells

    def extract_table_cell(self, table_html):
        """
        提取表格的cell

        :param table_html:
        :return:
        """
        table_html_clean = MatchUtils.clean_html_table_width(table_html)
        table_html_clean = re.sub(r"<tr\s*>", "<tr>", table_html_clean)
        table_html_clean = re.sub(r"<td\s*>", "<td>", table_html_clean)
        row_list = table_html_clean.split('<tr>')

        tabel_cells = []
        for row in row_list:
            if str(row).find("<table") > -1:
                continue
            columns = row.split('<td>')

            column_list = []
            for column in columns:
                if not str(column).rfind("</td>") > -1:
                    continue
                column_clean = column.replace("</td>", "").replace("</tr>", "").replace("</table>", "")
                column_clean = CommonUtils.clean_sentence_remove_space(column_clean)
                column_list.append(column_clean)
            tabel_cells.append(column_list)

        return tabel_cells


def main():
    runner = TableResultCompare()
    runner.run()


if __name__ == '__main__':
    main()
