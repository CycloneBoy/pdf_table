# pdf_table

## install
```shell
# install  ghostscript for pdf to image
apt install ghostscript

# install pdftable
#pip install pdftable

python setup.py install
```

## Usage

### env 

To download model from modelscope, please set the environment variable PDFTABLE_USE_MODELSCOPE_HUB to 1, otherwise huggingface will be used by default to download the model.

### cli 
```text
# pdftable --help

usage: pdftable [-h] --output_dir OUTPUT_DIR --file_path_or_url FILE_PATH_OR_URL [--lang LANG] [--debug [DEBUG]] [--pages PAGES]
                [--html_page_merge_sep HTML_PAGE_MERGE_SEP] [--detect_model DETECT_MODEL] [--detect_db_thresh DETECT_DB_THRESH]
                [--recognizer_model RECOGNIZER_MODEL] [--recognizer_task_type RECOGNIZER_TASK_TYPE]
                [--table_structure_model TABLE_STRUCTURE_MODEL] [--table_structure_task_type TABLE_STRUCTURE_TASK_TYPE]
                [--layout_model LAYOUT_MODEL]

options:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory (default: None)
  --file_path_or_url FILE_PATH_OR_URL, --file-path-or-url FILE_PATH_OR_URL
                        file path or url (default: None)
  --lang LANG           ocr recognition language (default: en)
  --debug [DEBUG]       debug mode (default: False)
  --pages PAGES         need process page. Comma-separated page numbers. Example: '1,3,4' or '1,4-end' or 'all'. (default: all)
  --html_page_merge_sep HTML_PAGE_MERGE_SEP, --html-page-merge-sep HTML_PAGE_MERGE_SEP
                        The delimiter that separates each page of PDF conversion results in the final converted html result page.
                        (default: @@@@@@)
  --detect_model DETECT_MODEL, --detect-model DETECT_MODEL
                        ocr detect model, optional items: PP-OCRv4, PP-OCRv3, resnet18, proxylessnas. (default: PP-OCRv4)
  --detect_db_thresh DETECT_DB_THRESH, --detect-db-thresh DETECT_DB_THRESH
                        db threshold (default: 0.2)
  --recognizer_model RECOGNIZER_MODEL, --recognizer-model RECOGNIZER_MODEL
                        ocr recognize model, optional items: PP-OCRv4, PP-OCRv3, PP-Table, ConvNextViT, CRNN, LightweightEdge (default:
                        PP-OCRv4)
  --recognizer_task_type RECOGNIZER_TASK_TYPE, --recognizer-task-type RECOGNIZER_TASK_TYPE
                        ocr recognizer task type, It only takes effect when recognizer_model is ConvNextViT, optional items: general,
                        handwritten, document, licenseplate, scene. (default: document)
  --table_structure_model TABLE_STRUCTURE_MODEL, --table-structure-model TABLE_STRUCTURE_MODEL
                        table structure model, optional items: CenterNet, SLANet, Lore, Lgpma, MtlTabNet, TableMaster, LineCell. (default:
                        Lore)
  --table_structure_task_type TABLE_STRUCTURE_TASK_TYPE, --table-structure-task-type TABLE_STRUCTURE_TASK_TYPE
                        table structure task type, optional items: ptn, wtw, wireless, fin. ptn represents the data set as PubTabNet. fin
                        represents FinTabNet, which is only valid when the table_structure_model is MtlTabNet. (default: wtw)
  --layout_model LAYOUT_MODEL, --layout-model LAYOUT_MODEL
                        layout model, optional items: picodet, DocXLayout (default: picodet)

```


## TODO
- [ ] Write project documentation
- [ ] Optimize project code
- [ ] Add the latest table recognition model
- [ ] other

## Thanks to the following projects
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [modelscope](https://github.com/modelscope/modelscope)
- [camelot](https://github.com/camelot-dev/camelot)

## References

If you use pdf_table in your projects, please consider citing the following:
```text
@article{sheng2024pdftable,
  title={PdfTable: A Unified Toolkit for Deep Learning-Based Table Extraction},
  author={Sheng, Lei and Xu, Shuai-Shuai},
  journal={arXiv preprint arXiv:2409.05125},
  url = {https://arxiv.org/abs/2409.05125},
  eprint = {2409.05125},
  doi = {10.48550/arXiv.2409.05125},
  year={2024}
}
```