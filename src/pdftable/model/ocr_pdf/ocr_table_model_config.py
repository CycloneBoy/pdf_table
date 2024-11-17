#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：ocr_table_model_config
# @Author  ：cycloneboy
# @Date    ：20xx/9/13 10:21

"""
表格识别模型配置
"""

__all__ = [
    'TABLE_MODEL_DICT',
]

TABLE_MODEL_DICT = {
    "model_scope": {
        "detection": {
            "resnet18": {
                "general": {
                    "backbone": "resnet18",
                    "model": 'damo/cv_resnet18_ocr-detection-db-line-level_damo',
                    "hf_model": 'cycloneboy/cv_resnet18_ocr-detection-db-line-level_damo',
                    "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg'
                },
            },
            "proxylessnas": {
                "general": {
                    "backbone": "proxylessnas",
                    "model": "damo/cv_proxylessnas_ocr-detection-db-line-level_damo",
                    "hf_model": "cycloneboy/cv_proxylessnas_ocr-detection-db-line-level_damo",
                    "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg'
                },
            },
        },
        "recognition": {
            "CRNN": {
                "general": {
                    "recognizer": "CRNN",
                    "model": 'damo/cv_crnn_ocr-recognition-general_damo',
                    "hf_model": 'cycloneboy/cv_crnn_ocr-recognition-general_damo',
                    "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
                },
            },
            "LightweightEdge": {
                "general": {
                    "recognizer": "LightweightEdge",
                    "model": 'damo/cv_LightweightEdge_ocr-recognitoin-general_damo',
                    "hf_model": 'cycloneboy/cv_LightweightEdge_ocr-recognitoin-general_damo',
                    "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
                },
            },
            "ConvNextViT": {
                "general": {
                    "recognizer": "ConvNextViT",
                    "model": 'cycloneboy/cv_convnextTiny_ocr-recognition-general_damo',
                    "hf_model": 'cycloneboy/cv_convnextTiny_ocr-recognition-general_damo',
                    "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
                },
                "handwritten": {
                    "recognizer": "ConvNextViT",
                    "model": 'cycloneboy/cv_convnextTiny_ocr-recognition-handwritten_damo',
                    "hf_model": 'cycloneboy/cv_convnextTiny_ocr-recognition-handwritten_damo',
                    "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_handwritten.jpg'
                },
                "document": {
                    "recognizer": "ConvNextViT",
                    "model": 'cycloneboy/cv_convnextTiny_ocr-recognition-document_damo',
                    "hf_model": 'cycloneboy/cv_convnextTiny_ocr-recognition-document_damo',
                    "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_document.png'
                },
                "licenseplate": {
                    "recognizer": "ConvNextViT",
                    "model": 'cycloneboy/cv_convnextTiny_ocr-recognition-licenseplate_damo',
                    "hf_model": 'cycloneboy/cv_convnextTiny_ocr-recognition-licenseplate_damo',
                    "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_licenseplate//ocr_recognition_licenseplate.jpg'
                },
                "scene": {
                    "recognizer": "ConvNextViT",
                    "model": 'cycloneboy/cv_convnextTiny_ocr-recognition-scene_damo',
                    "hf_model": 'cycloneboy/cv_convnextTiny_ocr-recognition-scene_damo',
                    "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
                },
            },
        },
        "table_structure": {
            "CenterNet": {
                "wtw": {
                    "backbone": "dla34",
                    "model": 'iic/cv_dla34_table-structure-recognition_cycle-centernet',
                    "hf_model": 'cycloneboy/cv_dla34_table-structure-recognition_cycle-centernet',
                    "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/table_recognition.jpg'
                },
            },
            "Lore": {
                "wireless": {
                    "backbone": "ResNet-18",
                    "model": 'cycloneboy/cv_resnet-transformer_table-structure-recognition_lore_wireless',
                    "model_hub": 'damo/cv_resnet-transformer_table-structure-recognition_lore',
                    "model_hub2": 'iic/cv_resnet-transformer_table-structure-recognition_lore',
                    "model_url": 'https://drive.google.com/file/d/1cBaewRwlZF1tIZovT49HpJZ5wlb3nSCw/view?usp=sharing',
                    "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/lineless_table_recognition.jpg'
                },
                "wtw": {
                    "backbone": "DLA-34",
                    "model": 'cycloneboy/cv_dla34-transformer_table-structure-recognition_lore_wtw',
                    "model_url": 'https://drive.google.com/file/d/1n33c9jmGmjSfRbheleE1pqiIXBb_BCEw/view?usp=sharing',
                    "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/lineless_table_recognition.jpg'
                },
                "ptn": {
                    "backbone": "DLA-34",
                    "model": 'cycloneboy/cv_dla34-transformer_table-structure-recognition_lore_ptn',
                    "model_url": 'https://drive.google.com/file/d/1hg5R42u_6xaoO-6Ft18Ctu86HB_N2Bzu/view?usp=sharing',
                    "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/lineless_table_recognition.jpg'
                },
                "PubTabNet": {
                    "backbone": "DLA-34",
                    "model": 'cycloneboy/cv_dla34-transformer_table-structure-recognition_lore_ptn',
                    "model_url": 'https://drive.google.com/file/d/1hg5R42u_6xaoO-6Ft18Ctu86HB_N2Bzu/view?usp=sharing',
                    "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/lineless_table_recognition.jpg'
                },
            },
        },
        "layout": {
            "DocXLayout": {
                "general": {
                    "backbone": "dla34",
                    "model": 'cycloneboy/cv_dla34_layout-analysis_docxlayout_general',
                    "image_url": 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/table_recognition.jpg'
                },
            }
        }
    },
    "PaddleOCR": {
        "detection": {
            "PP-OCRv4": {
                "ch": {
                    "model": "cycloneboy/ch_PP-OCRv4_det_infer",
                    "server_model": "cycloneboy/ch_PP-OCRv4_det_server_infer",
                },
                "en": {
                    "model": "cycloneboy/en_PP-OCRv3_det_infer",
                },
                "ml": {
                    "model": "cycloneboy/Multilingual_PP-OCRv3_det_infer",
                },
            },
            "PP-OCRv3": {
                "ch": {
                    "model": "cycloneboy/ch_PP-OCRv3_det_infer",
                },
                "en": {
                    "model": "cycloneboy/en_PP-OCRv3_det_infer",
                },
                "ml": {
                    "model": "cycloneboy/Multilingual_PP-OCRv3_det_infer",
                },
            },
            "PP-Table": {
                "en": {
                    "model": "cycloneboy/en_ppocr_mobile_v2.0_table_det_infer",
                },
            }
        },
        "recognition": {
            "PP-OCRv4": {
                "ch": {
                    "model": "cycloneboy/ch_PP-OCRv4_rec_infer",
                    "server_model": "cycloneboy/ch_PP-OCRv4_rec_server_infer",
                },
                "en": {
                    "model": "cycloneboy/en_PP-OCRv4_rec_infer",
                },
                "korean": {
                    "model": "cycloneboy/korean_PP-OCRv4_rec_infer",
                },
                "japan": {
                    "model": "cycloneboy/japan_PP-OCRv4_rec_infer",
                },
                "chinese_cht": {
                    "model": "cycloneboy/chinese_cht_PP-OCRv3_rec_infer",
                },
                "ta": {
                    "model": "cycloneboy/ta_PP-OCRv4_rec_infer",
                },
                "te": {
                    "model": "cycloneboy/te_PP-OCRv4_rec_infer",
                },
                "ka": {
                    "model": "cycloneboy/ka_PP-OCRv4_rec_infer",
                },
                "latin": {
                    "model": "cycloneboy/latin_PP-OCRv3_rec_infer",
                },
                "arabic": {
                    "model": "cycloneboy/arabic_PP-OCRv4_rec_infer",
                },
                "cyrillic": {
                    "model": "cycloneboy/cyrillic_PP-OCRv3_rec_infer",
                },
                "devanagari": {
                    "model": "cycloneboy/devanagari_PP-OCRv4_rec_infer",
                },
            },
            "PP-OCRv3": {
                "ch": {
                    "model": "cycloneboy/ch_PP-OCRv3_rec_infer",
                },
                "en": {
                    "model": "cycloneboy/en_PP-OCRv3_rec_infer",
                },
                "korean": {
                    "model": "cycloneboy/korean_PP-OCRv3_rec_infer",
                },
                "japan": {
                    "model": "cycloneboy/japan_PP-OCRv3_rec_infer",
                },
                "chinese_cht": {
                    "model": "cycloneboy/chinese_cht_PP-OCRv3_rec_infer",
                },
                "ta": {
                    "model": "cycloneboy/ta_PP-OCRv3_rec_infer",
                },
                "te": {
                    "model": "cycloneboy/te_PP-OCRv3_rec_infer",
                },
                "ka": {
                    "model": "cycloneboy/ka_PP-OCRv3_rec_infer",
                },
                "latin": {
                    "model": "cycloneboy/latin_PP-OCRv3_rec_infer",
                },
                "arabic": {
                    "model": "cycloneboy/arabic_PP-OCRv3_rec_infer",
                },
                "cyrillic": {
                    "model": "cycloneboy/cyrillic_PP-OCRv3_rec_infer",
                },
                "devanagari": {
                    "model": "cycloneboy/devanagari_PP-OCRv3_rec_infer",
                },
            },
            "PP-Table": {
                "en": {
                    "model": "cycloneboy/en_ppocr_mobile_v2.0_table_rec_infer",
                },
            }
        },
        "table_structure": {
            "SLANet": {
                "ch": {
                    "model": 'cycloneboy/ch_ppstructure_mobile_v2.0_SLANet_infer',
                },
                "en": {
                    "model": 'cycloneboy/en_ppstructure_mobile_v2.0_SLANet_infer',
                },
            }
        },
        "cls_image": {
            "PPLCNet": {
                "table_attribute": {
                    "model": 'cycloneboy/cv_cls_pulc_table_attribute',
                },
                "text_image_orientation": {
                    "model": 'cycloneboy/cv_cls_pulc_text_image_orientation',
                },
                "textline_orientation": {
                    "model": 'cycloneboy/cv_cls_pulc_textline_orientation',
                },
                "language_classification": {
                    "model": 'cycloneboy/cv_cls_pulc_language_classification',
                },
            }
        },
        "layout": {
            "LCNet": {
                "ch": {
                    "model": 'cycloneboy/picodet_lcnet_x1_0_fgd_layout_cdla_infer',
                },
                "en": {
                    "model": 'cycloneboy/picodet_lcnet_x1_0_fgd_layout_infer',
                },
                "table": {
                    "model": 'cycloneboy/picodet_lcnet_x1_0_fgd_layout_table_infer',
                },
            }

        },
        "formula": {
            "latex": {
                "en": {
                    "model": "cycloneboy/rec_latex_ocr_infer",
                },
                "ch": {
                    "model": "cycloneboy/rec_latex_ocr_infer",
                },
            }

        },
    },
    "Other": {
        "table_structure": {
            "Lgpma": {
                "PubTabNet": {
                    "backbone": "ResNet",
                    "model": 'cycloneboy/en_table_structure_lgpma_pubtabnet',
                },
            },
            "MtlTabNet": {
                "PubTabNet": {
                    "backbone": "TableResNetExtra",
                    "model": 'cycloneboy/en_table_structure_mtltabnet_pubtabnet',
                },
                "FinTabNet": {
                    "backbone": "TableResNetExtra",
                    "model": 'cycloneboy/en_table_structure_mtltabnet_fintabnet',
                },
            },
            "TableMaster": {
                "PubTabNet": {
                    "backbone": "TableResNetExtra",
                    "model": 'cycloneboy/en_table_structure_tablemaster_pubtabnet',
                },
            },
            "LineCell": {
                "PubTabNet": {
                    "backbone": "opencv",
                    "model": 'cycloneboy/line_cell',
                },
            },
            "LineCellPdf": {
                "PubTabNet": {
                    "backbone": "digital_pdf",
                    "model": 'cycloneboy/line_cell_pdf',
                },
            },
        },
        "config": {
            "Pdftable": {
                "font": {
                    "model": "cycloneboy/pdftable_config"
                }
            }

        }
    }
}
