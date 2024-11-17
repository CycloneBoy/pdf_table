#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：common_entity
# @Author  ：cycloneboy
# @Date    ：20xx/10/13 16:05
import copy
import json
from dataclasses import field, dataclass
from typing import Optional

from pdftable.utils import Constants


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    show_info: bool = field(default=False, metadata={"help": "show_info"})

    ## 分类
    num_labels: int = field(default=2, metadata={"help": "需要预测的标签数量"})
    num_filters: int = field(default=256, metadata={"help": "num_filters"})
    filter_sizes: str = field(default="2,3,4", metadata={"help": "filter_sizes"})
    loss_weight: float = field(default=1.0, metadata={"help": "loss_weight"})
    font_dir: str = field(default="./", metadata={"help": "font_dir"})

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    model_type: Optional[str] = field(
        default="bert",
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(["bert"])},
    )

    model_name: str = field(default="bert_softmax", metadata={"help": "模型名称"})
    task_name: str = field(default="cluener", metadata={"help": "任务类型"})

    cache_dir: Optional[str] = field(
        default="data/", metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # csc_posted_type: str = field(default=None, metadata={"help": "纠错后处理类型"})
    wandb_run_id: str = field(default=None, metadata={"help": "wandb_run_id"})
    wandb_run_name: str = field(default=None, metadata={"help": "wandb_run_name"})
    not_save_metric: bool = field(default=False, metadata={"help": "not save metric to db"})
    # 推理参数
    inference_port: int = field(default=56101, metadata={"help": "inference_port"})
    inference_debug: bool = field(default=False, metadata={"help": "inference_debug"})

    use_args_eval_file: bool = field(default=False, metadata={"help": "是否采用命令行传递的参数文件"})
    run_type: str = field(default="成功", metadata={"help": "保存到数据库中的run_type：记录本次运行的类型"})
    flask_static_folder: str = field(default="./static", metadata={"help": "flask_static_folder"})
    save_index: bool = field(default=False, metadata={"help": "是否保存index"})
    faiss_fast: bool = field(default=False, metadata={"help": "是否使用faiss_fast"})
    rebuild_faiss_index: bool = field(default=False, metadata={"help": "是否重新构建索引faiss_fast"})
    modify_data_path: bool = field(default=True, metadata={"help": "是否修改data_path"})
    schema: str = field(default=None, metadata={"help": "uie schema"})

    redis_json_config_filename: str = field(default=None,
                                            metadata={"help": "redis json config file."})
    mysql_json_config_filename: str = field(default=None,
                                            metadata={"help": "mysql json config file."})
    sleep_second: int = field(default=10, metadata={"help": "sleep_second"})

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    run_log: str = field(default=None, metadata={"help": "run_log"})
    lr_step: str = field(default=None, metadata={"help": "lr_step"})
    run_time: str = field(default=None, metadata={"help": "run_time"})
    debug: bool = field(default=False, metadata={"help": "debug"})

    # model
    arch: str = field(default="dla_34", metadata={"help": "model architecture. Currently tested"
                                                          " res_18 | res_101 | resdcn_18 | resdcn_101 "
                                                          "| dlav0_34 | dla_34 | hourglass"})
    head_conv: int = field(default=-1, metadata={"help": 'conv layer channels for output head'
                                                         '0 for no conv layer'
                                                         '-1 for default setting: '
                                                         '64 for resnets and 256 for dla.'})

    down_ratio: int = field(default=4, metadata={"help": "output stride. Currently only supports 4."})
    load_mode: str = field(default="single", metadata={"help": "single or multiple"})
    wiz_2dpe: bool = field(default=False, metadata={"help": "whether to use the 2-dimensional position embeddings."})
    wiz_4ps: bool = field(default=False, metadata={"help": "whether to use the feature of the four corner points."})
    wiz_stacking: bool = field(default=False,
                               metadata={"help": "whether to use the cascading regressor or the vinilla regressor."})

    # table lora loss
    mse_loss: bool = field(default=False, metadata={"help": "use mse loss or focal loss to train keypoint heatmaps."})
    reg_loss: str = field(default="l1", metadata={"help": "regression loss: sl1 | l1 | l2"})
    hm_weight: float = field(default=1, metadata={"help": "loss weight for keypoint heatmaps."})
    mk_weight: float = field(default=1, metadata={"help": "loss weight for corner keypoint heatmaps."})
    off_weight: float = field(default=1, metadata={"help": "loss weight for keypoint local offsets."})
    wh_weight: float = field(default=1, metadata={"help": "loss weight for bounding box size."})
    st_weight: float = field(default=1, metadata={"help": "loss weight for cell coor reg."})
    norm_wh: bool = field(default=False, metadata={"help": "L1(\hat(y) / y, 1) or L1(\hat(y), y)"})
    dense_wh: bool = field(default=False, metadata={"help": "apply weighted regression near center "
                                                            "or just apply regression on center point."})
    cat_spec_wh: bool = field(default=False, metadata={"help": "category specific bounding box size."})
    reg_offset: bool = field(default=True, metadata={"help": "use regress local offset."})

    # ground truth validation
    eval_oracle_hm: bool = field(default=False, metadata={"help": "use ground center heatmap."})
    eval_oracle_mk: bool = field(default=False, metadata={"help": "use ground corner heatmap."})
    eval_oracle_wh: bool = field(default=False, metadata={"help": "use ground truth bounding box size."})
    eval_oracle_offset: bool = field(default=False, metadata={"help": "use ground truth local heatmap offset."})
    eval_oracle_kps: bool = field(default=False, metadata={"help": "use ground truth human pose offset."})
    eval_oracle_hmhp: bool = field(default=False, metadata={"help": "use ground truth human pose offset."})
    eval_oracle_hp_offset: bool = field(default=False, metadata={"help": "use ground truth human joint local offset."})
    eval_oracle_dep: bool = field(default=False, metadata={"help": "use ground truth depth."})
    num_stacks: int = field(default=1, metadata={"help": "num_stacks = 2 if opt.arch == 'hourglass' else 1"})

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)

        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(default="cls", metadata={"help": "数据集类型"})

    train_data_file: Optional[str] = field(
        default="/train.json",
        metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default="/dev.json",
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    test_data_file: Optional[str] = field(
        default="/test.json",
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    eval_label_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    vocab_file: Optional[str] = field(
        default="/vocab.pkl",
        metadata={
            "help": "vocab_file ."},
    )

    block_size: int = field(
        default=128,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )

    target_block_size: int = field(
        default=192,
        metadata={
            "help": "Optional output sequence length for generate."
        },
    )

    max_prefix_length: int = field(
        default=None,
        metadata={
            "help": "Optional output sequence length for generate."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    output_file: Optional[str] = field(
        default=Constants.SRC_HOME_DIR + "/outputs",
        metadata={"help": "The output data file."},
    )

    batch_size: int = field(
        default=24,
        metadata={"help": "batch size for train"},
    )

    target_size: int = field(
        default=1024,
        metadata={"help": "The maximum 2d pos size"},
    )
    use_segment_box: bool = field(
        default=False,
        metadata={"help": "Whether use segment box"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    lang: Optional[str] = field(
        default="en",
        metadata={"help": "Languge type of the dataset"},
    )


@dataclass
class PdfTableCliArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    output_dir: str = field(
        metadata={"help": "The output directory"},
    )

    file_path_or_url: str= field(
        metadata={"help": "file path or url"},
    )

    lang: Optional[str] = field(
        default="en",
        metadata={"help": "ocr recognition language"},
    )

    debug: bool = field(default=False, metadata={"help": "debug mode"})

    pages: Optional[str] = field(
        default="all",
        metadata={"help": "need process page.  Comma-separated page numbers. Example: '1,3,4' or '1,4-end' or 'all'."},
    )

    html_page_merge_sep: Optional[str] = field(
        default="@@@@@@",
        metadata={"help": "The delimiter that separates each page of PDF conversion results in the final converted html result page."},
    )

    detect_model: Optional[str] = field(
        default="PP-OCRv4",
        metadata={"help": "ocr detect model, optional items: PP-OCRv4, PP-OCRv3, resnet18, proxylessnas."},
    )

    detect_db_thresh: float = field(default=0.2, metadata={"help": "db threshold"})

    recognizer_model: Optional[str] = field(
        default="PP-OCRv4",
        metadata={"help": "ocr recognize model, optional items: PP-OCRv4, PP-OCRv3, PP-Table, ConvNextViT, CRNN, LightweightEdge"},
    )

    recognizer_task_type: Optional[str] = field(
        default="document",
        metadata={"help": "ocr recognizer task type, It only takes effect when recognizer_model is ConvNextViT, optional items: general, handwritten, document, licenseplate, scene."},
    )

    table_structure_model: Optional[str] = field(
        default="Lore",
        metadata={"help": "table structure model, optional items: CenterNet, SLANet, Lore, Lgpma, MtlTabNet, TableMaster, LineCell."},
    )

    table_structure_task_type: Optional[str] = field(
        default="wtw",
        metadata={"help": "table structure task type, optional items: ptn, wtw, wireless, fin. "
                          "ptn represents the data set as PubTabNet. fin represents FinTabNet, which is only valid when the table_structure_model is MtlTabNet."},
    )

    layout_model: Optional[str] = field(
        default="picodet",
        metadata={"help": "layout model, optional items: picodet, DocXLayout"},
    )





