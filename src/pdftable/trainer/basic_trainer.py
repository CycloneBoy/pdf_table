#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：basic_trainer
# @Author  ：cycloneboy
# @Date    ：20xx/11/6 17:46
import dataclasses
import math
import os
import time
import traceback
from copy import deepcopy

import numpy as np
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, EvalPrediction, \
    TrainerCallback, is_torch_tpu_available, is_datasets_available

from typing import Optional, List, Union, Callable, Dict, Tuple, Any

from transformers import Trainer

from pdftable.entity.common_entity import DataTrainingArguments, ModelArguments
from pdftable.utils import TimeUtils, FileUtils, logger, Constants


class BaseTrainer(Trainer):

    def __init__(
            self,
            model: Union["PreTrainedModel", nn.Module] = None,
            args: "TrainingArguments" = None,
            data_collator: Optional["DataCollator"] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
            compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
            callbacks: Optional[List["TrainerCallback"]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            data_args: DataTrainingArguments = None,
            model_args: ModelArguments = None,
            run_log=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.data_args = data_args
        self.model_args = model_args
        self.task_name = self.model_args.task_name
        self.show_info = self.model_args.show_info
        self.model_name = self.model_args.model_name
        self.run_log = run_log
        self.output_dir = self.args.output_dir

        self.eval_file_path = self.data_args.eval_data_file

        self.current_best_metric = 0 if self.args.greater_is_better else 1e9
        self.current_epoch_and_step = ""
        self.metric_show_file_name = f"{self.output_dir}/metric_show.txt"
        self.total_time = 0

        ####################################################################
        # 自己的训练函数
        #
        ####################################################################
        self.custom_train = False
        self.custom_eval = False

    def save_best_model(self, result, metric_key_prefix: str = "eval", ):
        """
        保存最好的模型

        :param result:
        :param metric_key_prefix:
        :return:
        """
        best_metric_name = f"{metric_key_prefix}_{self.args.metric_for_best_model}"
        best_metric = result[best_metric_name] if best_metric_name in result else None
        if best_metric is not None and self.state.epoch is not None:
            best_flag = best_metric > self.current_best_metric if self.args.greater_is_better else best_metric < self.current_best_metric
            if best_flag:
                model_state_dict = self.model.state_dict()
                save_dir = f"{self.output_dir}/best_model"
                save_best_model_path = f"{save_dir}/pytorch_model.bin"
                FileUtils.check_file_exists(save_best_model_path)
                torch.save(model_state_dict, save_best_model_path)

                # copy vocab and config file
                file_name_list = ["vocab.txt", "config.json", "special_tokens_map.json", "tokenizer_config.json"]
                for file_name in file_name_list:
                    file_path = f"{self.model_args.model_name_or_path}/{file_name}"
                    FileUtils.copy_file(file_path, save_dir)
                self.current_best_metric = best_metric
                self.current_epoch_and_step = self.get_current_epoch_and_step()
                logger.info(
                    f"保存最好的模型：{self.current_best_metric} - {self.current_epoch_and_step} - {save_best_model_path} ")

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_metrics = None
        eval_begin_time = TimeUtils.now_str()

        self.save_log(eval_begin_time)

        if self.custom_eval:
            eval_metrics = self.evaluate_custom(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                                metric_key_prefix=metric_key_prefix)
        else:
            # 默认的评估过程
            eval_metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                            metric_key_prefix=metric_key_prefix)

        # add save the eval metric to db
        eval_end_time = TimeUtils.now_str()

        all_metrics = {
            "eval_metrics": eval_metrics,
            "eval_begin_time": eval_begin_time,
            "eval_end_time": eval_end_time,
            "eval_use_time": TimeUtils.calc_diff_time(eval_begin_time, eval_end_time),
            "total": len(eval_dataset) if eval_dataset is not None else 0
        }

        self.save_log(eval_end_time)
        self.save_log(str(all_metrics))


        try:
            self.save_eval_predict_to_db(all_metrics)

            ######################################################################
            # 指标汇总
            self.save_best_model(eval_metrics)
        except Exception as e:
            logger.warning(f"处理异常：{e}")
            traceback.print_exc()

        return eval_metrics

    def evaluate_custom(self,
                        eval_dataset: Optional[Dataset] = None,
                        ignore_keys: Optional[List[str]] = None,
                        metric_key_prefix: str = "eval",
                        ) -> Dict[str, float]:
        pass

    def save_metric_to_file(self, all_metric, file_name):
        """
        保存评估指标到本地

        :param all_metric:
        :param file_name:
        :return:
        """
        # 添加其他的log
        run_log = []

        FileUtils.dump_json(file_name, all_metric, show_info=self.show_info)
        output_dir = os.path.join(self.output_dir, "eval_metric")
        add_suffix = self.get_current_epoch_and_step()
        eval_result_path = f"{output_dir}/eval_{add_suffix}_metric.json"
        FileUtils.dump_json(eval_result_path, all_metric)
        # logger.info(f"all_metric: {all_metric}")

        return run_log

    def get_result_metric_file_name(self, metric_key_prefix: str = "eval", ):
        """
        获取 预测的metric 的文件名称
        :param metric_key_prefix:
        :return:
        """

        run_result_metric_file = f"{Constants.HTML_BASE_DIR}/metric/{TimeUtils.get_time()}/{self.task_name}_{metric_key_prefix}_metric_{TimeUtils.now_str_short()}.json"
        return run_result_metric_file

    def get_copy_eval_file_to_separate_dir(self):
        """
        复制 评估过程文件到单独的文件夹

        :return:
        """
        current_epoch = self.state.epoch
        epoch = current_epoch if current_epoch is not None else 0

        add_suffix = self.get_current_epoch_and_step()
        output_dir = f"{self.output_dir}/eval_file/{int(epoch)}/{add_suffix}"

        if self.state.epoch is None and self.state.global_step == 0:
            output_dir = self.output_dir

        return output_dir

    def get_current_epoch_and_step(self):
        current_epoch = self.state.epoch
        epoch = current_epoch if current_epoch is not None else 0

        output_dir = f"{epoch:.2f}_{self.state.global_step}"
        return output_dir

    def save_log(self, msg_list, mode='a'):
        """
        保存日志到本地日志文件

        :param msg_list:
        :param mode:
        :return:
        """
        if isinstance(msg_list, str):
            msg_list = [msg_list, "\n"]
        FileUtils.save_to_text(self.metric_show_file_name, "\n".join(msg_list), mode=mode)

    def is_model_training(self):
        return self.model.training

    def save_eval_predict_to_db(self, all_metrics, metric_key_prefix: str = "eval", ):
        pass
