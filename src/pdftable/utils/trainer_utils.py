#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：trainer_utils
# @Author  ：cycloneboy
# @Date    ：20xx/11/6 18:59
from typing import List

from transformers import Trainer, PreTrainedTokenizer, TrainingArguments

from pdftable.entity.common_entity import ModelArguments, DataTrainingArguments
from pdftable.utils import BaseUtil, FileUtils, Constants, TimeUtils, logger, CommonUtils


class TrainerUtils(BaseUtil):
    """
    文件工具类
    """

    def init(self):
        pass


    @staticmethod
    def init_wandb(project_name):
        """
        初始化wandb

        :param project_name:
        :return:
        """
        import wandb
        wandb_run = wandb.init(project=project_name, entity=FileUtils.read_wandb_username(),
                               dir=Constants.WANDB_LOG_DIR)
        wandb_run_id = wandb.run.id
        return wandb_run_id

    @staticmethod
    def modify_wandb_run_name(model_args: ModelArguments,other_str=""):
        import wandb
        run_time = model_args.run_time if model_args.run_time is not None else TimeUtils.now_str_short()
        new_run_name = f"{model_args.model_name}_{model_args.task_name}{other_str}_{run_time}"
        wandb.run.name = new_run_name
        wandb.run.save()
        model_args.wandb_run_name = new_run_name
        logger.info(f"wandb run_name:{new_run_name}")
        return new_run_name


    @staticmethod
    def train_and_eval_model(trainer: Trainer, tokenizer: PreTrainedTokenizer, model_args: ModelArguments,
                             data_args: DataTrainingArguments, training_args: TrainingArguments,
                             test_dataset, ):
        """
        训练模型 ，评估模型

        :param trainer:
        :param tokenizer:
        :param model_args:
        :param data_args:
        :param training_args:
        :param test_dataset:
        :return:
        """
        # Training
        if training_args.do_train:
            train_begin_time = TimeUtils.now_str()
            logger.info(f"开始训练：{model_args.model_name} - {train_begin_time}")

            train_out = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            train_end_time = TimeUtils.now_str()
            logger.info(f"结束训练：{TimeUtils.calc_diff_time(train_begin_time, train_end_time)}")
            logger.info(f"训练指标： {train_out}")

            # 保存训练好的模型
            best_model_path = f"{training_args.output_dir}/best"
            trainer.save_model(best_model_path)
            logger.info(f"保存训练好的最好模型:{best_model_path}")

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            # if trainer.is_world_master():
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(best_model_path)
        # Evaluation
        results = {}
        if training_args.do_eval:
            eval_begin_time = TimeUtils.now_str()
            logger.info(f"开始进行评估，{eval_begin_time}")
            logger.info("*** Evaluate ***")

            eval_output = trainer.evaluate()
            eval_end_time = TimeUtils.now_str()
            logger.info(f"结束评估,耗时：{TimeUtils.calc_diff_time(eval_begin_time, eval_end_time)} s")
            logger.info(f"评估指标： {eval_output}")

            results["eval"] = eval_output
        # Predict
        if training_args.do_predict:
            logger.info(f"开始进行预测")
            trainer.predict(test_dataset=test_dataset)
            logger.info(f"完成预测")
        return results

    @staticmethod
    def freeze_model(model, unfreeze_layers: List, model_args: ModelArguments = None):

        """
        冻结部分层的参数

        :param model:
        :param unfreeze_layers:
        :param model_args:
        :return:
        """
        for name, param in model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        logger.info(f"----------------------------------模型参数汇总:--------------------------------")
        CommonUtils.calc_model_parameter_number(model)

        logger.info(f"----------------------------------需要优化的参数BEGIN:--------------------------------")
        # 验证一下
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"{name} - {param.size()} - {param.requires_grad}")

        logger.info(f"----------------------------------需要优化的参数END----------------------------------")

        logger.info(f"----------------------------------修改模型参数END----------------------------------")
