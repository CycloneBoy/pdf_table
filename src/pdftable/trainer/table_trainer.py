#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：table_trainer
# @Author  ：cycloneboy
# @Date    ：20xx/11/6 17:29

import dataclasses
import math
import os
import time
import traceback
from copy import deepcopy

import numpy as np
import torch

from torch import nn, optim
from torch.optim.lr_scheduler import SequentialLR, ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, EvalPrediction, \
    TrainerCallback, get_scheduler, SchedulerType

from typing import Optional, List, Union, Callable, Dict, Tuple, Any

from pdftable.entity.common_entity import DataTrainingArguments, ModelArguments
from pdftable.model import TableLorePostProcessor
from pdftable.trainer.basic_trainer import BaseTrainer
from pdftable.utils import logger
from pdftable.utils.model.model_utils import get_polynomial_constant_schedule_with_warmup


class TableTrainer(BaseTrainer):
    """
    Mrc trainer
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            data_args: DataTrainingArguments = None,
            model_args: ModelArguments = None,
            run_log=None,
    ):
        super().__init__(model=model, args=args, data_collator=data_collator,
                         train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer,
                         model_init=model_init, compute_metrics=compute_metrics, callbacks=callbacks,
                         optimizers=optimizers, preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                         data_args=data_args, model_args=model_args, run_log=run_log)

        self.current_best_metric = 0
        self.post_processor = TableLorePostProcessor(model.config,
                                                     output_dir=self.output_dir,
                                                     show_info=False
                                                     )

        use_compile = True

        if use_compile:
            # self.model.detect_infer_model = torch.compile(self.model.detect_infer_model)
            self.model.detect_infer_model.base = torch.compile(self.model.detect_infer_model.base)
            self.model.process_infer_model = torch.compile(self.model.process_infer_model)
            logger.info(f"使用torch compile 训练推理")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        start = time.time()
        if self.args.fp16_full_eval and inputs["pixel_values"].dtype != torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        if  "meta" not in inputs:
            inputs = {
                "pixel_values": inputs["pixel_values"],
                "labels": inputs
            }
        outputs = super().compute_loss(model=model, inputs=inputs, return_outputs=return_outputs)
        self.total_time += time.time() - start
        if return_outputs:
            loss, outputs = outputs
        else:
            loss = outputs

        if "meta" in inputs:
            image_name = self.eval_dataset.get_image_path_from_meta(inputs["meta"], )
            predict = self.post_processor(outputs, image_name=image_name)

        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            if self.model_args.lr_step is not None:
                self.lr_scheduler = get_polynomial_constant_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    total_epoch=self.args.num_train_epochs,
                    step_epoch=[int(i) for i in self.model_args.lr_step.split(',')],
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler
