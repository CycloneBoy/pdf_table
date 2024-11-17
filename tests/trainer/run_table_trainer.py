#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_tabl_trainer
# @Author  ：cycloneboy
# @Date    ：20xx/11/6 18:30
import os

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from pdftable.dataset.table.wtw_dataset import WtwDataset
from pdftable.entity.common_entity import ModelArguments, DataTrainingArguments
from pdftable.eval.table_metric import TableWtwComputeMetric
from pdftable.model import LoreModel, TableLorePostProcessor
from pdftable.trainer.table_trainer import TableTrainer
from pdftable.utils import logger, Constants, TimeUtils, CommonUtils
from pdftable.utils.trainer_utils import TrainerUtils


class RunTableTrainer(object):

    def __init__(self):
        self.wtw_dir = f"{Constants.DATA_DIR}/table/wtw"
        self.base_dir = f"{self.wtw_dir}/test"
        self.base_dir2 = f"{self.wtw_dir}/WTW-labels"
        self.run_time = TimeUtils.now_str_short()
        # self.run_time = "20231108_114846"
        self.output_base_dir = f"{Constants.HTML_BASE_DIR}/model_trainer/table/{TimeUtils.get_time()}"
        self.output_dir = f"{self.output_base_dir}/{self.run_time}"
        self.device = CommonUtils.get_torch_device()

    def build_dataset(self, split='train', batch_size=4, txt_file=None, base_dir=None):
        if base_dir is None:
            base_dir = f"{self.wtw_dir}/{split}"

        image_path = os.path.join(base_dir, "images")
        label_path = os.path.join(base_dir, "json", f"{split}.json")
        # label_path = os.path.join(base_dir, "txt")

        if split != "train":
            batch_size = 1
        eval_dataset = WtwDataset(image_path, label_path=label_path, split=split, txt_file=txt_file)

        collate_fn = eval_dataset.get_collate_fn()
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 # collate_fn=collate_fn,
                                 drop_last=True
                                 )
        logger.info(f"{split} - total - {txt_file} : {len(eval_loader)}")
        return eval_dataset, eval_loader

    def run_batch(self):
        # eval_dataset, eval_loader = self.build_dataset()
        eval_dataset, eval_loader = self.build_dataset(split="train", batch_size=4, txt_file=None)

        logger.info(f"total: {len(eval_loader)}")
        batch = next(iter(eval_loader))

        logger.info(f"total: {len(eval_loader)}")

        logger.info(batch)

        config = eval_dataset.config
        model = LoreModel(config)
        model.eval()
        model.to(self.device)

        with torch.no_grad():
            run_batch = {
                "image": batch["image"].to(self.device),
                "meta": batch["meta"].to(self.device),
            }

            result = model(**run_batch)
        logger.info(f"result: {result}")

        post_processor = TableLorePostProcessor(config, output_dir=self.output_dir)

        predict = post_processor(result, image_name=batch["inputs"])
        logger.info(f"predict: {predict}")

    def run(self):
        # split = 'train'
        split = 'test'
        batch_size = 4
        dataloader_num_workers = 1
        do_train = True
        # do_train = False

        """
           427 curved.txt   5
          1525 extremeratio.txt  51
           670 Inclined.txt 12
           749 muticolorandgrid.txt 3.50
            72 occblu.txt   9
            75 overlaid.txt 6
            93 simple.txt 2 
          3611 total
          
          2862 wtw_sub_simple 移除了muticolorandgrid  74 min

        """
        # name = "occblu"
        # name = "overlaid"
        name = "simple"
        # name = "curved"
        # name = "Inclined"
        # name = "muticolorandgrid"
        # name = "extremeratio"

        # name = "wtw_sub_simple"

        # txt_file = None
        # txt_file = 2370
        txt_file = f"{self.wtw_dir}/test/sub_classes/{name}.txt"

        if name in ["wtw_sub_simple"]:
            txt_file = f"{self.wtw_dir}/test/our_test/{name}.txt"

        # base_dir = f"{self.wtw_dir}/eval_demo/{name}"
        other_str = f"_{name}"
        if txt_file is None:
            other_str = ""

        if do_train:
            name = "valid"
            base_dir = f"{self.wtw_dir}/{name}"
            train_dataset, train_loader = self.build_dataset(split="train", batch_size=batch_size,
                                                             base_dir=base_dir,
                                                             txt_file=None)
        else:
            train_dataset = None
            train_loader = None

        name = "valid_test"
        base_dir = f"{self.wtw_dir}/{name}"
        txt_file = None
        # base_dir=None
        eval_dataset, eval_loader = self.build_dataset(split=split, batch_size=batch_size, txt_file=txt_file,
                                                       base_dir=base_dir)

        # one_item = next(iter(eval_loader))
        # logger.info(one_item)

        config = eval_dataset.config

        if do_train:
            config.pretrained = True
            dataloader_num_workers = 6
        model = LoreModel(config)

        wandb_run_id = TrainerUtils.init_wandb(project_name="tsr")

        model_args = ModelArguments(
            model_name="Lore",
            task_name="table",
            wandb_run_id=wandb_run_id,
            run_time=self.run_time,
            run_log=other_str,
            # lr_step="70,90"
            lr_step="30,40"
        )

        # 修改wandb run name
        TrainerUtils.modify_wandb_run_name(model_args=model_args, other_str=other_str)

        data_args = DataTrainingArguments(block_size=512, batch_size=1)
        training_args = TrainingArguments(output_dir=self.output_dir,
                                          no_cuda=False,
                                          # torch_compile=True,
                                          fp16=True,
                                          fp16_full_eval=True,
                                          half_precision_backend="apex",
                                          dataloader_num_workers=dataloader_num_workers,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=1,
                                          do_train=do_train,
                                          do_eval=True,
                                          remove_unused_columns=False,
                                          label_names=["meta"],
                                          run_name=model_args.wandb_run_name,
                                          adam_beta1=0.9,
                                          adam_beta2=0.98,
                                          adam_epsilon=1e-9,
                                          weight_decay=0.05,
                                          learning_rate=1e-4,
                                          lr_scheduler_type="linear",
                                          warmup_ratio=0.05,
                                          num_train_epochs=50,
                                          evaluation_strategy="epoch",
                                          # evaluation_strategy="steps",
                                          # eval_steps=100,
                                          logging_strategy="steps",
                                          logging_steps=500,
                                          save_strategy="epoch",
                                          save_total_limit=3,
                                          metric_for_best_model="acc"
                                          )

        compute_metrics = TableWtwComputeMetric(predict_dir=f"{training_args.output_dir}/{eval_dataset.task_type}",
                                                label_path=f"{base_dir}/txt",
                                                metric_dir=training_args.output_dir)

        trainer = TableTrainer(
            model=model,
            args=training_args,
            # data_collator=eval_dataset.get_collate_fn(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            model_args=model_args,
            data_args=data_args,
        )

        # 开始训练，评估模型
        results = TrainerUtils.train_and_eval_model(trainer=trainer,
                                                    tokenizer=None,
                                                    model_args=model_args,
                                                    data_args=data_args,
                                                    training_args=training_args,
                                                    test_dataset=eval_dataset)

    def compute_metric(self):
        output_dir = f"{self.output_base_dir}/20231108_103718"
        name = "simple"

        base_dir = f"{self.wtw_dir}/eval_demo/{name}"
        # image_path = os.path.join(base_dir, "images")
        # label_path = os.path.join(base_dir, "json", "test.json")
        # label_path = os.path.join(base_dir, "txt")
        label_path = f"{self.wtw_dir}/test/txt"

        compute_metrics = TableWtwComputeMetric(predict_dir=f"{output_dir}/wtw",
                                                label_path=label_path,
                                                metric_dir=output_dir)

        compute_metrics("demo")


def main():
    runner = RunTableTrainer()
    runner.run()
    # runner.run_batch()
    # runner.compute_metric()


if __name__ == '__main__':
    main()
