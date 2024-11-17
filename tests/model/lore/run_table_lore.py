#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_table_lore
# @Author  ：cycloneboy
# @Date    ：20xx/5/26 14:11
import os

import torch
from torch.utils.data import DataLoader
from torchvision.ops import deform_conv2d

from pdftable.dataset.table.wtw_dataset import WtwDataset, DataCollatorTableWtw
from pdftable.entity.common_entity import ModelArguments
from pdftable.model import TableLorePreProcessor, TableLorePostProcessor
from pdftable.model.lore import LoreModel, load_lore_model
from pdftable.model.lore.configuration_lore import LoreConfig
from pdftable.model.ocr_pdf import OcrTableStructureTask

from pdftable.utils import CommonUtils, Constants, logger, TimeUtils, FileUtils


class RunTableLore(object):

    def __init__(self):
        self.scope_model_base_dir = Constants.SCOPE_MODEL_BASE_DIR
        self.model_name = f"damo/cv_resnet-transformer_table-structure-recognition_lore"
        self.model_name_or_path = f"{self.scope_model_base_dir}/{self.model_name}"

        # self.fp16_full_eval = True
        self.fp16_full_eval = False
        self.device = CommonUtils.get_torch_device()
        self.infer_model = None
        self.predictor_type = "pytorch"
        self.output_dir = FileUtils.get_output_dir_with_time()

        self.img_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/lineless_table_recognition.jpg'

        self.K = 1000
        self.MK = 4000

    def run(self):
        pass

    def init_model(self, model_path, model):

        checkpoint = torch.load(model_path, map_location='cpu')
        load_lore_model(model.detect_infer_model, checkpoint, 'model')
        load_lore_model(model.process_infer_model, checkpoint, 'processor')

        logger.info(f'加载模型：{model_path}')

    def our_run_model(self, model_name_or_path=None, img_url=None):
        if model_name_or_path is None:
            model_name_or_path = self.model_name_or_path
        model = LoreModel(model_dir=self.model_name_or_path)

        # model_path = f"{self.model_name_or_path}/pytorch_model.pt"
        # self.init_model(model_path=model_path, model=model)

        save_dir = f"{Constants.DATA_DIR}/txt/model_scope/cv_resnet-transformer_table-structure-recognition_lore/lore"
        model_net, model_params = CommonUtils.print_model_param(model, show_info=False, save_dir=save_dir,
                                                                use_numpy=True)

        model.to(self.device)
        if self.fp16_full_eval:
            model.half()
        model.eval()

        if img_url is None:
            img_url = self.img_url

        raw_batch = self.pre_processor(img_url)
        one_batch = raw_batch[0]

        # batch = one_batch["img"]
        one_batch = {k: v.to(self.device) if k in ["img"] else v for k, v in one_batch.items()}

        if self.fp16_full_eval:
            one_batch = {k: v.half() if k in ["img"] else v for k, v in one_batch.items()}

        logger.info(f"one_batch: {one_batch}")

        with torch.no_grad():
            outputs = model(one_batch)

        print(f"outputs:{outputs}")

        # results = {'results': outputs, 'meta': one_batch['meta']}

        out_preds = self.pre_processor.postprocess(outputs)

        preds = out_preds
        print(f"preds:{preds}")

        return preds

    def dcn_demo(self):
        input = torch.rand(1, 512, 32, 32)
        kh, kw = 3, 3
        weight = torch.rand(256, 512, kh, kw)
        # offset and mask should have the same spatial size as the output
        # of the convolution. In this case, for an input of 10, stride of 1
        # and kernel size of 3, without padding, the output size is 8
        offset = torch.rand(1, 2 * kh * kw, 32, 32)
        mask = torch.rand(1, kh * kw, 32, 32)
        out = deform_conv2d(input, offset, weight, mask=mask)
        print(out.shape)
        # returns
        # torch.Size([4, 5, 8, 8])

    def run_ocr_table_structure_task(self):

        lang = "ch"
        # lang = "en"

        # task_type = "wireless"
        task_type = "ptn"
        # task_type = "wtw"

        # model = "CenterNet"
        # model = "SLANet"
        model = "Lore"

        # task = OcrTableStructureTask(predictor_type=self.predictor_type, )
        task = OcrTableStructureTask(model=model,
                                     predictor_type=self.predictor_type,
                                     output_dir=self.output_dir,
                                     lang=lang,
                                     task_type=task_type)

        # image = self.img_url
        if lang == "ch":
            image = f"{Constants.SRC_IMAGE_DIR}/layout_demo3.jpg"
        else:
            image = f"{Constants.SRC_IMAGE_DIR}/table_01.jpg"
        result = task(image)
        logger.info(f"result: {result}")

    def load_lore_dataset(self, split='train', batch_size=4):
        base_dir = f"{Constants.DATA_DIR}/table/wtw/test"
        base_dir2 = f"{Constants.DATA_DIR}/table/wtw/WTW-labels"
        image_path = os.path.join(base_dir, "images")
        label_path = os.path.join(base_dir, "json", "test.json")
        # label_path = os.path.join(base_dir2, "txt")

        if split != "train":
            batch_size = 1
        eval_dataset = WtwDataset(image_path, label_path=label_path, split=split)

        collate_fn = DataCollatorTableWtw()
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 # collate_fn=collate_fn
                                 )

        logger.info(f"total : {len(eval_loader)}")

        one_item = next(iter(eval_loader))

        # logger.info(one_item)

        return eval_dataset, eval_loader

    def test_loss(self):
        # split = 'train'
        split = 'eval'
        batch_size = 4

        eval_dataset, eval_loader = self.load_lore_dataset(split=split, batch_size=batch_size)
        config = eval_dataset.config
        model = LoreModel(config)

        batch = next(iter(eval_loader))

        logger.info(batch)

        model.eval()
        model.to(self.device)

        for k, v in batch.items():
            if k in ['meta', 'inputs']:
                continue
            batch[k] = v.to(self.device)

        with torch.no_grad():
            run_batch = {
                "pixel_values": batch["pixel_values"],
                # "meta": batch["meta"].to(self.device),
                # "meta": batch["meta"],
            }
            if "meta" in batch:
                run_batch["meta"] = batch["meta"]

            if split == "train":
                run_batch["labels"] = batch

            result = model(**run_batch)
        logger.info(f"result: {result}")

        post_processor = TableLorePostProcessor(config, output_dir=self.output_dir)

        if split != "train":
            image_name = eval_dataset.get_image_path_from_meta(batch["meta"], )
            logger.info(f"image_name: {image_name}")
            predict = post_processor(result, image_name=image_name)
            logger.info(f"predict: {predict}")


def main():
    runner = RunTableLore()
    # runner.run()
    runner.our_run_model()
    # runner.dcn_demo()
    # runner.run_ocr_table_structure_task()
    # runner.load_lore_dataset()
    # runner.test_loss()


if __name__ == '__main__':
    main()
