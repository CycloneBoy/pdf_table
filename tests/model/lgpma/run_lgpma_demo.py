#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_lgpma_demo
# @Author  ：cycloneboy
# @Date    ：20xx/9/21 17:21
import json
import warnings

import cv2
import torch

from pdftable.model.table.lgpma.base_config import Config
from pdftable.model.table.lgpma.base_utils import build
from pdftable.model.table.lgpma.checkpoint import load_checkpoint
from pdftable.model.table.lgpma.lgpma_preprocess import Compose
from pdftable.model.table.lgpma.model_lgpma import LGPMA
from pdftable.utils import Constants, TimeUtils, CommonUtils, FileUtils, logger

DETECTORS = {
    "LGPMA": LGPMA
}


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


class RunLgpmaDemo(object):

    def __init__(self):
        self.config_file = f"{Constants.SRC_HOME_DIR}/tests/model/lgpma/lgpma_base.py"
        self.config = Config.fromfile(self.config_file)
        self.checkpoint = f'{Constants.SCOPE_MODEL_BASE_DIR}/cycloneboy/en_table_structure_lgpma_pubtabnet/maskrcnn-lgpma-pub-e12-pub.pth'
        self.device = "cuda:0"
        self.do_visualize = True

    def run(self):
        # model = LGPMA(**self.config)
        run_time = TimeUtils.now_str_short()

        model = build_detector(self.config.model, test_cfg=self.config.get('test_cfg'))

        CommonUtils.print_model_param(model)

        if self.checkpoint is not None:
            map_loc = 'cpu' if self.device == 'cpu' else None
            checkpoint = load_checkpoint(model, self.checkpoint, map_location=map_loc)

        # Save the config in the model for convenience
        model.cfg = self.config
        model.to(self.device)
        model.eval()

        # Build the data pipeline
        test_pipeline = Compose(self.config.data.test.pipeline)

        img_path = f"{Constants.SRC_IMAGE_DIR}/table_01.jpg"
        # img_path = f"/nlp_data/pdftable/outputs/pdf/inference_results/pdf_debug/2024-11-16/20241116_132532/page-6.png"

        vis_dir = FileUtils.get_output_dir_with_time(add_now_end=False)
        save_path = f"{vis_dir}/{run_time}/predict_{run_time}.json"  # path to save prediction
        FileUtils.check_file_exists(save_path)

        # If the input is single image
        data = dict(img=img_path)
        data = test_pipeline(data)
        device = int(str(self.device).split(":")[-1])
        # data = scatter(collate([data], samples_per_gpu=1), [device])[0]

        image_data = data["img"][0].data
        run_data = {
            "img_metas": [[data["img_metas"][0].data]],
            "img": [image_data.to(self.device).unsqueeze(0)]
        }
        # Forward inference
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **run_data)[0]

        logger.info(f"result: {result}")

        pred_dict = dict()

        pred_dict[img_path] = result['html']

        print(f"bboxes: {len(result['content_ann']['bboxes'])}")
        print(f"bboxes: {result['content_ann']['bboxes']}")

        # detection results visualization
        if self.do_visualize:
            img = cv2.imread(img_path)
            img_name = img_path.split("/")[-1]
            bboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in result['content_ann']['bboxes']]
            for box in bboxes:
                for j in range(0, len(box), 2):
                    cv2.line(img, (box[j], box[j + 1]), (box[(j + 2) % len(box)], box[(j + 3) % len(box)]), (0, 0, 255),
                             1)
            image_file = f"{vis_dir}/{run_time}/{run_time}_{img_name}"
            cv2.imwrite(image_file, img)
            print(f'save image: {image_file}')

        with open(save_path, "w", encoding="utf-8") as writer:
            json.dump(pred_dict, writer, ensure_ascii=False)


def main():
    runner = RunLgpmaDemo()
    runner.run()


if __name__ == '__main__':
    main()
