#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ：PdfTable 
# @File     : table_inference.py
# @Author   : cycloneboy
# @Date     : 20xx/9/25 - 22:48
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lgpma.base_utils import imread


def build_model(config_file, checkpoint_file):
    device = 'cpu'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    return model


class Inference:
    def __init__(self, config_file, checkpoint_file, device=None):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = build_model(config_file, checkpoint_file)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Specify GPU device
            device = torch.device("cuda:{}".format(device))

        self.model.to(device)

    def result_format(self, pred, file_path):
        raise NotImplementedError

    def predict_single_file(self, file_path):
        pass

    def predict_batch(self, imgs):
        pass


class Structure_Recognition(Inference):
    def __init__(self, config_file, checkpoint_file, samples_per_gpu=4):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        super().__init__(config_file, checkpoint_file)
        self.samples_per_gpu = samples_per_gpu

    def result_format(self, pred, file_path=None):
        pred = pred[0]
        return pred

    def predict_single_file(self, file_path):
        # numpy inference
        img = imread(file_path)
        file_name = os.path.basename(file_path)
        result = model_inference(self.model, [img], batch_mode=True)
        result = self.result_format(result, file_path)
        result_dict = {file_name: result}
        return result, result_dict
