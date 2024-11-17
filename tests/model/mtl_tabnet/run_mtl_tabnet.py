#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ：PdfTable 
# @File     : run_mtl_tabnet.py
# @Author   : cycloneboy
# @Date     : 20xx/9/25 - 22:51
import copy
import os
import warnings

import numpy as np
import torch

from pdftable.model.table.lgpma.base_config import Config
from pdftable.model.table.lgpma.base_utils import build, imread
from pdftable.model.table.lgpma.checkpoint import load_checkpoint
from pdftable.model.table.lgpma.lgpma_preprocess import Compose
from pdftable.model.table.mtl_tabnet.master_post_processor import MasterPostProcessor
from pdftable.model.table.mtl_tabnet.table_master import MtlTabNet, TableMaster
from pdftable.utils import Constants, CommonUtils, FileUtils, logger
from pdftable.utils.constant import TABLE_ABS_PATH

DETECTORS = {
    "MtlTabNet": MtlTabNet,
    "TableMaster": TableMaster
}


def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


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


class RunMtlTabnet(object):
    def __init__(self):
        # self.model_name = "MtlTabNet"
        self.model_name = "TableMaster"
        self.device = "cuda:0"
        self.do_visualize = True
        self.output_dir = FileUtils.get_output_dir_with_time()

        self.config_file = None
        self.checkpoint = None
        self.config = None
        self.postprocess = None
        self.init_model_config()

    def init_model_config(self):
        if self.model_name == "MtlTabNet":
            self.config_file = f"{TABLE_ABS_PATH}/mtl_tabnet/mtl_tabnet_config.py"
            self.checkpoint = f'{Constants.SCOPE_MODEL_BASE_DIR}/cycloneboy/en_table_structure_mtltabnet_pubtabnet/pytorch_model.bin'
        else:
            self.config_file = f"{TABLE_ABS_PATH}/mtl_tabnet/table_master_config.py"
            self.checkpoint = f'{Constants.SCOPE_MODEL_BASE_DIR}/cycloneboy/en_table_structure_tablemaster_pubtabnet/pytorch_model.bin'

        self.config = Config.fromfile(self.config_file)
        self.postprocess = MasterPostProcessor(output_dir=self.output_dir)

        logger.info(f"init model: {self.model_name} - {self.checkpoint}")

    def build_model(self):
        model = build_detector(self.config.model, test_cfg=self.config.get('test_cfg'))

        os.makedirs(self.output_dir, exist_ok=True)
        save_dir = f"{self.output_dir}/{self.model_name.lower()}"
        CommonUtils.print_model_param(model, save_dir=save_dir)

        if self.checkpoint is not None:
            map_loc = 'cpu' if self.device == 'cpu' else None
            checkpoint = load_checkpoint(model, self.checkpoint, map_location=map_loc)

        # Save the config in the model for convenience
        model.cfg = self.config
        model.to(self.device)
        model.eval()

        return model

    def run(self):
        model = self.build_model()

        img_path = f"{Constants.SRC_IMAGE_DIR}/table_01.jpg"

        img = imread(img_path)
        is_ndarray = isinstance(img, np.ndarray)

        cfg = self.config.copy()
        if is_ndarray:
            # set loading pipeline type
            cfg.test_pipeline[0].type = 'LoadImageFromNdarray'

        cfg.test_pipeline = replace_ImageToTensor(cfg.test_pipeline)

        if is_ndarray:
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # Build the data pipeline
        test_pipeline = Compose(cfg.test_pipeline)

        data = test_pipeline(data)

        image_data = data["img"].data
        run_data = {
            "img_metas": [[data["img_metas"].data]],
            "img": image_data.to(self.device).unsqueeze(0)
        }
        # Forward inference
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **run_data)[0]

        table_results = self.postprocess(result, file_name=img_path)

        table_results["bbox"] = result["bbox"].tolist()
        table_results["new_bbox"] = result["new_bbox"].tolist()
        logger.info(f"result: {table_results}")

        FileUtils.dump_json(f"{self.output_dir}/{FileUtils.get_file_name(img_path)}.json",
                            table_results)

    def run_master(self):
        model = self.build_model()


def main():
    runner = RunMtlTabnet()
    runner.run()
    # runner.run_master()


if __name__ == '__main__':
    main()
