#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : configuration_mtl_tabnet.py
# @Author   : cycloneboy
# @Date     : 20xx/12/15 - 22:33
from typing import Dict

from transformers import PretrainedConfig

from pdftable.model.table.lgpma.base_config import Config
from pdftable.utils.constant import TABLE_ABS_PATH

"""
MtlTabnet configuration
"""

__all__ = [
    "MtlTabnetConfig",
]


class MtlTabnetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieModel`] or a [`TFErnieModel`]. It is used to
    instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ERNIE

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        task_type (`str`, *optional*, defaults 'ch' ):
            db_net - ['ch', 'en', 'table]

    Examples:

    """
    model_type = "MtlTabNet"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
            self,
            model_name: str = "MtlTabNet",
            backbone: str = "TableResNetExtra",
            task_type: str = "PubTabNet",
            config_file: str = f"{TABLE_ABS_PATH}/mtl_tabnet/mtl_tabnet_config.py",
            model_path: str = "",
            debug: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.backbone = backbone
        self.task_type = task_type
        self.config_file = self.get_config_file(config_file)
        self.model_path = model_path
        self.model_provider = "Other"
        self.predictor_type = "pytorch"
        self.debug = debug
        self.config = Config.fromfile(self.config_file)

        if task_type in ["FinTabNet", "fin"] and model_name == "MtlTabNet":
            self.task_type = "FinTabNet"
        else:
            self.task_type = "PubTabNet"

    def get_config_file(self, default_config):
        config_mapping = {
            "Lgpma": f"{TABLE_ABS_PATH}/lgpma/lgpma_base.py",
            "MtlTabNet": f"{TABLE_ABS_PATH}/mtl_tabnet/mtl_tabnet_config.py",
            "TableMaster": f"{TABLE_ABS_PATH}/mtl_tabnet/table_master_config.py",
        }

        config_file = config_mapping.get(self.model_name, default_config)
        return config_file

    def get_test_config(self):
        test_cfg = self.config.get('test_cfg')
        cfg = dict(train_cfg=None, test_cfg=test_cfg)
        return cfg

    def get_test_pipeline_config(self):
        cfg = self.config.test_pipeline
        return cfg

    def get_model_config(self):
        cfg = self.config.model
        return cfg
