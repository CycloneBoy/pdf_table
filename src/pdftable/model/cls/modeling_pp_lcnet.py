#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project  : PdfTable
# @File     : modeling_pp_lcnet.py
# @Author   : cycloneboy
# @Date     : 20xx/10/15 - 15:47
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from .configuration_cls_pulc import ClsPulcConfig
from ...utils import logger

"""
ocr detection
"""

__all__ = [
    "ClsPPLcnet"
]


class ClsPPLcnet(nn.Module):

    def __init__(self, config: ClsPulcConfig, **kwargs):
        """initialize the ocr recognition model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__()

        self.config = config
