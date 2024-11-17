#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ：PdfTable 
# @File     : configuration_lore.py
# @Author   : cycloneboy
# @Date     : 20xx/9/9 - 10:39


from collections import OrderedDict
from typing import Mapping, Dict, List

from transformers import PretrainedConfig

__all__ = [
    "LoreConfig",
]


class LoreConfig(PretrainedConfig):
    r"""

    """
    model_type = "table_structure"

    def __init__(
            self,
            model_name: str = "Lore",
            backbone: str = "ResNet-18",
            task_type: str = "wireless",
            resolution: str = "768, 768",
            stacking_layers: int = 4,
            tsfm_layers: int = 4,
            upper_left: bool = True,
            wiz_2dpe: bool = True,
            wiz_stacking: bool = True,
            wiz_rev: bool = False,
            vis_thresh_corner: float = 0.3,
            vis_thresh: float = 0.2,
            scores_thresh: float = 0.2,
            eval: bool = True,
            model_path: str = "",
            pretrained=False,
            debug: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.backbone = backbone
        self.task_type = task_type
        self.resolution = [int(v.strip()) for v in resolution.split(",")]
        self.stacking_layers = stacking_layers
        self.tsfm_layers = tsfm_layers
        self.upper_left = upper_left
        self.wiz_2dpe = wiz_2dpe
        self.wiz_stacking = wiz_stacking
        self.wiz_rev = wiz_rev
        self.vis_thresh_corner = vis_thresh_corner
        self.vis_thresh = vis_thresh
        self.scores_thresh = scores_thresh
        self.eval = eval
        self.model_path = model_path
        self.pretrained = pretrained
        self.model_provider = "model_scope"
        self.predictor_type = "pytorch"
        self.debug = debug

        self.reg_offset = True
        self.hm_weight = 1
        self.wh_weight = 1
        self.off_weight = 1

        if self.task_type == "wireless":
            self.backbone = "ResNet-18"
            self.resolution = [768, 768]
            self.stacking_layers = 4
            self.tsfm_layers = 4
            self.upper_left = True
            self.wiz_2dpe = True
            self.wiz_4ps = False
            self.wiz_stacking = True
            self.wiz_pairloss = False
            self.wiz_rev = False
            self.vis_thresh_corner = 0.3
            self.vis_thresh = 0.2
            self.scores_thresh = 0.2
        elif self.task_type == "wtw":
            self.backbone = "DLA-34"
            self.resolution = [1024, 1024]
            self.stacking_layers = 4
            self.tsfm_layers = 4
            self.upper_left = False
            self.wiz_2dpe = False
            self.wiz_4ps = True
            self.wiz_stacking = True
            self.wiz_pairloss = True
            self.wiz_rev = True
            self.vis_thresh_corner = 0.3
            self.vis_thresh = 0.2
            self.scores_thresh = 0.2
        # elif self.task_type == "ptn":
        else:
            self.backbone = "DLA-34"
            self.resolution = [512, 512]
            self.stacking_layers = 3
            self.tsfm_layers = 3
            self.upper_left = False
            self.wiz_2dpe = True
            self.wiz_4ps = False
            self.wiz_stacking = True
            self.wiz_pairloss = False
            self.wiz_rev = False
            self.vis_thresh_corner = 0.3
            self.vis_thresh = 0.35
            self.scores_thresh = 0.35

