#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ：PdfTable 
# @File     : base_decoder.py
# @Author   : cycloneboy
# @Date     : 20xx/9/25 - 17:26

import torch.nn as nn


class BaseDecoder(nn.Module):
    """Base decoder class for text recognition."""

    def __init__(self, **kwargs):
        super().__init__()

    def init_weights(self):
        pass

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)


class BaseEncoder(nn.Module):
    """Base Encoder class for text recognition."""

    def init_weights(self):
        pass

    def forward(self, feat, **kwargs):
        return feat
