#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：lore_loss
# @Author  ：cycloneboy
# @Date    ：20xx/11/7 11:32

import torch.nn as nn

from pdftable.loss.common_loss import FocalLoss, RegL1Loss, AxisLoss, PairLoss
from pdftable.model.lore.configuration_lore import LoreConfig
from pdftable.utils.model.model_utils import _sigmoid


class TableLoreLoss(nn.Module):

    def __init__(self, config: LoreConfig):
        super().__init__()
        self.config = config

        self.crit = FocalLoss()
        self.crit_mk = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = self.crit_reg
        self.crit_st = self.crit_reg
        self.crit_ax = AxisLoss()
        self.pair_loss = PairLoss()

    def forward(self, outputs, batch, logi=None, slogi=None):
        opt = self.config
        """hm, re, off, wh losses are original losses of CenterNet detector, and the st loss is the loss for parsing-grouping in Cycle-CenterNet."""
        hm_loss, st_loss, re_loss, off_loss, wh_loss, lo_loss, ax_loss, sax_loss, sm_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0

        output = outputs[0]
        output['hm'] = _sigmoid(output['hm'])

        """LOSS FOR DETECTION MODULE"""

        if opt.wiz_pairloss:
            hm_loss += self.crit(output['hm'], batch['hm'])

            loss1, loss2 = \
                self.pair_loss(output['wh'], batch['hm_ind'], output['st'], batch['mk_ind'], batch['hm_mask'], \
                               batch['mk_mask'], batch['ctr_cro_ind'], batch['wh'], batch['st'], batch['hm_ctxy'])

            wh_loss += loss1
            st_loss += loss2
        else:
            hm_loss += self.crit(output['hm'][:, 0, :, :],
                                 batch['hm'][:, 0, :, :])  # only supervision on centers
            wh_loss += self.crit_wh(output['wh'], batch['hm_mask'], batch['hm_ind'], batch['wh'])

        if opt.reg_offset and opt.off_weight > 0:
            off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['reg_ind'],
                                      batch['reg'])

        """LOSS FOR RECONSTRUCTION MODULE"""

        ax_loss = self.crit_ax(output['ax'], batch['hm_mask'], batch['hm_ind'], batch['logic'], logi)

        '''COMBINING LOSSES'''

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + 2 * ax_loss

        if opt.wiz_pairloss:
            loss = loss + st_loss

        if opt.wiz_stacking:
            sax_loss = self.crit_ax(output['ax'], batch['hm_mask'], batch['hm_ind'], batch['logic'], slogi)
            loss = loss + 2 * sax_loss
            # sacc = _axis_eval(output['ax'], batch['hm_mask'], batch['hm_ind'], batch['logic'], slogi)
            # logger.info(f'sacc: {sacc}')

        '''CONSTRUCTING LOSS STATUS'''

        # weather asking for grouping
        if opt.wiz_pairloss:
            loss_stats = {'loss': loss, 'hm_l': hm_loss, 'wh_l': wh_loss, "st_l": st_loss, "ax_l": ax_loss}
        else:
            loss_stats = {'loss': loss, 'hm_l': hm_loss, 'wh_l': wh_loss, "ax_l": ax_loss}

            # weather asking for stacking
        if opt.wiz_stacking:
            loss_stats['sax_l'] = sax_loss

        # logger.info(f'loss: {loss} - {loss_stats}')
        return loss, loss_stats
