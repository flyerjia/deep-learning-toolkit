# -*- encoding: utf-8 -*-
"""
@File    :   multi_label_crossentropy.py
@Time    :   2023/01/29 15:40:19
@Author  :   jiangjiajia
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelCrossentropy(nn.Module):
    def __init__(self, reduction='mean') -> None:
        super(MultiLabelCrossentropy, self).__init__()
        self.reduction = reduction
        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, y_pred, y_true):
        # y_pred: [..., c]
        # y_target: [..., c]
        # c: number of labels
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
