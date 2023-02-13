# -*- encoding: utf-8 -*-
"""
@File    :   global_pointer_crossentropy.py
@Time    :   2023/01/28 20:09:05
@Author  :   jiangjiajia
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_label_crossentropy import MultiLabelCrossentropy


class GlobalPointerCrossentropy(nn.Module):
    def __init__(self, reduction='mean') -> None:
        super(GlobalPointerCrossentropy, self).__init__()
        self.reduction = reduction
        self.multi_label_crossentropy = MultiLabelCrossentropy(reduction)
        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, logits, targets):
        bh = logits.shape[0] * logits.shape[1]
        y_pred = torch.reshape(logits, (bh, -1)) # [b, n, n, l] -> [b * n, n * l]
        y_true = torch.reshape(targets, (bh, -1))
        loss = self.multi_label_crossentropy(y_pred, y_true)
        return loss
