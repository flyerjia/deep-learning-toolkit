# -*- encoding: utf-8 -*-
"""
@File    :   ce_focalloss.py
@Time    :   2022/08/11 17:57:49
@Author  :   jiangjiajia
"""
import torch
import torch.nn as nn

class CEFocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(CEFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight,
                                            reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
