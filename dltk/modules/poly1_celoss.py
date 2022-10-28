# -*- encoding: utf-8 -*-
"""
@File    :   poly1_celoss.py
@Time    :   2022/08/17 12:04:27
@Author  :   jiangjiajia
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1
