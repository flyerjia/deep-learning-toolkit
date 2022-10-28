# -*- encoding: utf-8 -*-
"""
@File    :   poly1_cefocalloss.py
@Time    :   2022/08/17 14:35:20
@Author  :   jiangjiajia
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyFocalLoss(nn.Module):
    def __init__(self, epsilon: float = 1.0,
                 gamma: float = 2.0,
                 alpha: list[float] = None,
                 onehot_encoded: bool = False,
                 reduction: str = "mean",
                 weight: Optional[torch.Tensor] = None,
                 pos_weight: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the instance of PolyFocalLoss
        :param epsilon: scaling factor for leading polynomial coefficient
        :param gamma: exponent of the modulating factor (1 - p_t) to balance
                      easy vs hard examples.
        :param alpha: weighting factor per class in range (0,1) to balance
                      positive vs negative examples.
        :param onehot_encoded: True if target is one hot encoded.
        :param reduction: the reduction to apply to the output
                          'none': no reduction will be applied,
                          'mean': the weighted mean of the output is taken,
                          'sum': the output will be summed.
        """
        super(PolyFocalLoss, self).__init__()
        self.eps = epsilon
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, list):
                raise ValueError("Expected list of floats between 0-1"
                                 " for each class or None.")
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = alpha
        self.reduction = reduction
        self.onehot_encoded = onehot_encoded

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: output of neural network tensor of shape (n, num_classes)
                      or (n, num_classes, ...)
        :param target: ground truth tensor of shape (n, ) or (n, ...)
        :return: polyfocalloss
        """
        num_classes = input.shape[1]
        if not self.onehot_encoded:
            if target.ndim == 1:
                target1 = F.one_hot(target, num_classes=num_classes)
            else:
                # target is of size (n, ...)
                target1 = target.unsqueeze(1)  # (n, 1, ...)
                # (n, 1, ...) => (n, 1, ... , num_classes)
                target1 = F.one_hot(target1, num_classes=num_classes)
                # (n, 1, ..., num_classes) => (n, num_classes, ..., 1)
                target1 = target1.transpose(1, -1)
                # (n, num_classes, ..., 1) => (n, num_classes, ...)
                target1 = target1.squeeze(-1)

        target1 = target1.to(device=input.device, dtype=input.dtype)

        ce_loss =\
            F.cross_entropy(input, target1, reduction="none")

        p_t = torch.exp(-ce_loss)
        loss = torch.pow((1 - p_t), self.gamma) * ce_loss

        if self.alpha is not None:
            if len(self.alpha) != num_classes:
                raise ValueError("Alpha value is not available"
                                 " for all the classes.")
            if torch.count_nonzero(self.alpha) == 0:
                raise ValueError("All values can't be 0.")
            self.alpha = self.alpha/sum(self.alpha)
            alpha_t = self.alpha.gather(0, target.data.view(-1))
            loss *= alpha_t
            poly_loss = loss + self.eps * torch.pow(1-p_t,
                                                    self.gamma+1) * alpha_t
        else:
            poly_loss = loss + self.eps * torch.pow(1-p_t, self.gamma+1)

        if self.reduction == "mean":
            poly_loss = poly_loss.mean()
        elif self.reduction == "sum":
            poly_loss = poly_loss.sum()

        return poly_loss
