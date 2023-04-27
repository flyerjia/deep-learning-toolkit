# -*- encoding: utf-8 -*-
"""
@File    :   fgm.py
@Time    :   2022/07/21 15:12:09
@Author  :   jiangjiajia
"""
import torch


class FGM:
    def __init__(self, model, epsilon=0.3):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='shared.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='shared.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
