# -*- encoding: utf-8 -*-
"""
@File    :   base_model.py
@Time    :   2022/07/19 14:27:14
@Author  :   jiangjiajia
"""

import torch
import torch.nn as nn

from ..utils.common_utils import logger_output


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        for name, value in kwargs.items():
            setattr(self, name, value)
        # encoder的名字一定要包含‘encoder’，分层学习率才能生效

    def forward(self, **input_data):
        """
        参数 input_data 要包含phase
        """
        # train 返回loss 其他不返回loss
        raise NotImplementedError

    def get_metrics(self, phase, forward_output, forward_target, dataset=None):
        """
        计算评价指标, 参数固定

        Args:
            forward_output (Dict): {name:[batch1, batch2,...]} batch: numpy
            forward_target (Dict): {name:[batch1, batch2,...]} batch: numpy
            dataset(Dataset): dataset
        Raises:
            NotImplementedError: 模型单独实现
        """
        raise NotImplementedError

    def get_predictions(self, forward_output, dataset):
        """
        计算预测结果，参数固定

        Args:
            forward_output (Dict): {name:[batch1, batch2,...]} batch: numpy
            dataset (Dataset): dataset

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def save_predictions(self, forward_output, dataset, file_path):
        """
        保存预测结果，参数固定
        """
        raise NotImplementedError


model = BaseModel
