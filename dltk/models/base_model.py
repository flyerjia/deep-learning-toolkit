# -*- encoding: utf-8 -*-
"""
@File    :   base_model.py
@Time    :   2022/07/19 14:27:14
@Author  :   jiangjiajia
"""

import torch
import torch.nn as nn

from ..utils.common_utils import logger_output, write_json


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

    def get_metrics(self, phase, predictions, dataset):
        """
        计算评价指标, 参数固定

        Args:
            predictions (List): 预测结果
            dataset(Dataset): dataset
        Raises:
            NotImplementedError: 模型单独实现
        """
        raise NotImplementedError

    def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
        """
        计算预测结果，参数固定，对每个batch的数据进行解码

        Args:
            forward_output (Dict): {name:batch_data} batch_data: numpy
            forward_target (Dict): {name:batch_data} batch_data: numpy
            dataset (Dataset): dataset
            batch_start_index (int): 对于dataset中，对应的数据起始位置

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def save_predictions(self, predictions, file_path):
        """
        保存预测结果，参数固定，可根据需求自行设置保存格式
        """
        write_json(file_path, predictions)


model = BaseModel
