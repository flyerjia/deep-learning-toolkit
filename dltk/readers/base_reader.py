# -*- encoding: utf-8 -*-
"""
@File    :   base_reader.py
@Time    :   2022/07/12 20:44:12
@Author  :   jiangjiajia
"""
import torch
from torch.utils.data import Dataset


class BaseReader(Dataset):
    def __init__(self, phase, data, config):
        super(BaseReader, self).__init__()
        self.phase = phase  # trian dev test
        self.data = data
        self.config = config

    def convert_item(self, each_data):
        """
        每条数据的处理
        """
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch_data):
        raise NotImplementedError


reader = BaseReader
