# -*- encoding: utf-8 -*-
"""
@File    :   base_reader.py
@Time    :   2022/07/12 20:44:12
@Author  :   jiangjiajia
"""
import torch
from torch.utils.data import Dataset
from transformers.utils import PushToHubMixin


class BaseReader(Dataset):
    def __init__(self, phase, data, config):
        super(BaseReader, self).__init__()
        self.phase = phase  # trian dev test
        self.data = data
        self.config = config
        for name, value in config.items():
            setattr(self, name, value)

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
        outputs = {}
        for each_data in batch_data:
            for data_name, data in each_data.items():
                outputs.setdefault(data_name, []).append(data)
        for data_name, data in outputs.items():
            elem = data[0]
            if isinstance(elem, torch.Tensor):
                dim = elem.dim()
                numel = elem.numel()
                if dim == 1:
                    if numel == 1:
                        data = torch.cat(data, dim=0)
                    else:
                        data = torch.stack(data, dim=0)
                else:
                    data = torch.cat(data, dim=0)
        return outputs

    def save_tokenizer(self, save_path):
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer and isinstance(tokenizer, PushToHubMixin):
            return tokenizer.save_pretrained(save_path)
        processor = getattr(self, 'processor', None)
        if processor and isinstance(processor, PushToHubMixin):
            return processor.save_pretrained(save_path)
        return []


reader = BaseReader
