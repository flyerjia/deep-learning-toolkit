# -*- encoding: utf-8 -*-
"""
@File    :   ner_biaffine_reader.py
@Time    :   2023/01/29 15:20:51
@Author  :   jiangjiajia
"""
import torch
from tqdm import tqdm

from ..utils.common_utils import (TOKENIZERS, fine_grade_tokenize,
                                  logger_output, read_json)
from .base_reader import BaseReader


class NEBiaffineReader(BaseReader):
    def __init__(self, phase, data, config):
        super(NEBiaffineReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger_output('error', 'tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('vocab_path', ''))
        self.label2id = read_json(self.config['label_map_path'])
        self.id2label = {int(id): label for label, id in self.label2id.items()}
        self.num_labels = len(self.label2id.keys())

        self.convert_data = []
        if self.data and len(self.data) > 0:
            for each_data in tqdm(self.data):
                self.convert_data.append(self.convert_item(each_data))

    def convert_item(self, each_data):
        text = each_data['text']
        entities = each_data.get('entities', [])
        text = fine_grade_tokenize(text, self.tokenizer)
        tokenize_result = self.tokenizer.encode_plus(text=text, add_special_tokens=True, padding=self.config['padding'],
                                                     truncation=True, max_length=self.config['max_seq_len'], return_tensors='pt')
        input_ids = tokenize_result['input_ids']
        token_type_ids = tokenize_result['token_type_ids']
        attention_mask = tokenize_result['attention_mask']
        labels = torch.zeros((self.config['max_seq_len'], self.config['max_seq_len'])).long()
        label_mask = torch.ones((self.config['max_seq_len'], self.config['max_seq_len']))
        label_mask = label_mask.masked_fill((attention_mask.t() * attention_mask) == 0, 0)
        label_mask = torch.triu(label_mask) # 去除下三角
        for entity in entities:
            start_idx, end_idx, type = entity['start_idx'], entity['end_idx'], entity['type']
            if start_idx < self.config['max_seq_len'] - 2 and end_idx < self.config['max_seq_len'] - 2:
                labels[start_idx + 1, end_idx + 1] = self.label2id[type]  # 增加CLS位置
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'label_mask': label_mask
        }

    def __getitem__(self, index):
        return self.convert_data[index]

    def collate_fn(self, batch_data):
        input_ids = torch.cat([each_data['input_ids'] for each_data in batch_data], dim=0)
        token_type_ids = torch.cat([each_data['token_type_ids'] for each_data in batch_data], dim=0)
        attention_mask = torch.cat([each_data['attention_mask'] for each_data in batch_data], dim=0)
        labels = torch.stack([each_data['labels'] for each_data in batch_data], dim=0)
        label_mask = torch.stack([each_data['label_mask'] for each_data in batch_data], dim=0)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'label_mask': label_mask
        }


reader = NEBiaffineReader
