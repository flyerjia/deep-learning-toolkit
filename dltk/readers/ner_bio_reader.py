# -*- encoding: utf-8 -*-
"""
@File    :   ner_bio_reader.py
@Time    :   2022/08/08 14:22:16
@Author  :   jiangjiajia
"""
import torch
from tqdm import tqdm

from ..utils.common_utils import (TOKENIZERS, fine_grade_tokenize,
                                  logger_output, read_json)
from .base_reader import BaseReader


class NERBIOReader(BaseReader):
    def __init__(self, phase, data, config):
        super(NERBIOReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger_output('error', 'tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('vocab_path', ''))
        self.label2id = read_json(self.config['label_map_path'])
        self.id2label = {int(id): label for label, id in self.label2id.items()}

        self.convert_data = []
        if self.data and len(self.data) > 0:
            for each_data in tqdm(self.data):
                self.convert_data.append(self.convert_item(each_data))

    def convert_item(self, each_data):
        def label_data(data, start, end, _type):
            """label_data"""
            for i in range(start, min(end + 1, self.config['max_seq_len'])):
                suffix = "B-" if i == start else "I-"
                data[i] = "{}{}".format(suffix, _type)
            return data
        text = each_data['text']
        entities = each_data.get('entities', [])
        text = fine_grade_tokenize(text, self.tokenizer)
        tokenize_result = self.tokenizer.encode_plus(text=text, add_special_tokens=True, padding=self.config['padding'],
                                                     truncation=True, max_length=self.config['max_seq_len'], return_tensors='pt')
        input_ids = tokenize_result['input_ids']
        token_type_ids = tokenize_result['token_type_ids']
        attention_mask = tokenize_result['attention_mask']
        labels = ['O'] * self.config['max_seq_len']
        for entity in entities:
            if start_idx < self.config['max_seq_len'] - 2 and end_idx < self.config['max_seq_len'] - 2:
                start_idx, end_idx, type = entity['start_idx'], entity['end_idx'], entity['type']
                labels = label_data(labels, start_idx + 1, end_idx + 1, type)
        label_ids = [-100] * self.config['max_seq_len']
        for idx in range(min(len(text), self.config['max_seq_len'] - 1)):
            idx += 1
            label_ids[idx] = self.label2id[labels[idx]]
        label_ids = torch.LongTensor(label_ids)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids
        }

    def __getitem__(self, index):
        return self.convert_data[index]

    def collate_fn(self, batch_data):
        input_ids = torch.cat([each_data['input_ids'] for each_data in batch_data], dim=0)
        token_type_ids = torch.cat([each_data['token_type_ids'] for each_data in batch_data], dim=0)
        attention_mask = torch.cat([each_data['attention_mask'] for each_data in batch_data], dim=0)
        label_ids = torch.stack([each_data['label_ids'] for each_data in batch_data], dim=0)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids
        }


reader = NERBIOReader
