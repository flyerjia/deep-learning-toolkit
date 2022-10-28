# -*- encoding: utf-8 -*-
"""
@File    :   multi_span_relation_extraction_reader_bce.py
@Time    :   2022/07/28 10:39:41
@Author  :   jiangjiajia
"""

import logging

import torch

from ..utils.common_utils import TOKENIZERS, fine_grade_tokenize
from .base_reader import BaseReader

logger = logging.getLogger(__name__)


class MultiSREReader(BaseReader):
    def __init__(self, phase, data, config):
        super(MultiSREReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger.error('tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('vocab_path', ''))

        self.label2id = {
            'EH2ET_': 0,
            'H2H_1': 1,
            'H2H_2': 2,
            'H2H_3': 3,
            'T2T_1': 4,
            'T2T_2': 5,
            'T2T_3': 6
        }
        self.id2label = {id: label for label, id, in self.label2id.items()}
        self.num_labels = len(self.label2id.keys())

        self.convert_data = []
        if self.data and len(self.data) > 0:
            for each_data in self.data:
                self.convert_data.append(self.convert_item(each_data))

    def convert_item(self, each_data):
        text = each_data['text']
        relation_of_mention_list = each_data.get('relation_of_mention', [])
        text = fine_grade_tokenize(text, self.tokenizer)
        tokenize_result = self.tokenizer.encode_plus(text=text, add_special_tokens=True, padding=self.config['padding'],
                                                     truncation=True, max_length=self.config['max_seq_len'], return_tensors='pt')
        input_ids = tokenize_result['input_ids']
        token_type_ids = tokenize_result['token_type_ids']
        attention_mask = tokenize_result['attention_mask']
        labels = torch.zeros((self.config['max_seq_len'], self.config['max_seq_len'], self.num_labels))
        label_mask = torch.ones((self.config['max_seq_len'], self.config['max_seq_len']))
        label_mask = label_mask.masked_fill((attention_mask.t() * attention_mask) == 0, 0)

        for relation_of_mention in relation_of_mention_list:
            relation, head, tail = relation_of_mention['relation'], relation_of_mention['head'], relation_of_mention['tail']
            if relation in [1, 3]:
                head_start_idx, head_end_idx = head['start_idx'] + 1, head['end_idx']
                tail_start_idx, tail_end_idx = tail['start_idx'] + 1, tail['end_idx']
                labels[head_start_idx, head_end_idx, 0] = 1
                labels[tail_start_idx, tail_end_idx, 0] = 1
                # s->o
                labels[head_start_idx, tail_start_idx, self.label2id['H2H_'+str(relation)]] = 1
                labels[head_end_idx,  tail_end_idx, self.label2id['T2T_'+str(relation)]] = 1
            else:  # 2
                head_start_idx, head_end_idx = head['start_idx'] + 1, head['end_idx']
                sub_relation, sub_head, sub_tail = tail['relation'], tail['head'], tail['tail']
                assert sub_relation == 1
                sub_head_start_idx, sub_head_end_idx = sub_head['start_idx'] + 1, sub_head['end_idx']
                sub_tail_start_idx, sub_tail_end_idx = sub_tail['start_idx'] + 1, sub_tail['end_idx']
                labels[head_start_idx, head_end_idx, 0] = 1
                labels[sub_head_start_idx, sub_head_end_idx, 0] = 1
                labels[sub_tail_start_idx, sub_tail_end_idx, 0] = 1
                # s->o
                labels[sub_head_start_idx, sub_tail_start_idx, self.label2id['H2H_'+str(sub_relation)]] = 1
                labels[sub_head_end_idx,  sub_tail_end_idx, self.label2id['T2T_'+str(sub_relation)]] = 1
                # c->s
                labels[head_start_idx, sub_head_start_idx, self.label2id['H2H_'+str(relation)]] = 1
                labels[head_end_idx, sub_head_end_idx, self.label2id['T2T_'+str(relation)]] = 1
                # c->o
                labels[head_start_idx, sub_tail_start_idx, self.label2id['H2H_'+str(relation)]] = 1
                labels[head_end_idx, sub_tail_end_idx, self.label2id['T2T_'+str(relation)]] = 1
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


reader = MultiSREReader
