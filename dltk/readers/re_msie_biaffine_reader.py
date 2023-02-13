# -*- encoding: utf-8 -*-
"""
@File    :   re_msie_biaffine_reader.py
@Time    :   2023/02/01 17:05:55
@Author  :   jiangjiajia
"""
import torch
from tqdm import tqdm

from ..utils.common_utils import TOKENIZERS, fine_grade_tokenize, logger_output, read_json
from .base_reader import BaseReader


class REMSIEBiaffineReader(BaseReader):
    def __init__(self, phase, data, config):
        super(REMSIEBiaffineReader, self).__init__(phase, data, config)
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

    def search(self, text, word):
        """从text中寻找子串word
        如果找到，返回第一个下标；否则返回None。
        """
        n = len(word)
        for i in range(len(text)):
            if ''.join(text[i:i + n]) == ''.join(word):
                return i, i + n - 1
        return None

    def convert_item(self, each_data):
        text = each_data['text']
        spo_list = each_data.get('spo_list', [])
        text = fine_grade_tokenize(text, self.tokenizer)
        tokenize_result = self.tokenizer.encode_plus(text=text, add_special_tokens=True, padding=self.config['padding'],
                                                     truncation=True, max_length=self.config['max_seq_len'], return_tensors='pt')
        input_ids = tokenize_result['input_ids']
        token_type_ids = tokenize_result['token_type_ids']
        attention_mask = tokenize_result['attention_mask']
        labels = torch.zeros((self.config['max_seq_len'], self.config['max_seq_len'])).long()
        label_mask = torch.ones((self.config['max_seq_len'], self.config['max_seq_len']))
        label_mask = label_mask.masked_fill((attention_mask.t() * attention_mask) == 0, 0)
        for each_spo in spo_list:
            subject = each_spo['subject']
            subject_type = each_spo['subject_type']
            predicate = each_spo['predicate']
            object = each_spo['object']['@value']
            object_type = each_spo['object_type']['@value']
            subject_start_idx, subject_end_idx = self.search(text, fine_grade_tokenize(subject, self.tokenizer))
            object_start_idx, object_end_idx = self.search(text, fine_grade_tokenize(object, self.tokenizer))
            if subject_start_idx < self.config['max_seq_len'] - 2 and subject_end_idx < self.config['max_seq_len'] - 2 and \
                    object_start_idx < self.config['max_seq_len'] - 2 and object_end_idx < self.config['max_seq_len'] - 2:
                labels[subject_start_idx + 1, subject_end_idx + 1] = self.label2id['EH2ET_' + subject_type]
                labels[object_start_idx + 1, object_end_idx + 1] = self.label2id['EH2ET_' + object_type]
                labels[subject_start_idx + 1, object_start_idx + 1] = self.label2id['H2H_' + predicate]
                labels[subject_end_idx + 1, object_end_idx + 1] = self.label2id['T2T_' + predicate]

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


reader = REMSIEBiaffineReader
