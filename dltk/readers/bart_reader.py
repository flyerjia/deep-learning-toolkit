# -*- encoding: utf-8 -*-
"""
@File    :   bart_reader.py
@Time    :   2023/04/05 11:17:26
@Author  :   jiangjiajia
"""
import random
import torch
from tqdm import tqdm

from ..utils.common_utils import TOKENIZERS, logger_output
from .base_reader import BaseReader


class BartReader(BaseReader):
    def __init__(self, phase, data, config):
        super(BartReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger_output('error', 'tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('tokenizer_path', ''))
        self.vocab = list(self.tokenizer.get_vocab())[6:]  # 去除特殊字符

        # self.converted_data = []
        # if len(self.data) > 0:
        #     for each_data in tqdm(self.data):
        #         self.converted_data.append(self.convert_item(each_data))

    def process_description(self, description, p=0.1):
        new_description = []
        idx = 0
        while idx < len(description):
            prob = random.random()
            if prob > p:
                new_description.append(description[idx])
            else:
                if prob < p * 0.8:  # 80% 替换
                    new_token = random.choice(self.vocab)
                    new_description.append(new_token)
                elif prob < p * 0.9:  # 10% 删除
                    pass
                else:  # 10% 变成[UNK]
                    new_description.append('[UNK]')
            idx += 1
        return new_description

    def process_diagnosis(self, diagnosis, p1=0.7, p2=0.15):
        new_diagnosis = []
        idx = 0
        if random.random() > p1:
            return diagnosis
        while idx < len(diagnosis):
            prob = random.random()
            if prob > p2:
                new_diagnosis.append(diagnosis[idx])
            else:  # 替换
                new_token = random.choice(diagnosis)
                new_diagnosis.append(new_token)
            idx += 1
        return new_diagnosis

    def convert_item(self, each_data):
        infos = each_data.split(',')
        report_ID = infos[0].strip()
        description = infos[1].strip()
        if len(infos) >= 3:
            diagnosis = infos[2].strip()
            diagnosis = diagnosis.split(' ')
        else:
            diagnosis = ''
        if len(infos) >= 4:
            clinical = infos[3].strip()
            if clinical != '':
                clinical = clinical.split(' ')
            else:
                clinical = ''
        else:
            clinical = ''
        description = description.split(' ')
        # description进行编码
        if clinical == '':
            tokenize_result = self.tokenizer.encode_plus(text=description, add_special_tokens=True, padding=self.config['padding'],
                                                         truncation=True, max_length=self.config['max_source_len'], return_tensors='pt')
        else:
            tokenize_result = self.tokenizer.encode_plus(text=description, text_pair=clinical, add_special_tokens=True, padding=self.config['padding'],
                                                         truncation=True, max_length=self.config['max_source_len'], return_tensors='pt')
        input_ids = tokenize_result['input_ids']
        attention_mask = tokenize_result['attention_mask']

        # label进行编码
        if self.phase == 'train':
            label = diagnosis + ['[EOS]']
            tokenize_result = self.tokenizer.encode_plus(text=label, add_special_tokens=False, padding=self.config['padding'],
                                                         truncation=True, max_length=self.config['max_target_len'], return_tensors='pt')
            label = tokenize_result['input_ids']
            label[label == self.tokenizer.pad_token_id] = -100
            if self.config.get('use_eda', False):
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': label,
                }
            else:
                new_diagnosis = self.process_diagnosis(diagnosis, self.config.get('p1', 0.7), self.config.get('p2', 0.15))
                new_diagnosis = ['[EOS]'] + new_diagnosis + ['[EOS]']
                new_tokenize_result = self.tokenizer.encode_plus(text=new_diagnosis, add_special_tokens=False, padding=self.config['padding'],
                                                            truncation=True, max_length=self.config['max_target_len'], return_tensors='pt')
                decoder_input_ids = new_tokenize_result['input_ids']
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'decoder_input_ids': decoder_input_ids,
                    'labels': label,
                }

        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

    def __getitem__(self, index):
        # return self.converted_data[index]
        return self.convert_item(self.data[index])

    def __len__(self):
        return len(self.data)


reader = BartReader
