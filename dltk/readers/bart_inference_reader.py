# -*- encoding: utf-8 -*-
"""
@File    :   bart_inference_reader.py
@Time    :   2023/05/04 19:35:09
@Author  :   jiangjiajia
"""
import random
import torch
from tqdm import tqdm

from ..utils.common_utils import TOKENIZERS, logger_output
from .base_reader import BaseReader


class BartInferenceReader(BaseReader):
    def __init__(self, phase, data, config):
        super(BartInferenceReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger_output('error', 'tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('tokenizer_path', ''))

        self.converted_data = []
        if len(self.data) > 0:
            for each_data in tqdm(self.data):
                self.converted_data.append(self.convert_item(each_data))

    def convert_item(self, each_data):
        infos = each_data.split(',')
        report_ID = infos[0].strip()
        description = infos[1].strip()
        if len(infos) >= 3:
            clinical = infos[2].strip()
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
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def __getitem__(self, index):
        return self.converted_data[index]

    def __len__(self):
        return len(self.data)


reader = BartInferenceReader
