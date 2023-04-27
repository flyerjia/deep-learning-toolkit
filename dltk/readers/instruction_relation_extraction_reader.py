# -*- encoding: utf-8 -*-
"""
@File    :   instruction_relation_extraction_reader.py
@Time    :   2023/04/25 16:05:34
@Author  :   jiangjiajia
"""
import torch
from tqdm import tqdm

from ..utils.common_utils import TOKENIZERS, logger_output
from .base_reader import BaseReader


class InstructionREReader(BaseReader):
    def __init__(self, phase, data, config):
        super(InstructionREReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger_output('error', 'tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('tokenizer_path', ''))

        self.convert_data = []
        if self.data and len(self.data) > 0:
            for each_data in tqdm(self.data):
                self.convert_data.append(self.convert_item(each_data))

    def convert_item(self, each_data):
        instruction = each_data['instruction']
        input = each_data['input']
        text = instruction.replace('以下输入', f'【{input}】')
        encoding = self.tokenizer.encode_plus(text,
                                              add_special_tokens=False,
                                              padding=self.config['padding'],
                                              max_length=self.config['max_source_len'],
                                              truncation=True,
                                              return_tensors='pt',
                                              )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        if self.phase == 'train':
            output = each_data['output']
            output += self.tokenizer.eos_token
            encoding = self.tokenizer.encode_plus(output,
                                                  add_special_tokens=False,
                                                  padding=self.config['padding'],
                                                  max_length=self.config['max_target_len'],
                                                  truncation=True,
                                                  return_tensors='pt',
                                                  )
            labels = encoding.input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

    def __getitem__(self, index):
        # return self.convert_item(self.data[index])
        return self.convert_data[index]


reader = InstructionREReader
