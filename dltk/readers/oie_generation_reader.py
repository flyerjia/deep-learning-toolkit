# -*- encoding: utf-8 -*-
"""
@File    :   oie_generation_reader.py
@Time    :   2022/11/14 19:27:54
@Author  :   jiangjiajia
"""
import torch
from tqdm import tqdm

from ..utils.common_utils import TOKENIZERS, logger_output
from .base_reader import BaseReader


class OIEGenerationReader(BaseReader):
    def __init__(self, phase, data, config):
        super(OIEGenerationReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger_output('error', 'tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('vocab_path', ''))

        # self.convert_data = []
        # if self.data and len(self.data) > 0:
        #     for each_data in tqdm(self.data):
        #         self.convert_data.append(self.convert_item(each_data))

    def convert_item(self, each_data):
        def get_sub(spo_info_list):
            result = []
            for each_item in spo_info_list:
                if each_item['type'] == 'subject':
                    result.append(each_item['text'])
                    break
            return result

        def get_pre(spo_info_list):
            result = []
            for each_item in spo_info_list:
                if each_item['type'] == 'predicate':
                    result.append(each_item['text'])
            return result

        def get_obj(spo_info_list):
            result = []
            for each_item in spo_info_list:
                if each_item['type'] == 'object':
                    result.append(each_item['text'])
            return result

        text = each_data['text']
        encoding = self.tokenizer(text,
                                  padding=self.config['padding'],
                                  max_length=self.config['max_source_length'],
                                  truncation=True,
                                  return_tensors='pt',
                                  )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        # labels  (S:s_text,P:p_text,O:o_text1,O:o_text2);()
        if 'open_spo_list' not in each_data.keys():
            labels = torch.ones((1, 1)).long()
        else:
            labels = []
            for spo_info in each_data.get('open_spo_list', []):
                target_label = '('
                subjects = get_sub(spo_info)
                if len(subjects) > 0:
                    target_label += ','.join(['S:' + i for i in subjects])
                predicates = get_pre(spo_info)
                if len(predicates) > 0:
                    if target_label == '(':
                        target_label += ','.join(['P:' + i for i in predicates])
                    else:
                        target_label += ',' + ','.join(['P:' + i for i in predicates])
                else:
                    continue
                objects = get_obj(spo_info)
                if len(objects) > 0:
                    target_label += ',' + ','.join(['O:' + i for i in objects])
                if target_label != '(':
                    target_label += ')'
                    labels.append(target_label)
            labels = ';'.join(labels)
            target_encoding = self.tokenizer(labels,
                                             padding=self.config['padding'],
                                             max_length=self.config['max_target_length'],
                                             truncation=True,
                                             return_tensors='pt',
                                             )
            labels = target_encoding.input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __getitem__(self, index):
        return self.convert_item(self.data[index])

    def collate_fn(self, batch_data):
        input_ids = torch.cat([each_data['input_ids'] for each_data in batch_data], dim=0)
        attention_mask = torch.cat([each_data['attention_mask'] for each_data in batch_data], dim=0)
        labels = torch.cat([each_data['labels'] for each_data in batch_data], dim=0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


reader = OIEGenerationReader
