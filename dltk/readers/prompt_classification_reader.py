# -*- encoding: utf-8 -*-
"""
@File    :   prompt_classification_reader.py
@Time    :   2022/08/11 11:06:18
@Author  :   jiangjiajia
"""
import torch
from tqdm import tqdm

from ..utils.common_utils import TOKENIZERS, fine_grade_tokenize, logger_output
from .base_reader import BaseReader


class PromptClassificationReader(BaseReader):
    def __init__(self, phase, data, config):
        super(PromptClassificationReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger_output('error', 'tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('vocab_path', ''))

        self.label2id = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            9: 4
        }
        self.id2label = {id: label for label, id, in self.label2id.items()}
        self.num_labels = len(self.label2id.keys())

        self.convert_data = []
        if self.data and len(self.data) > 0:
            for each_data in tqdm(self.data):
                self.convert_data.append(self.convert_item(each_data))

    def convert_item(self, each_data):
        text = each_data['text']
        question = each_data['question']
        answers = each_data.get('answer', -1)
        # text = fine_grade_tokenize(text, self.tokenizer)
        # text太长的时候删除前80个字符
        # if len(text) > 500:
        #     for i in range(80, -1, -1):
        #         if text[i] in ['。', '！', '!', '？', '?']:
        #             text = text[i + 1:]
        #             break
        # question = fine_grade_tokenize(question, self.tokenizer)
        tokenize_result = self.tokenizer.encode_plus(text=question, text_pair=text, add_special_tokens=True, padding=self.config['padding'],
                                                     truncation=True, max_length=self.config['max_seq_len'], return_tensors='pt')
        input_ids = tokenize_result['input_ids']
        token_type_ids = tokenize_result['token_type_ids']
        attention_mask = tokenize_result['attention_mask']
        label = torch.LongTensor([self.label2id.get(answers, 0)])
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': label,
        }

    def __getitem__(self, index):
        return self.convert_data[index]

    def __len__(self):
        return len(self.convert_data)

    def collate_fn(self, batch_data):
        input_ids = torch.cat([each_data['input_ids'] for each_data in batch_data], dim=0)
        token_type_ids = torch.cat([each_data['token_type_ids'] for each_data in batch_data], dim=0)
        attention_mask = torch.cat([each_data['attention_mask'] for each_data in batch_data], dim=0)
        labels = torch.cat([each_data['labels'] for each_data in batch_data], dim=0)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


reader = PromptClassificationReader
