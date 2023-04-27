# -*- encoding: utf-8 -*-
"""
@File    :   bart_pretrain_reader.py
@Time    :   2023/04/04 16:29:07
@Author  :   jiangjiajia
"""
import torch
import copy
import random
import numpy as np
from typing import List

from ..utils.common_utils import TOKENIZERS, logger_output
from .base_reader import BaseReader


class BartPretrainReader(BaseReader):
    def __init__(self, phase, data, config):
        super(BartPretrainReader, self).__init__(phase, data, config)
        tokenizer = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not tokenizer:
            logger_output('error', 'tokenizer type wrong or not configured')
            raise ValueError('tokenizer type wrong or not configured')
        self.tokenizer = tokenizer.from_pretrained(self.config.get('tokenizer_path', ''))

    def _ngram_mask(self, input_tokens: List[str], mask_prob=0.2, max_ngram=3):
        """
        Get 0/1 labels for masked tokens with ngram mask proxy
        """
        # n-gram masking Albert
        ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[PAD]" or token == "[UNK]":
                continue
            cand_indexes.append(i)
        random.shuffle(cand_indexes)
        num_to_predict = max(1, int(round(len(cand_indexes) * mask_prob)))
        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            n = np.random.choice(ngrams, p=pvals)
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + n > num_to_predict:
                continue
            if index + n > len(cand_indexes) - 1:
                continue
            for temp_index in range(index, index + n):
                if temp_index in covered_indexes:
                    continue
                if input_tokens[temp_index] == '10':  # 句号不mask
                    break
                masked_lms.append(temp_index)
                covered_indexes.add(temp_index)

        mask_labels = [1 if i in masked_lms else 0 for i in range(len(input_tokens))]
        return mask_labels

    def convert_item(self, each_data):
        infos = each_data.split(',')
        report_ID = infos[0]
        description = infos[1]
        if len(infos) == 3:
            diagnosis = infos[2]
        description = description.split(' ')
        label = copy.deepcopy(description)
        # description进行mask
        masked = self._ngram_mask(description, self.mask_prob, self.max_ngram)

        idx = 0
        masked_description = []
        while idx < len(description):
            if masked[idx] == 0:
                masked_description.append(description[idx])
                idx += 1
                continue
            rand = random.random()
            if rand < 0.8:  # 其中80% span mask
                while idx < len(description) and masked[idx] == 1:
                    idx += 1
                masked_description.append('[MASK]')
            elif rand < 0.9:  # 其中10% span前插入一个[MASK]
                masked_description.append('[MASK]')
                while idx < len(description) and masked[idx] == 1:
                    idx += 1
            else:  # 剩下的10% 删掉
                while idx < len(description) and masked[idx] == 1:
                    idx += 1

        # description进行编码
        tokenize_result = self.tokenizer.encode_plus(text=masked_description, add_special_tokens=True, padding=self.config['padding'],
                                                     truncation=True, max_length=self.config['max_seq_len'], return_tensors='pt')
        input_ids = tokenize_result['input_ids']
        attention_mask = tokenize_result['attention_mask']

        # label进行编码
        label = label + ['[EOS]']
        tokenize_result = self.tokenizer.encode_plus(text=label, add_special_tokens=False, padding=self.config['padding'],
                                                     truncation=True, max_length=self.config['max_seq_len'], return_tensors='pt')
        label = tokenize_result['input_ids']
        label[label == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label,
        }

    def __getitem__(self, index):
        return self.convert_item(self.data[index])

    def __len__(self):
        return len(self.data)


reader = BartPretrainReader
