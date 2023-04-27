# -*- encoding: utf-8 -*-
"""
@File    :   instruction_relation_extraction_model.py
@Time    :   2023/04/25 16:06:19
@Author  :   jiangjiajia
"""
import copy
import re

import torch
from rouge_chinese import Rouge

from ..metrics.metric import compute_f1
from ..utils.common_utils import ENCODERS, logger_output, write_jsonline
from .base_model import BaseModel


class InstructionREModel(BaseModel):
    def __init__(self, **kwargs):
        super(InstructionREModel, self).__init__(**kwargs)
        t5_model = ENCODERS.get(self.encoder.get('type', ''), None)
        if not t5_model:
            logger_output('error', 't5_model type wrong or not configured')
            raise ValueError('t5_model type wrong or not configured')
        self.t5_model = t5_model.from_pretrained(self.encoder.get('pretrained_model_dir', ''))

    def forward(self, input_ids, attention_mask, labels=None, phase=None, **kwargs):
        if phase == 'train' and labels is not None:
            outputs = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return {
                'loss': outputs.loss,
                'logits': outputs.logits
            }
        else:
            return {}

    def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
        def get_kg(output):
            results = []
            kgs = re.finditer('\((.*?),(.*?),(.*?)\)', output)
            for each_kg in kgs:
                results.append([
                    each_kg.group(1),
                    each_kg.group(2),
                    each_kg.group(3)
                ])
            return results

        predictions = []
        idx = 0
        device = next(self.t5_model.parameters()).device

        for input_ids, attention_mask in zip(forward_target['input_ids'], forward_target['attention_mask']):
            input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(device)
            attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).to(device)
            outputs = self.t5_model.generate(input_ids,
                                             attention_mask=attention_mask,
                                             max_length=self.max_length,
                                             early_stopping=True,
                                             use_cache=True,
                                             num_beams=self.num_beams,
                                             length_penalty=self.length_penalty,
                                             return_dict_in_generate=True
                                             )
            each_output = dataset.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            each_data = copy.deepcopy(dataset.data[batch_start_index + idx])
            each_data['pred_output'] = each_output
            each_data['pred_kg'] = get_kg(each_output)
            predictions.append(each_data)
            idx += 1

        return predictions

    def get_metrics(self, phase, predictions, dataset):
        pred_kgs = []
        gold_kgs = []
        pred_output = []
        gold_output = []
        for each_data in predictions:
            pred_kgs.extend(each_data['pred_kg'])
            gold_kgs.extend(each_data['kg'])
            pred_output.append(' '.join([i for i in each_data['pred_output']]))
            gold_output.append(' '.join([i for i in each_data['output']]))

        F1 = compute_f1(pred_kgs, gold_kgs)['F1']
        rouge = Rouge()
        rouge_2 = rouge.get_scores(pred_output, gold_output, avg=True)['rouge-2']['f']
        results = {}
        results['F1'] = F1
        results['rouge_2'] = rouge_2
        results['score'] = 0.5 * F1 + 0.5 * rouge_2
        logger_output('info', 'F1:{}'.format(results['F1']))
        logger_output('info', 'rouge_2:{}'.format(results['rouge_2']))
        logger_output('info', 'score:{}'.format(results['score']))
        return results

    def save(self, save_path, only_save_model_weight=False):
        self.t5_model.save_pretrained(save_path)

    def save_predictions(self, predictions, file_path):
        return write_jsonline(predictions, file_path)


model = InstructionREModel
