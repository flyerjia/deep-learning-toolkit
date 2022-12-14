# -*- encoding: utf-8 -*-
"""
@File    :   oie_generation_model.py
@Time    :   2022/11/14 19:27:28
@Author  :   jiangjiajia
"""
import copy

import torch

from ..metrics.metric import compute_f1
from ..utils.common_utils import ENCODERS, logger_output, write_jsons
from .base_model import BaseModel


class OIEGenerationModel(BaseModel):
    def __init__(self, **kwargs):
        super(OIEGenerationModel, self).__init__(**kwargs)
        t5_model = ENCODERS.get(self.encoder.get('type', ''), None)
        if not t5_model:
            logger_output('error', 't5_model type wrong or not configured')
            raise ValueError('t5_model type wrong or not configured')
        self.t5_model = t5_model.from_pretrained(self.encoder.get('pretrained_model_dir', ''))

    def forward(self, input_ids, attention_mask, labels=None, phase=None, **kwargs):
        outputs = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if phase == 'train' and labels is not None:
            return {
                'loss': outputs.loss,
                'logits': outputs.logits
            }
        else:
            return {
                'logits': outputs.logits
            }

    def get_predictions(self, forward_output, forward_target, dataset, start_index=0):
        predictions = []
        idx = 0
        device = next(self.t5_model.parameters()).device

        batch_input_ids = torch.from_numpy(forward_target['input_ids']).to(device)
        batch_attention_mask = torch.from_numpy(forward_target['attention_mask']).to(device)
        outputs = self.t5_model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_mask,
                                         do_sample=False, max_new_tokens=256)
        outputs = dataset.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for each_output in outputs:
            data = copy.deepcopy(dataset.data[idx + start_index])
            idx += 1
            pred_open_spo_list = []
            spo_list = each_output.split(';')
            for each_spo in spo_list:
                each_spo_result = []
                each_spo = each_spo.strip('()').upper()
                items = each_spo.split(',')
                for each_item in items:
                    if each_item.startswith('S:') and len(each_item[2:]) > 0:
                        each_spo_result.append({
                            'type': 'subject',
                            'text': each_item[2:]
                        })
                    elif each_item.startswith('P:') and len(each_item[2:]) > 0:
                        each_spo_result.append({
                            'type': 'predicate',
                            'text': each_item[2:]
                        })
                    elif each_item.startswith('O:') and len(each_item[2:]) > 0:
                        each_spo_result.append({
                            'type': 'object',
                            'text': each_item[2:]
                        })
                if len(each_spo_result) > 0:
                    pred_open_spo_list.append(each_spo_result)
            data['pred_open_spo_list'] = pred_open_spo_list
            predictions.append(data)
        return predictions

    def get_metrics(self, phase, predictions, dataset):
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

        def get_spo(data, pred):
            results = []
            for each_data in data:
                temp_spo = []
                if pred:
                    spo_info = each_data.get('pred_open_spo_list', [])
                else:
                    spo_info = each_data.get('open_spo_list', [])
                for each_spo_info in spo_info:
                    each_spo_info = sorted(each_spo_info, key=lambda x: x['text'])
                    subjects = get_sub(each_spo_info)
                    temp_spo.extend(subjects)
                    predicates = get_pre(each_spo_info)
                    temp_spo.extend(predicates)
                    objects = get_obj(each_spo_info)
                    temp_spo.extend(objects)
                    results.append(tuple(temp_spo))
            return results
        targets = get_spo(dataset.data, False)
        predictions = get_spo(predictions, True)
        results = compute_f1(predictions, targets)
        logger_output('info', 'F1:{}'.format(results['F1']))
        return results

    def save_predictions(self, predictions, file_path):
        write_jsons(file_path, predictions)


model = OIEGenerationModel
