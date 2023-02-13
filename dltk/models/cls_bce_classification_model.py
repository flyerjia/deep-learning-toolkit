# -*- encoding: utf-8 -*-
"""
@File    :   cls_bce_classification_model.py
@Time    :   2022/09/05 15:19:12
@Author  :   jiangjiajia
"""
import copy

import numpy as np
import torch

from ..utils.common_utils import (ENCODERS, logger_output, numpy_sigmoid,
                                  write_json)
from .base_model import BaseModel


class CLSClassificationModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        encoder = ENCODERS.get(self.encoder.get('type', ''), None)
        if not encoder:
            logger_output('error', 'encoder type wrong or not configured')
            raise ValueError('encoder type wrong or not configured')
        self.encoder = encoder.from_pretrained(self.encoder.get('pretrained_model_dir', ''))
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, phase=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask).last_hidden_state
        cls_encoder = encoder_outputs[:, 0]
        logits = self.classifier(cls_encoder)
        if phase == 'train' and labels is not None:
            loss = self.criterion(logits, labels)
            return {
                'loss': loss,
                'logits': logits
            }
        else:
            return {
                'logits': logits
            }

    def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
        predictions = []
        idx = 0
        for each_output in forward_output['logits']:
            target_data = copy.deepcopy(dataset.data[idx + batch_start_index])
            target_data['pred_label'] = []
            target_data['pred_label_prob'] = []
            idx += 1
            each_output = numpy_sigmoid(each_output)
            pred_idx = np.where(each_output > 0.5)[0]
            for pred in pred_idx:
                pred_id = pred.item()
                pred = dataset.id2label[pred_id]
                target_data['pred_label'].append(pred)
                target_data['pred_label_prob'].append(each_output[pred_id].item())
            # 默认在大于0.5中取概率最高的那个
            # import pdb;pdb.set_trace()
            # max_idx = -1
            # max_prob = -1
            # for index, label_prob in enumerate(target_data['pred_label_prob']):
            #     if label_prob >= max_prob:
            #         max_prob = label_prob
            #         max_idx = index
            # if max_idx != -1:
            #     target_data['pred_label'] = [target_data['pred_label'][max_idx]]
            #     target_data['pred_label_prob'] = [target_data['pred_label_prob'][max_idx]]

            if len(target_data['pred_label']) == 0:
                target_data['pred_label'].append('其他证件')
            target_data['pred_label'] = ','.join(target_data['pred_label'])
            predictions.append(target_data)
        return predictions

    def get_metrics(self, phase, predictions, dataset):
        label_result_dict = {}
        num_all = 0
        num_correct = 0
        for prediction in predictions:
            num_all += 1
            target_labels = prediction['label'].split(',')
            pred_labels = prediction['pred_label'].split(',')
            for pred_label in pred_labels:
                label_result_dict.setdefault(pred_label, {'pred_num': 0, 'target_num': 0, 'correct_num': 0})
                label_result_dict[pred_label]['pred_num'] += 1
                if pred_label in target_labels:
                    num_correct += 1
            for target_label in target_labels:
                label_result_dict.setdefault(target_label, {'pred_num': 0, 'target_num': 0, 'correct_num': 0})
                label_result_dict[target_label]['target_num'] += 1
            for temp_label in set(target_labels) & set(pred_labels):
                label_result_dict[temp_label]['correct_num'] += 1

        label_result_dict = {key: value for key, value in sorted(label_result_dict.items(), key=lambda x: x[0])}
        results = {}
        results['F1'] = 0.
        results['ACC'] = num_correct / num_all
        for label, values in label_result_dict.items():
            p = values['correct_num'] / values['pred_num'] if values['pred_num'] != 0 else 0.
            r = values['correct_num'] / values['target_num'] if values['target_num'] != 0 else 0.
            f1 = 2 * p * r / (p + r) if p + r != 0 else 0.
            results[label + '_cnum'] = values['correct_num']
            results[label + '_pnum'] = values['pred_num']
            results[label + '_tnum'] = values['target_num']
            results[label + '_P'] = p
            results[label + '_R'] = r
            results[label + '_F1'] = f1
            results['F1'] += f1
        results['F1'] /= len(label_result_dict.keys())
        logger_output('info', 'F1:{}'.format(results['F1']))
        return results


model = CLSClassificationModel
