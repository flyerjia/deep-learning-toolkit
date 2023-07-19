# -*- encoding: utf-8 -*-
"""
@File    :   cls_classification_model.py
@Time    :   2022/08/12 17:42:29
@Author  :   jiangjiajia
"""
import copy

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

from ..utils.common_utils import ENCODERS, logger_output, numpy_softmax
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
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

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
            idx += 1
            each_output = numpy_softmax(each_output)
            pred = np.argmax(each_output).item()
            target_data['pred_label_prob'] = each_output[pred].item()
            pred = dataset.id2label[pred]
            target_data['pred_label'] = pred
            predictions.append(target_data)
        return predictions

    def get_metrics(self, phase, predictions, dataset):
        temp_results = {'target': [], 'pred': []}
        for prediction, target in zip(predictions, dataset.data):
            temp_results['target'].append(target['label'])
            temp_results['pred'].append(prediction['pred_label'])
        results = {}
        results['F1'] = f1_score(temp_results['target'], temp_results['pred'], average='macro')
        scores = classification_report(temp_results['target'], temp_results['pred'], output_dict=True)
        results['ACC'] = scores['accuracy']
        for key in sorted(scores.keys()):
            if key not in ['accuracy', 'macro avg', 'weighted avg']:
                results[key + '_p'] = scores[key]['precision']
                results[key + '_r'] = scores[key]['recall']
                results[key + '_f1'] = scores[key]['f1-score']

        logger_output('info', 'F1:{}'.format(results['F1']))
        logger_output('info', 'ACC:{}'.format(results['ACC']))
        return results


model = CLSClassificationModel
