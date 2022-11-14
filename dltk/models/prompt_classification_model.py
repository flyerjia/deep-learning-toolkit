# -*- encoding: utf-8 -*-
"""
@File    :   prompt_classification_model.py
@Time    :   2022/08/11 11:41:59
@Author  :   jiangjiajia
"""
import numpy as np
import torch
from sklearn.metrics import classification_report

from ..utils.common_utils import ENCODERS, logger_output, write_json
from .base_model import BaseModel


class PromptClassificationModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        encoder = ENCODERS.get(self.encoder.get('type', ''), None)
        if not encoder:
            logger_output('error', 'encoder type wrong or not configured')
            raise ValueError('encoder type wrong or not configured')
        self.encoder = encoder.from_pretrained(self.encoder.get('pretrained_model_dir', ''))
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        label_weight = torch.FloatTensor([2.0, 1.0, 2.0, 1.0, 1.0])
        self.criterion = torch.nn.CrossEntropyLoss(weight=label_weight, reduction='mean')

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

    def get_predictions(self, forward_output, forward_target, dataset):
        predictions = []
        idx = 0
        for batch_output in forward_output['logits']:
            for each_output in batch_output:
                target_data = dataset.data[idx]
                text = target_data['text']
                question = target_data['question']
                idx += 1
                pred = np.argmax(each_output).item()
                pred = dataset.id2label[pred]
                target_data['pred_answer'] = pred
                predictions.append(target_data)
        return predictions

    def get_metrics(self, phase, forward_output, forward_target, dataset=None):
        predictions = self.get_predictions(forward_output, forward_target, dataset)
        label_results = {}
        for prediction in predictions:
            label = prediction['question']
            label_results.setdefault(label, {'target': [], 'pred': []})
            label_results[label]['target'].append(prediction['answer'])
            label_results[label]['pred'].append(prediction['pred_answer'])
        results = {}
        results['F1'] = 0

        for label, values in label_results.items():
            temp_scores = classification_report(values['target'], values['pred'], output_dict=True)
            results['F1'] += temp_scores['macro avg']['f1-score']
            for key, value in temp_scores.items():
                if key not in ['accuracy', 'macro avg', 'weighted avg']:
                    results[label + '_' + key + '_' + 'P'] = value['precision']
                    results[label + '_' + key + '_' + 'R'] = value['recall']
                    results[label + '_' + key + '_' + 'F1'] = value['f1-score']
            results[label + '_acc'] = temp_scores['accuracy']
        results['F1'] /= len(label_results.keys())
        logger_output('info', 'F1:{}'.format(results['F1']))
        return results

    def save_predictions(self, forward_output, forward_target, dataset, file_path):
        predictions = self.get_predictions(forward_output, forward_target, dataset)
        write_json(file_path, predictions)


model = PromptClassificationModel
