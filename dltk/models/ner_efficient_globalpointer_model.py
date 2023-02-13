# -*- encoding: utf-8 -*-
"""
@File    :   ner_efficient_globalpointer_model.py
@Time    :   2023/01/29 17:38:53
@Author  :   jiangjiajia
"""
import copy

import numpy as np
import torch

from ..metrics.metric import compute_f1
from ..modules.efficient_globalpointer import EfficientGlobalPointer
from ..modules.global_pointer_crossentropy import GlobalPointerCrossentropy
from ..utils.common_utils import ENCODERS, logger_output
from .base_model import BaseModel


class NEREfficientGlobalPointerModel(BaseModel):
    def __init__(self, **kwargs):
        super(NEREfficientGlobalPointerModel, self).__init__(**kwargs)
        encoder = ENCODERS.get(self.encoder.get('type', ''), None)
        if not encoder:
            logger_output('error', 'encoder type wrong or not configured')
            raise ValueError('encoder type wrong or not configured')
        self.encoder = encoder.from_pretrained(self.encoder.get('pretrained_model_dir', ''))

        # efficient globalpointer
        self.egp = EfficientGlobalPointer(self.encoder.config.hidden_size, self.num_labels, self.hidden_size)

        self.criterion = GlobalPointerCrossentropy(reduction='mean')

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, label_mask=None, phase=None):
        encoder_outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask).last_hidden_state
        # efficient globalpointer
        logits = self.egp(encoder_outputs)
        if phase == 'train' and labels is not None and label_mask is not None:
            logits = logits - (1 - label_mask.unsqueeze(-1).expand(-1, -1, -1, logits.shape[-1])) * 1e12
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
            target_data = copy.deepcopy(dataset.data[batch_start_index + idx])
            idx += 1
            text = target_data['text']
            entities = []
            for start_index, end_index, label_id in zip(*np.where(each_output > 0)):
                start_index = start_index.item() - 1
                end_index = end_index.item() - 1
                label_id = label_id.item()
                if end_index < start_index or start_index > len(text) or end_index > len(text):
                    continue
                entity_label = dataset.id2label[label_id]
                entities.append({
                    'start_idx': start_index,
                    'end_idx': end_index,
                    'type': entity_label,
                    'entity': text[start_index: end_index + 1]
                })
            target_data['entities'] = entities
            predictions.append(target_data)
        return predictions

    def get_metrics(self, phase, predictions, dataset):
        def get_entities(data):
            entities = []
            entities_dict = {}
            for each in data:
                for entity in each['entities']:
                    entities.append((entity['start_idx'], entity['end_idx'], entity['type'], entity['entity']))
                    entities_dict.setdefault(entity['type'], []).append(
                        (entity['start_idx'], entity['end_idx'], entity['type'], entity['entity']))
            return entities, entities_dict

        predictions, predictions_dict = get_entities(predictions)
        targets, targets_dict = get_entities(dataset.data)
        results = {}
        metric = compute_f1(predictions, targets)
        results.update(metric)
        logger_output('info', 'metrics F1:{}'.format(results['F1']))
        return results


model = NEREfficientGlobalPointerModel
