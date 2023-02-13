# -*- encoding: utf-8 -*-
"""
@File    :   ner_biaffine_model.py
@Time    :   2023/01/29 15:23:48
@Author  :   jiangjiajia
"""
import copy

import numpy as np
import torch

from ..metrics.metric import compute_f1
from ..modules.biaffine import Biaffine
from ..modules.focal_loss import MultiFocalLoss
from ..utils.common_utils import ENCODERS, logger_output
from .base_model import BaseModel


class NERBiaffineModel(BaseModel):
    def __init__(self, **kwargs):
        super(NERBiaffineModel, self).__init__(**kwargs)
        encoder = ENCODERS.get(self.encoder.get('type', ''), None)
        if not encoder:
            logger_output('error', 'encoder type wrong or not configured')
            raise ValueError('encoder type wrong or not configured')
        self.encoder = encoder.from_pretrained(self.encoder.get('pretrained_model_dir', ''))

        # biaffine
        self.query_layer = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=self.encoder.config.hidden_size,
                            out_features=self.hidden_size),
            torch.nn.GELU()
        )
        self.key_layer = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=self.encoder.config.hidden_size,
                            out_features=self.hidden_size),
            torch.nn.GELU()
        )
        self.biaffine = Biaffine(self.hidden_size, self.num_labels)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.criterion = MultiFocalLoss(self.num_labels, reduction='mean')

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, label_mask=None, phase=None):
        encoder_outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask).last_hidden_state
        # biaffine
        query = self.query_layer(encoder_outputs)
        key = self.key_layer(encoder_outputs)
        logits = self.biaffine(query, key)
        if phase == 'train' and labels is not None and label_mask is not None:
            logits = logits[label_mask == 1].view(-1, self.num_labels)
            labels = labels[label_mask == 1].view(-1)
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
            preds = np.argmax(each_output, axis=-1)
            for i in range(1, min(len(text) + 1, dataset.config['max_seq_len'] - 1)):  # 过滤CLS和SEP
                for j in range(i, min(len(text) + 1, dataset.config['max_seq_len'] - 1)):
                    start_index = i - 1
                    end_index = j - 1
                    label_id = preds[i, j].item()
                    if label_id == 0:
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


model = NERBiaffineModel
