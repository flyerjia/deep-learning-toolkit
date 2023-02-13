# -*- encoding: utf-8 -*-
"""
@File    :   re_msie_biaffine_model.py
@Time    :   2023/02/01 17:07:00
@Author  :   jiangjiajia
"""
import copy

import numpy as np
import torch

from ..metrics.metric import compute_f1
from ..modules.biaffine import Biaffine
from ..utils.common_utils import (ENCODERS, logger_output, read_jsons)
from .base_model import BaseModel


class REMSIEBiaffineModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        encoder = ENCODERS.get(self.encoder.get('type', ''), None)
        if not encoder:
            logger_output('error', 'encoder type wrong or not configured')
            raise ValueError('encoder type wrong or not configured')
        self.encoder = encoder.from_pretrained(self.encoder.get('pretrained_model_dir', ''))
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

        # 读取Schema，用于预测结果过滤
        schema_info = read_jsons(self.schema_path)
        self.schema = {}
        for each_schema in schema_info:
            subject_type = each_schema['subject_type']
            predicate = each_schema['predicate']
            object_type = each_schema['object_type']
            self.schema.setdefault(predicate, set())
            self.schema[predicate].add(subject_type + '|' + object_type)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, label_mask=None, phase=None):
        encoder_outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask).last_hidden_state
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
            target_data = copy.deepcopy(dataset.data[idx + batch_start_index])
            idx += 1
            text = target_data['text']

            entity_dict = {}
            head2head_set = set()
            tail2tail_set = set()
            preds = np.argmax(each_output, axis=-1)
            for i in range(1, min(len(text) + 1, dataset.config['max_seq_len'])):
                for j in range(1, min(len(text) + 1, dataset.config['max_seq_len'])):
                    if i == j:
                        continue
                    label = preds[i, j].item()
                    start_index = i - 1
                    end_index = j - 1
                    label_type, label = dataset.id2label[label].split('_')
                    if label_type == 'EH2ET':
                        if end_index < start_index:
                            continue
                        entity = (text[start_index:end_index + 1], start_index, end_index, label)
                        entity_dict.setdefault(start_index, []).append(entity)
                    elif label_type == 'H2H':
                        head2head_set.add((start_index, end_index, label))
                    elif label_type == 'T2T':
                        tail2tail_set.add((start_index, end_index, label))

            relations_dict = []
            for start_index, end_index, label in head2head_set:
                sub_entity_list = entity_dict.get(start_index, [])
                obj_entity_list = entity_dict.get(end_index, [])
                for sub in sub_entity_list:
                    for obj in obj_entity_list:
                        sub_text, sub_start_index, sub_end_index, sub_label = sub
                        obj_text, obj_start_index, obj_end_index, obj_label = obj
                        if (sub_end_index, obj_end_index, label) in tail2tail_set:
                            relations_dict.append((sub, label, obj))
            spo_list = []
            for sub, label, obj in relations_dict:
                subject = sub[0]
                subject_type = sub[3]
                predicate = label
                object = obj[0]
                object_type = obj[3]
                if subject_type + '|' + object_type not in self.schema[predicate]:
                    continue
                spo_list.append({
                    'subject': subject,
                    'subject_type': subject_type,
                    'predicate': predicate,
                    'object': {'@value': object},
                    'object_type': {'@value': object_type}
                })
            target_data['spo_list'] = spo_list
            predictions.append(target_data)
        return predictions

    def get_metrics(self, phase, predictions, dataset):
        def get_relations(data):
            relations = []
            entities = []
            for each_data in data:
                spo_list = each_data['spo_list']
                for each_spo in spo_list:
                    subject = each_spo['subject']
                    subject_type = each_spo['subject_type']
                    predicate = each_spo['predicate']
                    object = each_spo['object']['@value']
                    object_type = each_spo['object_type']['@value']
                    entities.append((subject, subject_type))
                    entities.append((object, object_type))
                    relations.append((subject, subject_type, predicate, object, object_type))
            return relations, entities

        predictions, pred_entities = get_relations(predictions)
        targets, target_entities = get_relations(dataset.data)
        results = {}
        results['F1'] = compute_f1(predictions, targets)['F1']
        results['entity_F1'] = compute_f1(pred_entities, target_entities)['F1']
        logger_output('info', 'metrics F1:{}'.format(results['F1']))
        return results


model = REMSIEBiaffineModel
