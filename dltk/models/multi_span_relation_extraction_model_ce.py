# -*- encoding: utf-8 -*-
"""
@File    :   multi_span_relation_extraction_model_ce.py
@Time    :   2022/08/12 10:38:49
@Author  :   jiangjiajia
"""
import numpy as np
import torch

from ..metrics.metric import compute_f1
from ..modules.biaffine import Biaffine
from ..utils.common_utils import (ENCODERS, logger_output, numpy_softmax,
                                  write_json)
from .base_model import BaseModel


class MultiSREModel(BaseModel):
    """
    multi span and relation extraction model.
    """

    def __init__(self, **kwargs):
        super(MultiSREModel, self).__init__(**kwargs)
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
        label_weight = torch.FloatTensor([1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 4.0, 3.0])
        self.criterion = torch.nn.CrossEntropyLoss(weight=label_weight, reduction='mean')

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, label_mask=None, phase=None, **kwargs):
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

    def get_metrics(self, phase, predictions, dataset):
        def get_relations(data):
            relations = {}
            entities = set()
            for each_data in data:
                relation_of_mention_list = each_data['relation_of_mention']
                for relation_of_mention in relation_of_mention_list:
                    relation, head, tail = relation_of_mention['relation'], relation_of_mention['head'], relation_of_mention['tail']
                    if relation in [1, 3]:
                        head_start_idx, head_end_idx = head['start_idx'], head['end_idx']
                        tail_start_idx, tail_end_idx = tail['start_idx'], tail['end_idx']
                        relations.setdefault(relation, []).append(
                            ((head_start_idx, head_end_idx), relation, (tail_start_idx, tail_end_idx)))
                        entities.add((head_start_idx, head_end_idx, head['mention']))
                        entities.add((tail_start_idx, tail_end_idx, tail['mention']))
                    else:  # 2
                        head_start_idx, head_end_idx = head['start_idx'], head['end_idx']
                        sub_relation, sub_head, sub_tail = tail['relation'], tail['head'], tail['tail']
                        sub_head_start_idx, sub_head_end_idx = sub_head['start_idx'], sub_head['end_idx']
                        sub_tail_start_idx, sub_tail_end_idx = sub_tail['start_idx'], sub_tail['end_idx']
                        relations.setdefault(relation, []).append(((head_start_idx, head_end_idx), relation,
                                                                   (sub_head_start_idx, sub_head_start_idx), sub_relation, (sub_tail_start_idx, sub_tail_end_idx)))
                        entities.add((head_start_idx, head_end_idx, head['mention']))
                        entities.add((sub_head_start_idx, sub_head_end_idx, sub_head['mention']))
                        entities.add((sub_tail_start_idx, sub_tail_end_idx, sub_tail['mention']))

            return relations, list(entities)
        predictions, pred_entities = get_relations(predictions)
        targets, target_entities = get_relations(dataset.data)
        results = {}
        results['F1'] = 0.
        results['entity_F1'] = compute_f1(pred_entities, target_entities)
        for label, gold in targets.items():
            prediction = predictions.get(label, [])
            label_metric = compute_f1(prediction, gold)
            results[str(label) + '_P'] = label_metric['P']
            results[str(label) + '_R'] = label_metric['R']
            results[str(label) + '_F1'] = label_metric['F1']
            results['F1'] += label_metric['F1']
        results['F1'] /= 3
        logger_output('info', 'metrics F1:{}'.format(results['F1']))
        return results

    def get_predictions(self, forward_output, forward_target, dataset, start_index=0):
        predictions = []
        idx = 0

        for each_output in forward_output['logits']:
            target_data = dataset.data[idx + start_index]
            idx += 1
            text = target_data['text']

            entity_dict = {}
            head2head_set = set()
            tail2tail_set = set()

            probs = {}
            each_output = numpy_softmax(each_output)
            preds = np.argmax(each_output, axis=-1)
            for i in range(1, min(len(text) + 1, dataset.config['max_seq_len'])):
                for j in range(1, min(len(text) + 1, dataset.config['max_seq_len'])):
                    if i == j:
                        continue
                    label = preds[i, j].item()
                    start_index = i - 1
                    end_index = j - 1
                    prob = each_output[i, j, label].item()
                    probs[(start_index, end_index)] = prob
                    label_type, label = dataset.id2label[label].split('_')
                    if label_type == 'EH2ET':
                        if end_index < start_index:
                            continue
                        entity = (text[start_index:end_index + 1], start_index, end_index)
                        entity_dict.setdefault(start_index, []).append(entity)
                    elif label_type == 'H2H':
                        head2head_set.add((start_index, end_index, label))
                    elif label_type == 'T2T':
                        tail2tail_set.add((start_index, end_index, label))
            relations_dict = {}
            for start_index, end_index, label in head2head_set:
                sub_entity_list = entity_dict.get(start_index, [])
                obj_entity_list = entity_dict.get(end_index, [])
                for sub in sub_entity_list:
                    for obj in obj_entity_list:
                        sub_text, sub_start_index, sub_end_index = sub
                        obj_text, obj_start_index, obj_end_index = obj
                        if (sub_end_index, obj_end_index, label) in tail2tail_set:
                            relations_dict.setdefault(label, []).append((sub, label, obj))
            relation_of_mention = []
            # 先看标签为3 上下位关系的
            for (sub, label, obj) in relations_dict.get('3', []):
                temp_result = {
                    'head': {
                        'mention': sub[0],
                        'start_idx': sub[1],
                        'end_idx': sub[2] + 1
                    },
                    'relation': 3,
                    'tail': {
                        'mention': obj[0],
                        'start_idx': obj[1],
                        'end_idx': obj[2] + 1
                    }
                }
                if self.output_probs:
                    temp_prob = 0.
                    temp_prob += probs[(sub[1], sub[2])]
                    temp_prob += probs[(obj[1], obj[2])]
                    temp_prob += probs[(sub[1], obj[1])]
                    temp_prob += probs[(sub[2], obj[2])]
                    temp_result['prob'] = temp_prob / 4
                relation_of_mention.append(temp_result)

            # 再看标签2 条件关系的
            used_relations = set()
            for relation_i in relations_dict.get('2', []):
                for relation_j in relations_dict.get('2', []):
                    if relation_i == relation_j or relation_i[0] != relation_j[0]:
                        continue
                    if (relation_i[2], '1', relation_j[2]) in relations_dict.get('1', []):
                        used_relations.add((relation_i[2], '1', relation_j[2]))
                        temp_result = {
                            'head': {
                                'mention': relation_i[0][0],
                                'start_idx': relation_i[0][1],
                                'end_idx': relation_i[0][2] + 1
                            },
                            'relation': 2,
                            'tail': {
                                'type': 'relation',
                                'head': {
                                    'mention': relation_i[2][0],
                                    'start_idx': relation_i[2][1],
                                    'end_idx': relation_i[2][2] + 1
                                },
                                'relation': 1,
                                'tail': {
                                    'mention': relation_j[2][0],
                                    'start_idx': relation_j[2][1],
                                    'end_idx': relation_j[2][2] + 1
                                }
                            }
                        }
                        if self.output_probs:
                            temp_prob = 0.
                            temp_prob += probs[(relation_i[0][1], relation_i[0][2])]
                            temp_prob += probs[(relation_i[2][1], relation_i[2][2])]
                            temp_prob += probs[(relation_j[2][1], relation_j[2][2])]
                            temp_prob += probs[(relation_i[0][1], relation_i[2][1])]
                            temp_prob += probs[(relation_i[0][2], relation_i[2][2])]
                            temp_prob += probs[(relation_i[0][1], relation_j[2][1])]
                            temp_prob += probs[(relation_i[0][2], relation_j[2][2])]
                            temp_prob += probs[(relation_i[2][1], relation_j[2][1])]
                            temp_prob += probs[(relation_i[2][2], relation_j[2][2])]
                            temp_result['prob'] = temp_prob / 9
                        relation_of_mention.append(temp_result)
            # 最后处理标签1 因果关系的
            for (sub, label, obj) in set(relations_dict.get('1', [])) - used_relations:
                temp_result = {
                    'head': {
                        'mention': sub[0],
                        'start_idx': sub[1],
                        'end_idx': sub[2] + 1
                    },
                    'relation': 1,
                    'tail': {
                        'mention': obj[0],
                        'start_idx': obj[1],
                        'end_idx': obj[2] + 1
                    }
                }
                if self.output_probs:
                    temp_prob = 0.
                    temp_prob += probs[(sub[1], sub[2])]
                    temp_prob += probs[(obj[1], obj[2])]
                    temp_prob += probs[(sub[1], obj[1])]
                    temp_prob += probs[(sub[2], obj[2])]
                    temp_result['prob'] = temp_prob / 4
                relation_of_mention.append(temp_result)
            predictions.append({
                'text': text,
                'relation_of_mention': relation_of_mention
            })
        return predictions


model = MultiSREModel
