# -*- encoding: utf-8 -*-
"""
@File    :   ner_bio_model.py
@Time    :   2022/08/08 15:12:39
@Author  :   jiangjiajia
"""
import copy

import numpy as np
import torch

from ..metrics.metric import compute_f1
from ..utils.common_utils import ENCODERS, logger_output
from .base_model import BaseModel


class NERBIOModel(BaseModel):
    def __init__(self, **kwargs):
        super(NERBIOModel, self).__init__(**kwargs)
        encoder = ENCODERS.get(self.encoder.get('type', ''), None)
        if not encoder:
            logger_output('error', 'encoder type wrong or not configured')
            raise ValueError('encoder type wrong or not configured')
        self.encoder = encoder.from_pretrained(self.encoder.get('pretrained_model_dir', ''))
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids=None, phase=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(encoder_outputs)
        if phase == 'train' and label_ids is not None:
            loss = self.criterion(logits.view(-1, self.num_labels), label_ids.view(-1))
            return {
                'loss': loss,
                'logits': logits
            }
        else:
            return {
                'logits': logits
            }

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

    def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
        def extract_result(results, text):
            text = "".join(text)
            ret = []
            entity_name = ""
            flag = []
            visit = False
            start_idx, end_idx = 0, 0
            for i, (char, tag) in enumerate(zip(text, results)):
                tag = tag.item()
                tag = dataset.id2label[tag]
                if tag[0] == "B":
                    if entity_name != "":
                        x = dict((a, flag.count(a)) for a in flag)
                        y = [k for k, v in x.items() if max(x.values()) == v]
                        ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
                        flag.clear()
                        entity_name = ""
                    visit = True
                    start_idx = i
                    entity_name += char
                    flag.append(tag[2:])
                    end_idx = i
                elif tag[0] == "I" and visit:
                    entity_name += char
                    flag.append(tag[2:])
                    end_idx = i
                else:
                    if entity_name != "":
                        x = dict((a, flag.count(a)) for a in flag)
                        y = [k for k, v in x.items() if max(x.values()) == v]
                        ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
                        flag.clear()
                    start_idx = i + 1
                    visit = False
                    flag.clear()
                    entity_name = ""

            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
            return ret

        predictions = []
        idx = 0
        for each_output in forward_output['logits']:  # [max_len, num_labels]
            target_data = copy.deepcopy(dataset.data[idx + batch_start_index])
            idx += 1
            text = target_data['text'][:dataset.config['max_seq_len']]
            preds = np.argmax(each_output, axis=1)  # [max_len]
            preds = preds[1:dataset.config['max_seq_len'] + 1]
            predicts = extract_result(preds, text)
            target_data['entities'] = predicts
            predictions.append(target_data)
        return predictions


model = NERBIOModel
