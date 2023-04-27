# -*- encoding: utf-8 -*-
"""
@File    :   ner_crf_model.py
@Time    :   2023/01/19 16:03:28
@Author  :   jiangjiajia
"""
import copy

import numpy as np
import torch

from ..metrics.metric import compute_f1
from ..modules.crf import CRF
from ..utils.common_utils import ENCODERS, logger_output
from .base_model import BaseModel


class NERCRFModel(BaseModel):
    def __init__(self, **kwargs):
        super(NERCRFModel, self).__init__(**kwargs)
        encoder = ENCODERS.get(self.encoder.get('type', ''), None)
        if not encoder:
            logger_output('error', 'encoder type wrong or not configured')
            raise ValueError('encoder type wrong or not configured')
        self.encoder = encoder.from_pretrained(self.encoder.get('pretrained_model_dir', ''))
        self.dropout = torch.nn.Dropout(self.dropout)
        self.lstm = torch.nn.LSTM(self.encoder.config.hidden_size, self.encoder.config.hidden_size // 2,
                                  bidirectional=True, batch_first=True, bias=False)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids=None, phase=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask).last_hidden_state
        encoder_outputs = self.dropout(encoder_outputs)
        encoder_outputs, _  = self.lstm(encoder_outputs)
        logits = self.classifier(encoder_outputs)
        if phase == 'train' and label_ids is not None:
            loss = -1.0 * self.crf(logits, label_ids, attention_mask.byte(), reduction='mean')
            return {
                'loss': loss,
                'logits': logits
            }
        else:
            return {
                'logits': logits
            }

    def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
        def extract_result(results, text):
            text = "".join(text)
            ret = []
            entity_name = ""
            flag = []
            visit = False
            start_idx, end_idx = 0, 0
            for i, (char, tag) in enumerate(zip(text, results)):
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
        label_ids = self.crf.decode(torch.from_numpy(forward_output['logits']).to(next(self.crf.parameters()).device),
                                    torch.from_numpy(forward_target['attention_mask']).byte().to(next(self.crf.parameters()).device))
        idx = 0
        for preds in label_ids:
            target_data = copy.deepcopy(dataset.data[idx + batch_start_index])
            idx += 1
            text = target_data['text'][:dataset.max_seq_len - 2]
            preds = preds[1:-1]
            predicts = extract_result(preds, text)
            target_data['entities'] = predicts
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


model = NERCRFModel
