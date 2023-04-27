# -*- encoding: utf-8 -*-
"""
@File    :   bart_model.py
@Time    :   2023/04/05 15:41:52
@Author  :   jiangjiajia
"""
import copy

import torch

from ..utils.common_utils import ENCODERS, logger_output
from .base_model import BaseModel
from ..metrics.evaluate import CiderD
from ..modules.bart import BartForConditionalGeneration


class BartPretrainMdoel(BaseModel):
    def __init__(self, **kwargs):
        super(BartPretrainMdoel, self).__init__(**kwargs)
        model = ENCODERS.get('bart')
        loss_type = kwargs.get('loss_type', 'ce')
        if loss_type == 'lsce':
            self.model = BartForConditionalGeneration.from_pretrained(self.pretrained_model_dir)
        else:
            self.model = model.from_pretrained(self.pretrained_model_dir)


    def forward(self, input_ids, attention_mask, labels=None, phase=None, **kwargs):
        if phase == 'train' and labels is not None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            return {
                'loss': outputs.loss,
                'logits': outputs.logits
            }
        else:
            return {}

    def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
        predictions = []
        index = 0
        device = next(self.model.parameters()).device
        for input_ids, attention_mask in zip(forward_target['input_ids'], forward_target['attention_mask']):
            input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(device)
            attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).to(device)
            outputs = self.model.generate(input_ids,
                                          attention_mask=attention_mask,
                                          max_length=self.max_length,
                                          early_stopping=True,
                                          use_cache=True,
                                          num_beams=self.num_beams,
                                          length_penalty=self.length_penalty,
                                          return_dict_in_generate=True
                                          )
            each_output = dataset.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            each_data = copy.deepcopy(dataset.data[batch_start_index + index])
            infos = each_data.split(',')
            report_ID = infos[0]
            description = infos[1]
            if len(infos) == 3:
                diagnosis = infos[2]
            else:
                diagnosis = ''
            pred_diagnosis = each_output
            predictions.append({
                'report_ID': report_ID,
                'description': description,
                'diagnosis': diagnosis,
                'pred_diagnosis': pred_diagnosis
            })
            index += 1
        return predictions

    def get_metrics(self, phase, predictions, dataset):
        CiderD_scorer = CiderD(df='corpus', sigma=15)
        res, gts = [], {}
        for each_data in predictions:
            res.append({
                'image_id': each_data['report_ID'],
                'caption': [each_data['pred_diagnosis']]
            })
            gts[each_data['report_ID']] = [each_data['diagnosis']]
        cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
        results = {}
        results['cider_score'] = cider_score
        # results['cider_scores'] = cider_scores
        logger_output('info', 'cider_score:{}'.format(results['cider_score']))
        return results

    def save(self, save_path, only_save_model_weight=False):
        self.model.save_pretrained(save_path)


model = BartPretrainMdoel
