# -*- encoding: utf-8 -*-
"""
@File    :   ensembled_bart.py
@Time    :   2023/04/11 17:44:17
@Author  :   jiangjiajia
"""
import copy

import torch

from ..utils.common_utils import ENCODERS, logger_output
from .base_model import BaseModel
from ..metrics.ciderd_evaluator import CiderD
from ..modules.ensembled_bart_for_conditional_generation import EnsembledBartForConditionalGeneration


class EnsembledBartPretrainMdoel(BaseModel):
    def __init__(self, **kwargs):
        super(EnsembledBartPretrainMdoel, self).__init__(**kwargs)
        self.model = EnsembledBartForConditionalGeneration(self.config_path, self.model_path_list, self.model_weight)


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
        input_ids = torch.from_numpy(forward_target['input_ids']).to(device)
        attention_mask = torch.from_numpy(forward_target['attention_mask']).to(device)
        outputs = self.model.generate(input_ids,
                                      attention_mask=attention_mask,
                                      max_length=self.max_length,
                                      early_stopping=True,
                                      use_cache=True,
                                      num_beams=self.num_beams,
                                      length_penalty=self.length_penalty,
                                      return_dict_in_generate=True
                                      )
        outputs = dataset.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        for each_output in outputs:
            each_data = copy.deepcopy(dataset.data[batch_start_index + index])
            infos = each_data.split(',')
            report_ID = infos[0].strip()
            description = infos[1].strip()
            if len(infos) >= 3:
                diagnosis = infos[2].strip()
            else:
                diagnosis = ''
            if len(infos) >= 4:
                clinical = infos[3].strip()
            else:
                clinical = ''
            pred_diagnosis = each_output
            predictions.append({
                'report_ID': report_ID,
                'description': description,
                'diagnosis': diagnosis,
                'clinical': clinical,
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


model = EnsembledBartPretrainMdoel
