# -*- encoding: utf-8 -*-
"""
@File    :   bart_model.py
@Time    :   2023/05/04 14:37:26
@Author  :   jiangjiajia
"""
import copy

import torch
import torch.nn.functional as F

from ..utils.common_utils import ENCODERS, logger_output
from .base_model import BaseModel
from ..metrics.ciderd_evaluator import CiderD
from ..metrics.bleu_evaluator import Bleu
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

    def compute_kl_loss(self, p, q, pad_mask=None):

        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, phase=None, **kwargs):
        if phase == 'train' and labels is not None:
            if self.use_rdrop is False:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                     labels=labels, return_dict=True)
                return {
                    'loss': outputs.loss,
                    'logits': outputs.logits
                }
            else:
                outputs1 = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                      labels=labels, return_dict=True)
                outputs2 = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                      labels=labels, return_dict=True)
                pad_mask = torch.zeros_like(labels)
                pad_mask.masked_fill_(labels==-100, 1)
                loss = (outputs1.loss + outputs2.loss) / 2 + self.compute_kl_loss(outputs1.logits, outputs2.logits, pad_mask[:, :, None].bool()) * self.use_rdrop
                return {
                    'loss': loss,
                    'logits': (outputs1.logits + outputs2.logits) / 2
                }

        else:
            return {}

    # def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
    #     predictions = []
    #     index = 0
    #     device = next(self.model.parameters()).device
    #     for input_ids, attention_mask in zip(forward_target['input_ids'], forward_target['attention_mask']):
    #         input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(device)
    #         attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).to(device)
    #         outputs = self.model.generate(input_ids,
    #                                       attention_mask=attention_mask,
    #                                       max_length=self.max_length,
    #                                       early_stopping=True,
    #                                       use_cache=True,
    #                                       num_beams=self.num_beams,
    #                                       length_penalty=self.length_penalty,
    #                                       return_dict_in_generate=True
    #                                       )
    #         each_output = dataset.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    #         each_data = copy.deepcopy(dataset.data[batch_start_index + index])
    #         infos = each_data.split(',')
    #         report_ID = infos[0].strip()
    #         description = infos[1].strip()
    #         if len(infos) >= 3:
    #             diagnosis = infos[2].strip()
    #         else:
    #             diagnosis = ''
    #         if len(infos) >= 4:
    #             clinical = infos[3].strip()
    #         else:
    #             clinical = ''
    #         pred_diagnosis = each_output
    #         predictions.append({
    #             'report_ID': report_ID,
    #             'description': description,
    #             'diagnosis': diagnosis,
    #             'clinical': clinical,
    #             'pred_diagnosis': pred_diagnosis
    #         })
    #         index += 1
    #     return predictions

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
        Bleu_scorer = Bleu(4)
        res, res_bleu, gts = [], {}, {}
        for each_data in predictions:
            res.append({
                'image_id': each_data['report_ID'],
                'caption': [each_data['pred_diagnosis']]
            })
            gts[each_data['report_ID']] = [each_data['diagnosis']]
            res_bleu[each_data['report_ID']] = [each_data['pred_diagnosis']]
        cider_score, cider_scores = CiderD_scorer.compute_score(copy.deepcopy(gts), copy.deepcopy(res))
        bleu_score, bleu_scores = Bleu_scorer.compute_score(copy.deepcopy(gts), copy.deepcopy(res_bleu))
        results = {}
        results['cider_score'] = cider_score
        results['bleu4_score'] = bleu_score[-1]
        results['score'] = (2 * cider_score + 1 * bleu_score[-1]) / 3
        logger_output('info', 'cider_score:{}'.format(results['cider_score']))
        logger_output('info', 'bleu4_score:{}'.format(results['bleu4_score']))
        logger_output('info', 'score:{}'.format(results['score']))
        return results

    def save(self, save_path, only_save_model_weight=False):
        self.model.save_pretrained(save_path)


model = BartPretrainMdoel
