# -*- encoding: utf-8 -*-
"""
@File    :   donut_model.py
@Time    :   2023/02/21 15:25:02
@Author  :   jiangjiajia
"""
import re
import json
import copy

import torch
from transformers import VisionEncoderDecoderConfig

from ..metrics.json_parse_evaluator import JSONParseEvaluator
from ..utils.common_utils import ENCODERS, logger_output
from .base_model import BaseModel


class DonutModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        donut_model = ENCODERS.get(self.encoder.get('type', ''), None)
        if not donut_model:
            logger_output('error', 'donut_model type wrong or not configured')
            raise ValueError('donut_model type wrong or not configured')
        config = VisionEncoderDecoderConfig.from_pretrained(self.encoder.get('pretrained_model_dir', ''))
        config.encoder.image_size = self.image_size
        config.decoder.max_length = self.max_length
        self.donut_model = donut_model.from_pretrained(self.encoder.get('pretrained_model_dir', ''), config=config)
        if 'train' in self.datasets:  # resize model embeddings
            train_dataset = self.datasets['train']['dataset']
            processor = train_dataset.processor
            self.donut_model.decoder.resize_token_embeddings(len(processor.tokenizer))
            self.donut_model.config.pad_token_id = processor.tokenizer.pad_token_id
            self.donut_model.config.decoder_start_token_id = train_dataset.decoder_start_token_id

    def forward(self, input_tensor, input_ids, labels=None, answers=None, phase=None, **kwarg):
        if phase == 'train' and labels is not None:
            outputs = self.donut_model(pixel_values=input_tensor, labels=labels)
            return {
                'loss': outputs.loss,
                'logits': outputs.logits
            }
        else:
            return {}

    def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
        predictions = []
        index = 0
        device = next(self.donut_model.parameters()).device

        for input_tensor, input_ids, answer in zip(forward_target['input_tensor'], forward_target['input_ids'], forward_target['answers']):
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
            input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(device)
            outputs = self.donut_model.generate(input_tensor,
                                                decoder_input_ids=input_ids,
                                                max_length=self.max_length,
                                                early_stopping=True,
                                                pad_token_id=dataset.processor.tokenizer.pad_token_id,
                                                eos_token_id=dataset.processor.tokenizer.eos_token_id,
                                                use_cache=True,
                                                num_beams=1,
                                                bad_words_ids=[[dataset.processor.tokenizer.unk_token_id]],
                                                return_dict_in_generate=True,
                                                )
            each_output = dataset.processor.batch_decode(outputs.sequences)[0]

            logger_output('info', 'target: {}'.format(answer))
            logger_output('info', 'prediction: {}'.format(each_output))

            data = copy.deepcopy(dataset.data[index + batch_start_index])
            each_output = each_output.replace(dataset.processor.tokenizer.eos_token, '').replace(dataset.processor.tokenizer.pad_token, '')
            each_output = re.sub(r"<.*?>", '', each_output, count=1).strip()  # remove first task start token
            answer = answer.replace(dataset.processor.tokenizer.eos_token, '').replace(dataset.processor.tokenizer.pad_token, '')
            predictions.append({
                'name': data['name'],
                'target': dataset.processor.token2json(answer),
                'prediction': dataset.processor.token2json(each_output)
            })
            index += 1
        return predictions

    def get_metrics(self, phase, predictions, dataset):
        results = {}
        targets = []
        preds = []
        for each_data in predictions:
            target = each_data['target']
            prediction = each_data['prediction']
            targets.append(target)
            preds.append(prediction)
        evaluator = JSONParseEvaluator()
        F1 = evaluator.cal_f1(preds, targets)
        results['F1'] = F1
        logger_output('info', 'F1:{}'.format(results['F1']))
        return results


model = DonutModel
