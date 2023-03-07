# -*- encoding: utf-8 -*-
"""
@File    :   donut_reader.py
@Time    :   2023/02/21 15:24:25
@Author  :   jiangjiajia
"""
import json
import random
from typing import Any, List, Tuple

import torch
from tqdm import tqdm

from ..utils.common_utils import TOKENIZERS, logger_output, read_text
from .base_reader import BaseReader


class DonutReader(BaseReader):
    def __init__(self, phase, data, config):
        super(DonutReader, self).__init__(phase, data, config)
        processor = TOKENIZERS.get(self.config.get('tokenizer', ''), None)
        if not processor:
            logger_output('error', 'donut processor wrong or not configured')
            raise ValueError('donut processor wrong or not configured')
        self.processor = processor.from_pretrained(self.processor_path)
        self.processor.image_processor.size = self.image_size[::-1]  # should be (width, height)
        self.processor.image_processor.do_align_long_axis = False
        self.ignore_id = -100

        self.prompt_end_token = self.prompt_end_token if self.prompt_end_token else self.task_start_token

        self.added_tokens = read_text(self.add_tokens_path)
        self.gt_token_sequences = []
        if self.data and len(self.data) > 0:
            for sample in tqdm(self.data):
                ground_truth = json.loads(sample["ground_truth"])
                if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                    assert isinstance(ground_truth["gt_parses"], list)
                    gt_jsons = ground_truth["gt_parses"]
                else:
                    assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                    gt_jsons = [ground_truth["gt_parse"]]
                self.gt_token_sequences.append(
                    [
                        self.json2token(
                            gt_json,
                            update_special_tokens_for_json_key=self.phase == "train",
                            sort_json_key=self.sort_json_key,
                        )
                        + self.processor.tokenizer.eos_token
                        for gt_json in gt_jsons  # load json from list of json
                    ]
                )

        self.add_tokens(self.added_tokens + [self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.task_start_token)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.added_tokens.extend([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        list_of_tokens = list(set(list_of_tokens))
        self.processor.tokenizer.add_tokens(list_of_tokens)

    def __getitem__(self, idx: int):
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        """
        sample = self.data[idx]
        # inputs
        pixel_values = self.processor(sample["image"].convert("RGB"), random_padding=self.phase == "train", return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        if self.phase == 'train':
            # targets
            input_ids = self.processor.tokenizer(target_sequence,
                                                 add_special_tokens=False,
                                                 max_length=self.max_length,
                                                 padding="max_length",
                                                 truncation=True,
                                                 return_tensors="pt",
                                                 )["input_ids"].squeeze(0)
            labels = input_ids.clone()
            # model doesn't need to predict pad token
            labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id
            # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return {
                'input_tensor': pixel_values,
                'input_ids': input_ids,
                'labels': labels,
                'answers': target_sequence
            }
        else:
            input_ids = self.processor.tokenizer(self.task_start_token,
                                                 add_special_tokens=False,
                                                 return_tensors="pt",
                                                 )["input_ids"].squeeze(0)
            labels = input_ids.clone()
            # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return {
                'input_tensor': pixel_values,
                'input_ids': input_ids,
                'labels': labels,
                'answers': target_sequence
            }

    def collate_fn(self, batch_data):
        input_tensor = torch.stack([each_data['input_tensor'] for each_data in batch_data], dim=0)
        input_ids = torch.stack([each_data['input_ids'] for each_data in batch_data], dim=0)
        labels = torch.stack([each_data['labels'] for each_data in batch_data], dim=0)
        answers = [each_data['answers'] for each_data in batch_data]
        return {
            'input_tensor': input_tensor,
            'input_ids': input_ids,
            'labels': labels,
            'answers': answers
        }


reader = DonutReader
