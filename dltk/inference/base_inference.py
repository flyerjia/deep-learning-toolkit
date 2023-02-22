# -*- encoding: utf-8 -*-
"""
@File    :   base_inference.py
@Time    :   2022/08/08 19:08:52
@Author  :   jiangjiajia
"""
import torch
from torch.utils.data import DataLoader, SequentialSampler

from ..utils.common_utils import write_json, logger_output


class BaseInference:
    def __init__(self, device, model, **kwargs):
        self.device = device
        self.model = model
        self.kwargs = kwargs

    def service_inference(self, dataset, example):
        dataset.data = [example]
        example_instance = dataset.convert_item(example)
        for name, value in example_instance.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 1:
                    value = torch.unsqueeze(value, 0)
                value = value.to(self.device)
            example_instance[name] = value
        predictions = []
        start_index = 0
        with torch.no_grad():
            example_instance['phase'] = 'inference'
            output = self.model(**example_instance)
            forward_output = {}
            forward_target = {}
            for data_name, data_value in example_instance.items():
                if isinstance(data_value, torch.Tensor):
                    data_value = data_value.detach().cpu().numpy()
                forward_target[data_name] = data_value
            for data_name, data_value in output.items():
                if isinstance(data_value, torch.Tensor):
                    data_value = data_value.detach().cpu().numpy()
                forward_output[data_name] = data_value
            batch_predictions = self.model.get_predictions(forward_output, forward_target, dataset, start_index)
            predictions.extend(batch_predictions)
            start_index += max([forward_target[data_name].shape[0] for data_name in forward_target.keys()])
        prediction = predictions[0]
        example.update(prediction)
        return example

    def inference(self, dataset):
        logger_output('info', 'start inference')
        data_sampler = SequentialSampler(dataset)
        batch_size = self.kwargs.get('batch_size', 1)
        data_loader = DataLoader(dataset=dataset,
                                 sampler=data_sampler,
                                 batch_size=batch_size,
                                 collate_fn=dataset.collate_fn)
        predictions = []
        start_index = 0
        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                for data_name, data_value in batch_data.items():
                    if isinstance(data_value, torch.Tensor):
                        batch_data[data_name] = data_value.to(self.device)
                batch_data['phase'] = 'inference'
                output = self.model(**batch_data)
                forward_output = {}
                forward_target = {}
                for data_name, data_value in batch_data.items():
                    if isinstance(data_value, torch.Tensor):
                        data_value = data_value.detach().cpu().numpy()
                    forward_target[data_name] = data_value
                for data_name, data_value in output.items():
                    if isinstance(data_value, torch.Tensor):
                        data_value = data_value.detach().cpu().numpy()
                    forward_output[data_name] = data_value
                batch_predictions = self.model.get_predictions(forward_output, forward_target, dataset, start_index)
                predictions.extend(batch_predictions)
                start_index += max([forward_target[data_name].shape[0] for data_name in forward_target.keys()])
                logger_output('info', 'batch data {}/{} inference done'.format(step + 1, len(data_loader)))
        logger_output('info', 'inference done')
        write_json(self.kwargs['inference_output'], predictions)


inference = BaseInference
