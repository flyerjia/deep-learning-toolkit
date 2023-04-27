# -*- encoding: utf-8 -*-
"""
@File    :   base_controller.py
@Time    :   2022/07/19 11:18:28
@Author  :   jiangjiajia
"""
import copy
import importlib
import os
import time
import shutil
from typing import Dict

import torch
import torch.distributed as dist
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI
from sklearn.model_selection import KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler

from ..utils.common_utils import (OPTIMIZERS, get_device, logger_output, read_text,
                                  read_json, read_jsonline, write_yaml)


class BaseController:
    def __init__(self, rank, gpu_ids, ddp_flag, cmd, config):
        self.rank = rank
        self.gpu_ids = sorted([int(gpu_id) for gpu_id in gpu_ids.split(',') if gpu_id != ''])
        self.ddp_flag = ddp_flag
        self.cmd = cmd
        self.config = config
        self.device = get_device(config.get('use_gpu', False), rank, self.gpu_ids)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()

    def get_data(self, phase, data_path, data_type):
        if not data_path:
            logger_output('error', '{} data path not configured'.format(phase), self.rank)
            raise ValueError('{} data path not configured'.format(phase))
        if data_type == 'jsonl':
            data = read_jsonline(data_path)
        elif data_type == 'json':
            data = read_json(data_path)
        elif data_type == 'text':
            data = read_text(data_path)
        elif data_type == 'images':
            data = load_dataset(data_path, split='train')
        else:
            logger_output('error', 'wrong {} data type or {} data type not configured'.format(phase, phase), self.rank)
            raise ValueError('wrong {} data type or {} data type not configured'.format(phase, phase))
        return data

    def build_dataloader(self, phase, data, reader, reader_config, batch_size, shuffle=False):
        dataset = reader(phase, data, reader_config)
        if shuffle:
            if self.ddp_flag:
                data_sampler = DistributedSampler(dataset)
            else:
                data_sampler = RandomSampler(dataset)
        else:
            data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler,
                                batch_size=batch_size, collate_fn=dataset.collate_fn)
        return {
            'dataset': dataset,
            'sampler': data_sampler,
            'dataloader': dataloader
        }

    def import_reader(self, phase, reader_typer):
        try:
            dataset_reader = importlib.import_module('...readers.' + reader_typer,
                                                     __name__).reader
        except Exception as ex:
            logger_output('error', '{}\'s {} reader not existed or import error'.format(phase, reader_typer), self.rank)
            raise ex
        return dataset_reader

    def init_dataset(self):
        logger_output('info', 'init dataset', self.rank)
        dataset_config = self.config['dataset']
        dataset = {}
        train_dataset_config = dataset_config.get('train', None)
        if train_dataset_config:
            train_dataset_reader_config = train_dataset_config['reader']
            train_dataset_reader = self.import_reader('train', train_dataset_reader_config.get('type', None))

            train_data = self.get_data('trian', train_dataset_config.get('data_path', None),
                                       train_dataset_config.get('type', None))
            logger_output('info', 'Load trian data: {}'.format(len(train_data)), self.rank)

            train_data_batch_size = train_dataset_config.get('batch_size', 12)
            dataset['train'] = self.build_dataloader('train', train_data, train_dataset_reader,
                                                     train_dataset_reader_config, train_data_batch_size, True)
        else:
            dataset['train'] = None
        dev_dataset_config = dataset_config.get('dev', None)
        if dev_dataset_config:
            dev_dataset_reader_config = dev_dataset_config.get('reader', None)
            if not dev_dataset_reader_config:
                dev_dataset_reader_config = copy.deepcopy(train_dataset_config['reader'])
            dev_dataset_reader = self.import_reader('dev', dev_dataset_reader_config.get('type', None))

            data_type = dev_dataset_config.get('type', None)
            if not data_type:
                data_type = copy.deepcopy(train_dataset_config.get('type', None))
            dev_data = self.get_data('dev', dev_dataset_config.get('data_path', None), data_type)
            logger_output('info', 'Load dev data: {}'.format(len(dev_data)), self.rank)

            dev_data_batch_size = dev_dataset_config.get('batch_size', None)
            if not dev_data_batch_size:
                dev_data_batch_size = copy.deepcopy(train_dataset_config.get('batch_size', 12))
            dataset['dev'] = self.build_dataloader('dev', dev_data, dev_dataset_reader,
                                                   dev_dataset_reader_config, dev_data_batch_size, False)
        else:
            dataset['dev'] = None
        test_dataset_config = dataset_config.get('test', None)
        if test_dataset_config:
            test_dataset_reader_config = test_dataset_config.get('reader', None)
            if not test_dataset_reader_config:
                test_dataset_reader_config = copy.deepcopy(train_dataset_config['reader'])
            test_dataset_reader = self.import_reader('test', test_dataset_reader_config.get('type', None))

            data_type = test_dataset_config.get('type', None)
            if not data_type:
                data_type = copy.deepcopy(train_dataset_config.get('type', None))
            test_data = self.get_data('test', test_dataset_config.get('data_path', None), data_type)
            logger_output('info', 'Load test data: {}'.format(len(test_data)), self.rank)

            test_data_batch_size = test_dataset_config.get('batch_size', None)
            if not test_data_batch_size:
                test_data_batch_size = copy.deepcopy(train_dataset_config.get('batch_size', 12))
            dataset['test'] = self.build_dataloader('test', test_data, test_dataset_reader,
                                                    test_dataset_reader_config, test_data_batch_size, False)
        else:
            dataset['test'] = None
        if self.ddp_flag:
            dist.barrier()
        return dataset

    def init_dataset_kfold(self):
        logger_output('info', 'init k_fold dataset', self.rank)
        k_fold = self.config['trainer'].get('k_fold', None)
        if not k_fold:
            logger_output('error', 'k_fold not configured')
            raise ValueError('k_fold not configured')
        dataset_config = self.config['dataset']

        dataset = {}
        train_dataset_config = dataset_config.get('train', None)
        if not train_dataset_config:
            logger_output('error', 'train data configs not exit', self.rank)
            raise ValueError('train data configs not exit')
        dev_dataset_config = dataset_config.get('dev', None)
        if dev_dataset_config:
            logger_output('warning', 'training_cv: dev data should not exit', self.rank)
        test_dataset_config = dataset_config.get('test', None)
        if test_dataset_config:
            test_dataset_reader_config = test_dataset_config.get('reader', None)
            if not test_dataset_reader_config:
                test_dataset_reader_config = copy.deepcopy(train_dataset_config['reader'])
            test_dataset_reader = self.import_reader('test', test_dataset_reader_config.get('type', None))

            data_type = test_dataset_config.get('type', None)
            if not data_type:
                data_type = copy.deepcopy(train_dataset_config.get('type', None))
            test_data = self.get_data('test', test_dataset_config.get('data_path', None), data_type)
            logger_output('info', 'Load test data: {}'.format(len(test_data)), self.rank)

            test_data_batch_size = test_dataset_config.get('batch_size', None)
            if not test_data_batch_size:
                test_data_batch_size = copy.deepcopy(train_dataset_config.get('batch_size', 12))
            dataset['test'] = self.build_dataloader('test', test_data, test_dataset_reader,
                                                    test_dataset_reader_config, test_data_batch_size, False)
        else:
            dataset['test'] = None

        train_dataset_reader_config = train_dataset_config['reader']
        train_dataset_reader = self.import_reader('trian', train_dataset_reader_config.get('type', None))

        train_data = self.get_data('train', train_dataset_config.get('data_path', None),
                                   train_dataset_config.get('type', None))
        logger_output('info', 'Load trian data: {}'.format(len(train_data)), self.rank)

        train_data_batch_size = train_dataset_config.get('batch_size', 12)
        # k_fold
        kfolds = KFold(n_splits=k_fold, shuffle=True, random_state=self.config.get('random_seed', 2022))
        for k, (train_indexs, dev_indexs) in enumerate(kfolds.split(train_data)):
            temp_train_data = [train_data[index] for index in train_indexs]
            temp_dev_data = [train_data[index] for index in dev_indexs]
            if self.ddp_flag:
                dist.barrier()
            dataset['train'] = self.build_dataloader('train', temp_train_data, train_dataset_reader,
                                                     train_dataset_reader_config, train_data_batch_size, True)
            dataset['dev'] = self.build_dataloader('dev', temp_dev_data, train_dataset_reader,
                                                   train_dataset_reader_config, train_data_batch_size, False)
            logger_output('info', 'KFold {}: train data {}  dev data {}'.format(
                str(k), len(temp_train_data), len(temp_dev_data)), self.rank)
            if self.ddp_flag:
                dist.barrier()
            yield k + 1, dataset

    def init_dataset_inference(self):
        logger_output('info', 'init dataset', self.rank)
        dataset_config = self.config.get('dataset', None)
        if dataset_config:
            dataset_reader_config = dataset_config['reader']
            dataset_reader = self.import_reader('inference', dataset_reader_config.get('type', None))
            data_path = dataset_config.get('data_path', None)
            if data_path:
                inference_data = self.get_data('inference', data_path, dataset_config.get('type', None))
            else:
                inference_data = None
                logger_output('warning', 'wrong data type or data path not configured', self.rank)
            dataset = dataset_reader('inference', inference_data, dataset_reader_config)
            return dataset
        else:
            logger_output('info', 'inference dataset config not configured', self.rank)
            raise ValueError('inference dataset config not configured')

    def init_model(self, datasets):
        logger_output('info', 'init model', self.rank)
        model_config = self.config['model']
        model_type = model_config.get('type', None)
        if model_type:
            try:
                model = importlib.import_module('...models.' + model_type, __name__).model
            except Exception as ex:
                logger_output('error', '{} model not existed or import error'.format(model_type), self.rank)
                raise ex
            model_configs = model_config | {'datasets': datasets}
            model = model(**model_configs)
            # 加载权重
            weight_path = model_config.get('weight_path', None)
            if weight_path:
                logger_output('info', 'load weights from [{}]'.format(weight_path), self.rank)
                weights = torch.load(weight_path, map_location='cpu')
                if isinstance(weights, Dict):
                    load_model_info = model.load_state_dict(weights)
                elif isinstance(weights, torch.nn.Module):
                    load_model_info = model.load_state_dict(weights.state_dict())
                else:
                    logger_output('error', 'weights can not been loaded', self.rank)
                    raise ValueError('weights can not been loaded')
                logger_output('warning', 'missing_keys: {}'.format(load_model_info.missing_keys), self.rank)
                logger_output('warning', 'unexpected_keys: {}'.format(load_model_info.unexpected_keys), self.rank)
            model = model.to(self.device)
            if self.ddp_flag:
                model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
                dist.barrier()
            return model
        else:
            logger_output('error', 'model type not configured', self.rank)
            raise ValueError('model type not configured')

    def init_inference_model(self):
        model_config = self.config.get('model', None)
        inference_model_path = self.config.get('inference_model', None)
        if model_config and not inference_model_path:
            model = self.init_model(datasets=None)
        elif inference_model_path and not model_config:
            model = torch.load(inference_model_path, map_location='cpu')
            model = model.to(self.device)
        else:
            logger_output('error', 'model and inference_model both not configured')
            raise ValueError('model and inference_model both not configured')
        return model

    def init_optimizer(self, train_dataset, model):
        logger_output('info', 'init optimizer', self.rank)
        optimizer_config = self.config['optimizer']

        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        warmup = optimizer_config.get('warmup', 0.1)
        encoder_lr = optimizer_config.get('encoder_lr', None)
        if encoder_lr is not None:
            # 分层学习率
            encoder_modules_parameters = []
            other_modules_parameters = []
            for name, parameter in list(model.named_parameters()):
                if name.startswith(optimizer_config.get('encoder_name', 'encoder')):
                    encoder_modules_parameters.append((name, parameter))
                else:
                    other_modules_parameters.append((name, parameter))
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in encoder_modules_parameters if not any(nd in n for nd in no_decay)],
                 'weight_decay': optimizer_config['weight_decay'], 'lr': float(encoder_lr)},
                {'params': [p for n, p in encoder_modules_parameters if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': float(encoder_lr)},
                {'params': [p for n, p in other_modules_parameters if not any(nd in n for nd in no_decay)],
                 'weight_decay': optimizer_config['weight_decay'], 'lr': float(lr)},
                {'params': [p for n, p in other_modules_parameters if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': float(lr)}
            ]
        else:
            optimizer_grouped_parameters = model.parameters()

        optimizer = OPTIMIZERS[optimizer_config['type']](optimizer_grouped_parameters,
                                                         lr=float(lr),
                                                         weight_decay=weight_decay)

        scheduler_type = optimizer_config.get('lr_scheduler', 'linear')
        tol_steps = len(train_dataset['dataloader']) * self.config['trainer']['epoch']
        warmup_steps = int(tol_steps * warmup)
        scheduler = get_scheduler(name=scheduler_type, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=tol_steps)

        if self.ddp_flag:
            dist.barrier()
        return optimizer, scheduler

    def init_trainer(self, model, optimizer, scheduler):
        logger_output('info', 'init trainer', self.rank)
        trainer_config = self.config['trainer']
        trainer_type = trainer_config.get('type', 'base_trainer')
        try:
            trainer = importlib.import_module('...trainer.' + trainer_type, __name__).trainer
        except Exception as ex:
            logger_output('error', '{} not existed or import error'.format(trainer_type), self.rank)
            raise ex
        trainer = trainer(self.rank, self.ddp_flag, model, optimizer, scheduler, self.device, **trainer_config)
        if self.ddp_flag:
            dist.barrier()
        return trainer

    def training(self):
        dataset = self.init_dataset()
        model = self.init_model(datasets=dataset)
        optimizer, scheduler = self.init_optimizer(dataset['train'], model)
        trainer = self.init_trainer(model, optimizer, scheduler)
        trainer.train_and_eval(dataset['train'], dataset['dev'], dataset['test'])
        logger_output('info', 'all done', self.rank)

    def training_cv(self):
        for k, dataset in self.init_dataset_kfold():
            model = self.init_model(datasets=dataset)
            optimizer, scheduler = self.init_optimizer(dataset['train'], model)
            trainer = self.init_trainer(model, optimizer, scheduler)
            logger_output('info', 'start training_cv {}/{}'.format(k, self.config['trainer']['k_fold']), self.rank)
            trainer.train_and_eval(dataset['train'], dataset['dev'], dataset['test'], info=str(k))
            logger_output('info', 'training_cv {}/{} done'.format(k, self.config['trainer']['k_fold']), self.rank)
        logger_output('info', 'all done', self.rank)

    def init_inference(self, model):
        logger_output('info', 'init inferencer', self.rank)
        inference_config = self.config['inference']
        inference_type = inference_config.get('type', 'base_inference')
        try:
            inference = importlib.import_module('...inference.' + inference_type, __name__).inference
        except Exception as ex:
            logger_output('error', '{} not existed or import error'.format(inference_type), self.rank)
            raise ex
        inference = inference(self.device, model, **inference_config)
        return inference

    def inference(self):
        dataset = self.init_dataset_inference()
        model = self.init_inference_model()
        inference = self.init_inference(model)
        inference.inference(dataset)
        logger_output('info', 'all done', self.rank)

    def service(self):
        dataset = self.init_dataset_inference()
        model = self.init_inference_model()
        inference = self.init_inference(model)
        input_data_type = self.config['inference'].get('input_data_type', 'base_input')
        try:
            InputData = importlib.import_module('...service_input.' + input_data_type, __name__).input_data
        except Exception as ex:
            logger_output('error', '{} not existed or import error'.format(input_data_type), self.rank)
            raise ex

        app = FastAPI()
        port = self.config.get('port', 8080)
        url = self.config.get('url', '/innerapi/ai/python/agent')

        @app.get("/health.json")
        def health():
            return {"status": "UP"}

        @app.post(url)
        async def default_interface(input_data: InputData):
            try:
                logger_output('info', 'intput data:{}'.format(input_data.dict()))
                result = inference.service_inference(dataset, input_data.dict())
                logger_output('info', 'output data:{}'.format(result))
            except Exception as ex:
                return {
                    'code': -1,
                    'msg': ex
                }
            return {
                'code': 0,
                'msg': 'OK',
                'data': result
            }
        uvicorn.run(app, host='0.0.0.0', port=port)

    def predict(self, example):
        """
        预测一条数据，用于将框架嵌入到其他项目中
        """
        dataset = self.init_dataset_inference()
        model = self.init_inference_model()
        inference = self.init_inference(model)
        result = inference.service_inference(dataset, example)
        return result

    def run(self):
        # 把config复制一遍到输出目录中，用作查看提醒
        if 'trainer' in self.config and self.rank == 0:
            output_path = self.config['trainer'].get('output_path', 'output')
            if os.path.exists(output_path):
                time_ = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                output_path_bk = output_path + '_' + time_ + '_bk'
                shutil.move(output_path, output_path_bk)
            os.makedirs(output_path)
            write_yaml(os.path.join(output_path, 'config_bk.yaml'), self.config)
        cmd_func = getattr(self, self.cmd)
        cmd_func()


controller = BaseController
