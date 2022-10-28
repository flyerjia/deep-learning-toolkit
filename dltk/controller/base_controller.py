# -*- encoding: utf-8 -*-
"""
@File    :   base_controller.py
@Time    :   2022/07/19 11:18:28
@Author  :   jiangjiajia
"""

import os
import copy
import importlib
import logging
from typing import Dict

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from ..utils.common_utils import OPTIMIZERS, read_json, read_jsons, write_yaml

logger = logging.getLogger(__name__)


class BaseController:
    def __init__(self, cmd, config):
        self.cmd = cmd
        self.config = config

    def get_data(self, phase, data_path, data_type):
        if not data_path:
            logger.error('{} data path not configured'.format(phase))
            raise ValueError('{} data path not configured'.format(phase))
        if data_type == 'text':
            data = read_jsons(data_path)
        elif data_type == 'json':
            data = read_json(data_path)
        else:
            logger.error('wrong {} data type or {} data type not configured'.format(phase, phase))
            raise ValueError('wrong {} data type or {} data type not configured'.format(phase, phase))
        return data

    def build_dataloader(self, phase, data, reader, reader_config, batch_size, shuffle=False):
        dataset = reader(phase, data, reader_config)
        if shuffle:
            data_sampler = RandomSampler(dataset)
        else:
            data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler,
                                batch_size=batch_size, collate_fn=dataset.collate_fn)
        return {
            'dataset': dataset,
            'dataloader': dataloader
        }

    def import_reader(self, phase, reader_typer):
        try:
            dataset_reader = importlib.import_module('...readers.' + reader_typer,
                                                     __name__).reader
        except Exception as ex:
            logger.error('{}\'s {} reader not existed or import error'.format(phase, reader_typer))
            raise ex
        return dataset_reader

    def init_dataset(self):
        logger.info('init dataset')
        dataset_config = self.config['dataset']
        dataset = {}
        train_dataset_config = dataset_config.get('train', None)
        if train_dataset_config:
            train_dataset_reader_config = train_dataset_config['reader']
            train_dataset_reader = self.import_reader('train', train_dataset_reader_config.get('type', None))

            train_data = self.get_data('trian', train_dataset_config.get('data_path', None),
                                       train_dataset_config.get('type', None))
            logger.info('Load trian data: {}'.format(len(train_data)))

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
            logger.info('Load dev data: {}'.format(len(dev_data)))

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
            logger.info('Load test data: {}'.format(len(test_data)))

            test_data_batch_size = test_dataset_config.get('batch_size', None)
            if not test_data_batch_size:
                test_data_batch_size = copy.deepcopy(train_dataset_config.get('batch_size', 12))
            dataset['test'] = self.build_dataloader('test', test_data, test_dataset_reader,
                                                    test_dataset_reader_config, test_data_batch_size, False)
        else:
            dataset['test'] = None
        return dataset

    def init_dataset_kfold(self):
        logger.info('init k_fold dataset')
        k_fold = self.config['trainer'].get('k_fold', None)
        if not k_fold:
            logger.error('k_fold not configured')
            raise ValueError('k_fold not configured')
        dataset_config = self.config['dataset']

        dataset = {}
        train_dataset_config = dataset_config.get('train', None)
        if not train_dataset_config:
            logger.error('train data configs not exit')
            raise ValueError('train data configs not exit')
        dev_dataset_config = dataset_config.get('dev', None)
        if dev_dataset_config:
            logger.warning('training_cv: dev data should not exit')
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
            logger.info('Load test data: {}'.format(len(test_data)))

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
        logger.info('Load trian data: {}'.format(len(train_data)))

        train_data_batch_size = train_dataset_config.get('batch_size', 12)
        # k_fold
        kfolds = KFold(n_splits=k_fold, shuffle=True, random_state=self.config.get('random_seed', 2022))
        for k, (train_indexs, dev_indexs) in enumerate(kfolds.split(train_data)):
            temp_train_data = [train_data[index] for index in train_indexs]
            temp_dev_data = [train_data[index] for index in dev_indexs]
            dataset['train'] = self.build_dataloader('train', temp_train_data, train_dataset_reader,
                                                     train_dataset_reader_config, train_data_batch_size, True)
            dataset['dev'] = self.build_dataloader('dev', temp_dev_data, train_dataset_reader,
                                                   train_dataset_reader_config, train_data_batch_size, False)
            logger.info('KFold {}: train data {}  dev data {}'.format(str(k), len(temp_train_data), len(temp_dev_data)))
            yield k + 1, dataset

    def init_dataset_inference(self):
        logger.info('init dataset')
        dataset_config = self.config.get('dataset', None)
        if dataset_config:
            dataset_reader_config = dataset_config['reader']
            dataset_reader = self.import_reader('inference', dataset_reader_config.get('type', None))

            data_path = dataset_config.get('data_path', None)
            if data_path:
                inference_data = self.get_data('inference', data_path, dataset_config.get('type', None))
            else:
                inference_data = None
                logger.warning('wrong data type or data path not configured')
            dataset = dataset_reader('inference', inference_data, dataset_reader_config)
            return dataset
        else:
            logger.error('inference dataset config not configured')
            raise ValueError('inference dataset config not configured')

    def init_model(self):
        logger.info('init model')
        model_config = self.config['model']
        model_type = model_config.get('type', None)
        if model_type:
            try:
                model = importlib.import_module('...models.' + model_type, __name__).model
            except Exception as ex:
                logger.error('{} model not existed or import error'.format(model_type))
                raise ex
            model = model(**model_config)
            # 加载权重
            weight_path = model_config.get('weight_path', None)
            if weight_path:
                weights = torch.load(weight_path, map_location='cpu')
                if isinstance(weights, Dict):
                    load_model_info = model.load_state_dict(weights)
                elif isinstance(weights, torch.nn.Module):
                    load_model_info = model.load_state_dict(weights.state_dict())
                else:
                    logger.error('weights can not been loaded')
                    raise ValueError('weights can not been loaded')
                logger.warning('missing_keys: {}'.format(load_model_info['missing_keys']))
                logger.warning('unexpected_keys: {}'.format(load_model_info['unexpected_keys']))
            return model
        else:
            logger.error('model type not configured')
            raise ValueError('model type not configured')

    def init_optimizer(self, train_dataset, model):
        logger.info('init optimizer')
        optimizer_config = self.config['optimizer']

        # 分层学习率
        encoder_modules_parameters = []
        other_modules_parameters = []
        for name, parameter in list(model.named_parameters()):
            if 'encoder' in name:
                encoder_modules_parameters.append((name, parameter))
            else:
                other_modules_parameters.append((name, parameter))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in encoder_modules_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': optimizer_config['weight_decay'], 'lr': float(optimizer_config['encoder_lr'])},
            {'params': [p for n, p in encoder_modules_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': float(optimizer_config['encoder_lr'])},
            {'params': [p for n, p in other_modules_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': optimizer_config['weight_decay'], 'lr': float(optimizer_config['lr'])},
            {'params': [p for n, p in other_modules_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,     'lr': float(optimizer_config['lr'])}
        ]
        optimizer = OPTIMIZERS[optimizer_config['type']](optimizer_grouped_parameters,
                                                         lr=float(optimizer_config['lr']),
                                                         weight_decay=optimizer_config['weight_decay'])

        scheduler_type = optimizer_config.get('lr_scheduler', None)
        if scheduler_type:
            # 不同的scheduler的参数不同，需要单独写
            if scheduler_type == 'linear_schedule_with_warmup':
                tol_steps = len(train_dataset['dataloader']) * self.config['trainer']['epoch']
                scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                            num_warmup_steps=int(
                                                                tol_steps * optimizer_config['warmup']),
                                                            num_training_steps=tol_steps)
        else:
            scheduler = None
        return optimizer, scheduler

    def init_trainer(self, model, optimizer, scheduler):
        logger.info('init trainer')
        trainer_config = self.config['trainer']
        trainer_type = trainer_config.get('type', 'base_trainer')
        try:
            trainer = importlib.import_module('...trainer.' + trainer_type, __name__).trainer
        except Exception as ex:
            logger.error('{} not existed or import error'.format(trainer_type))
            logger.error(ex)
            raise ex
        trainer = trainer(model, optimizer, scheduler, **trainer_config)
        return trainer

    def training(self):
        dataset = self.init_dataset()
        model = self.init_model()
        optimizer, scheduler = self.init_optimizer(dataset['train'], model)
        trainer = self.init_trainer(model, optimizer, scheduler)
        trainer.train_and_eval(dataset['train'], dataset['dev'], dataset['test'])
        logger.info('all done')

    def training_cv(self):
        for k, dataset in self.init_dataset_kfold():
            model = self.init_model()
            optimizer, scheduler = self.init_optimizer(dataset['train'], model)
            trainer = self.init_trainer(model, optimizer, scheduler)
            logger.info('start training_cv {}/{}'.format(k, self.config['trainer']['k_fold']))
            trainer.train_and_eval(dataset['train'], dataset['dev'], dataset['test'], info=str(k))
            logger.info('training_cv {}/{} done'.format(k, self.config['trainer']['k_fold']))
        logger.info('all done')

    def init_inference(self):
        logger.info('init inferencer')
        inference_config = self.config['inference']
        inference_type = inference_config.get('type', 'base_inference')
        try:
            inference = importlib.import_module('...inference.' + inference_type, __name__).inference
        except Exception as ex:
            logger.error('{} not existed or import error'.format(inference_type))
            raise ex
        inference = inference(**inference_config)
        return inference

    def inference(self):
        dataset = self.init_dataset_inference()
        inference = self.init_inference()
        inference.inference(dataset)
        logger.info('all done')

    def service(self):
        dataset = self.init_dataset_inference()
        inference = self.init_inference()

        app = FastAPI()

        class InputData(BaseModel):
            text: str

        @app.get("/health.json")
        def health():
            return {"status": "UP"}

        @app.post('/innerapi/ai/python/agent')
        async def default_interface(input_data: InputData):
            try:
                result = inference.service_inference(dataset, input_data.dict())
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
        uvicorn.run(app, host='0.0.0.0', port=8080)

    def predict(self, example):
        """
        预测一条数据，用于将框架嵌入到其他项目中
        """
        dataset = self.init_dataset_inference()
        inference = self.init_inference()
        result = inference.service_inference(dataset, example)
        return result

    def run(self):
        # 把config复制一遍到输出目录中，用作查看提醒
        if 'trainer' in self.config:
            output_path = self.config['trainer'].get('output_path', 'output')
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            write_yaml(os.path.join(output_path, 'config_bk.yaml'), self.config)
        cmd_func = getattr(self, self.cmd)
        cmd_func()


controller = BaseController
