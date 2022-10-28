# -*- encoding: utf-8 -*-
"""
@File    :   base_trainer.py
@Time    :   2022/07/19 20:44:42
@Author  :   jiangjiajia
"""

import logging
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast

from ..modules.ema import EMA
from ..modules.fgm import FGM
from ..utils.common_utils import get_device

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, model, optimizer, scheduler, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.kwargs = kwargs
        # 设置显卡
        self.device = get_device(self.kwargs.get('use_gpu', False), self.kwargs.get('gpu_id', 0))

        # 设置保存文件路径
        output_path = self.kwargs.get('output_path', 'output')
        if self.kwargs.get('save_evaluations', False):
            self.kwargs['save_evaluations'] = os.path.join(output_path, 'evaluations')
        if self.kwargs.get('save_checkpoints', False):
            self.kwargs['save_checkpoints'] = os.path.join(output_path, 'checkpoints')
        if self.kwargs.get('save_predictions', False):
            self.kwargs['save_predictions'] = os.path.join(output_path, 'predictions')

        self.all_eval_info = {}
        self.saved_model_info = []

    def train_and_eval(self, train_dataset, dev_dataset, test_dataset, info=None):
        logger.info('starting training')
        self.model.to(self.device)

        if self.kwargs.get('use_fp16', False):
            scaler = GradScaler()
        if self.kwargs.get('use_fgm', False):
            adv_model = FGM(self.model)
        if self.kwargs.get('use_ema', False):
            self.ema_model = EMA(self.model, 0.9)
            self.ema_model.register()
        best_model_index = -1
        best_model_epoch = -1
        for epoch in range(1, self.kwargs.get('epoch', 10) + 1):
            self.model.train()
            for step, batch_data in enumerate(train_dataset['dataloader']):
                begin_time = time.time()
                self.optimizer.zero_grad()
                batch_data = {data_name: data_value.to(self.device) for data_name, data_value in batch_data.items()}
                batch_data['phase'] = 'train'
                if self.kwargs.get('use_fp16', False):
                    with autocast():
                        output = self.model(**batch_data)
                    scaler.scale(output['loss']).backward()
                else:
                    output = self.model(**batch_data)
                    output['loss'].backward()
                end_time = time.time()
                used_time = end_time - begin_time
                if not info:
                    logger.info('epoch:{} step:{}/{} train loss:{:.6f} time:{:.6f}'.format(epoch, step + 1, len(train_dataset['dataloader']),
                                                                                           output['loss'], used_time))
                else:
                    logger.info('cv[{}] epoch:{} step:{}/{} train loss:{:.6f} time:{:.6f}'.format(info, epoch, step + 1, len(train_dataset['dataloader']),
                                                                                                  output['loss'], used_time))

                if self.kwargs.get('use_fgm', False):
                    adv_model.attack()
                    if self.kwargs.get('use_fp16', False):
                        with autocast():
                            output = self.model(**batch_data)
                        scaler.scale(output['loss']).backward()
                    else:
                        output = self.model(**batch_data)
                        output['loss'].backward()
                    adv_model.restore()
                    adv_end_time = time.time()
                    used_time = adv_end_time - end_time
                    if not info:
                        logger.info('epoch:{} step:{}/{} adv train loss:{:.6f} time:{:.6f}'.format(epoch, step + 1, len(train_dataset['dataloader']),
                                                                                                   output['loss'], used_time))
                    else:
                        logger.info('cv[{}] epoch:{} step:{}/{} adv train loss:{:.6f} time:{:.6f}'.format(info, epoch, step + 1, len(train_dataset['dataloader']),
                                                                                                          output['loss'], used_time))

                if self.kwargs.get('use_fp16', False):
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                if self.kwargs.get('ema', False):
                    self.ema_model.update()

                if self.scheduler:
                    self.scheduler.step()

            # 对验证数据数据测评
            if 'eval_epoch' in self.kwargs.keys() and epoch % self.kwargs.get('eval_epoch', 1) == 0 and dev_dataset:
                dev_metrics_output = self.eval(epoch, 'dev', dev_dataset, info)
                if not info:
                    logger.info('epoch:{} time cost:{}'.format(epoch, dev_metrics_output['time']))
                else:
                    logger.info('cv[{}] epoch:{} time cost:{}'.format(info, epoch, dev_metrics_output['time']))
                if self.kwargs.get('save_evaluations', None):
                    self.save_evaluation_file(epoch, 'dev',  dev_metrics_output, info)

            # 测试集
            if 'test_epoch' in self.kwargs.keys() and epoch % self.kwargs.get('test_epoch', 1) == 0 and test_dataset:
                self.test(epoch, 'test', test_dataset, info)

            # 保存最优指标用来早停
            if epoch in self.all_eval_info.keys() and self.kwargs['model_sort_key'] in self.all_eval_info[epoch].keys() and \
                    self.all_eval_info[epoch][self.kwargs['model_sort_key']] >= best_model_index:
                best_model_index = self.all_eval_info[epoch][self.kwargs['model_sort_key']]
                best_model_epoch = epoch

            if epoch in self.all_eval_info.keys() and self.kwargs['model_sort_key'] in self.all_eval_info[epoch].keys() and self.kwargs.get('save_predictions', False):
                self.save_model(epoch, info)

            # 早停判断（多跑3轮，防止因波动早停）
            if self.kwargs.get('early_stopping', False):
                if epoch > 1 and epoch in self.all_eval_info.keys() and self.kwargs['model_sort_key'] in self.all_eval_info[epoch].keys() and \
                        epoch - best_model_epoch >= 3 * self.kwargs.get('eval_epoch', 1):  # 当3次eval后，指标持续下降，则早停
                    break
        # 在不设置早停的情况下，保存最后一次的模型
        if not self.kwargs.get('early_stopping', False):
            if not info:
                file_name = 'model_lastest.pth'
            else:
                file_name = info + '_model_lastest.pth'
            if not os.path.exists(self.kwargs['save_checkpoints']):
                os.mkdir(self.kwargs['save_checkpoints'])
            save_path = os.path.join(self.kwargs['save_checkpoints'], file_name)
            torch.save(self.model, save_path)
            if not info:
                logger.info('saving model: {}'.format(file_name))
            else:
                logger.info('cv[{}] saving model: {}'.format(info, file_name))

        logger.info('training done')

    def eval(self, epoch, phase, dataset, info=None):
        logger.info('starting eval')
        forward_output = {}
        forward_target = {}

        begin_time = time.time()
        if self.kwargs.get('use_ema', False):
            self.ema_model.apply_shadow()
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(dataset['dataloader']):
                step_begin_time = time.time()
                batch_data = {data_name: data_value.to(self.device) for data_name, data_value in batch_data.items()}
                batch_data['phase'] = phase
                output = self.model(**batch_data)
                # 全部转换成numpy，减少gpu显存消耗
                for data_name, data_value in batch_data.items():
                    if not isinstance(data_value, torch.Tensor):
                        continue
                    data_value = data_value.cpu().numpy()
                    forward_target.setdefault(data_name, []).append(data_value)
                for data_name, data_value in output.items():
                    if not isinstance(data_value, torch.Tensor):
                        continue
                    data_value = data_value.detach().cpu().numpy()
                    forward_output.setdefault(data_name, []).append(data_value)
                step_end_time = time.time()
                step_used_time = step_end_time - step_begin_time
                logger.info('epoch:{} eval data step:{}/{} time:{:.6f}'.format(epoch, step + 1, len(dataset['dataloader']), step_used_time))

        metrics_output = self.model.get_metrics(phase, forward_output, forward_target, dataset['dataset'])
        if self.kwargs.get('use_ema', False):
            self.ema_model.restore()

        end_time = time.time()
        metrics = {}
        metrics['epoch'] = epoch
        metrics['time'] = end_time - begin_time
        metrics.update(metrics_output)

        self.all_eval_info[epoch] = metrics

        if self.kwargs.get('save_predictions', None):
            self.save_prediction_file(epoch, phase, forward_output, dataset, info)

        logger.info('eval done')

        return metrics

    def test(self, epoch, phase, dataset, info=None):
        logger.info('starting test')
        forward_output = {}

        if self.kwargs.get('use_ema', False):
            self.ema_model.apply_shadow()
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(dataset['dataloader']):
                step_begin_time = time.time()
                batch_data = {data_name: data_value.to(self.device) for data_name, data_value in batch_data.items()}
                batch_data['phase'] = phase
                output = self.model(**batch_data)
                # 全部转换成numpy，减少gpu显存消耗
                for data_name, data_value in output.items():
                    if not isinstance(data_value, torch.Tensor):
                        continue
                    data_value = data_value.detach().cpu().numpy()
                    forward_output.setdefault(data_name, []).append(data_value)
                step_end_time = time.time()
                step_used_time = step_end_time - step_begin_time
                logger.info('epoch:{} test data step:{}/{} time:{:.6f}'.format(epoch, step + 1, len(dataset['dataloader']), step_used_time))
        if self.kwargs.get('use_ema', False):
            self.ema_model.restore()

        if self.kwargs.get('save_predictions', None):
            self.save_prediction_file(epoch, phase, forward_output, dataset, info)
        
        logger.info('test done')

    def save_evaluation_file(self, epoch, phase,  metrics_output, info=None):
        if not os.path.exists(self.kwargs['save_evaluations']):
            os.makedirs(self.kwargs['save_evaluations'])
        if not info:
            file_name = 'evaluations.{}'.format(phase)
        else:
            file_name = info + '_evaluations.{}'.format(phase)
        file_path = os.path.join(self.kwargs['save_evaluations'], file_name)
        is_new = False
        if not os.path.exists(file_path):
            is_new = True
        else:
            # 读取结果文件的列名输出对应的结果，以保证一致性（以第一次的结果为准）
            with open(file_path, 'r', encoding='utf-8') as fn:
                columns = fn.readlines()[0].strip()
                columns = columns.split('\t')
        with open(file_path, 'a', encoding='utf-8') as fn:
            if is_new:
                fn.write('{}\n'.format('\t'.join(metrics_output.keys())))
                output_values = metrics_output.values()
            else:
                output_values = [metrics_output.get(column, '') for column in columns]
            values = []
            for value in output_values:
                if isinstance(value, float):
                    val = '{:.6f}'.format(value)
                else:
                    val = str(value)
                values.append(val)
            fn.write('{}\n'.format('\t'.join(values)))

    def save_prediction_file(self, epoch, phase, forward_output, dataset, info=None):
        if not os.path.exists(self.kwargs['save_predictions']):
            os.makedirs(self.kwargs['save_predictions'])
        if not info:
            file_name = 'prediction.{}.{}'.format(phase, epoch)
        else:
            file_name = info + '_prediction.{}.{}'.format(phase, epoch)
        file_path = os.path.join(self.kwargs['save_predictions'], file_name)
        self.model.save_predictions(forward_output, dataset['dataset'], file_path)
        if not info:
            logger.info('saving predicton:{}'.format(file_name))
        else:
            logger.info('cv[{}] saving predicton:{}'.format(info, file_name))

    def save_model(self, epoch, info=None):

        def _model_sort_info(model_info):
            return model_info.get(self.kwargs['model_sort_key'], 0)

        model_info = self.all_eval_info.get(epoch, {'epoch': epoch})
        need_save = True
        if self.kwargs.get('max_model_num', None) and len(self.saved_model_info) >= self.kwargs.get('max_model_num', 1):
            if _model_sort_info(model_info) < _model_sort_info(self.saved_model_info[-1]):
                need_save = False
        if need_save and self.kwargs.get('save_checkpoints', None):
            if not info:
                file_name = 'model_{}.pth'.format(str(epoch))
            else:
                file_name = info + '_model_{}.pth'.format(str(epoch))
            if not os.path.exists(self.kwargs['save_checkpoints']):
                os.mkdir(self.kwargs['save_checkpoints'])
            save_path = os.path.join(self.kwargs['save_checkpoints'], file_name)
            if self.kwargs.get('use_ema', False):
                self.ema_model.apply_shadow()
            torch.save(self.model, save_path)
            if self.kwargs.get('use_ema', False):
                self.ema_model.restore()
            if not info:
                logger.info('saving model: {}'.format(file_name))
            else:
                logger.info('cv[{}] saving model: {}'.format(info, file_name))
        if need_save and self.kwargs.get('max_model_num', None):
            self.saved_model_info.insert(0, model_info)
            self.saved_model_info.sort(key=lambda x: _model_sort_info(x), reverse=True)
            models_for_dels = self.saved_model_info[self.kwargs['max_model_num']:]
            for model_info in models_for_dels:
                temp_epoch = model_info['epoch']
                if self.kwargs.get('save_checkpoints', None):
                    if not info:
                        file_name = 'model_{}.pth'.format(str(temp_epoch))
                    else:
                        file_name = info + '_model_{}.pth'.format(str(temp_epoch))
                    save_path = os.path.join(self.kwargs['save_checkpoints'], file_name)
                    os.remove(save_path)
                    if not info:
                        logger.info('remove model: {}'.format(file_name))
                    else:
                        logger.info('cv[{}] remove model: {}'.format(info, file_name))
            self.saved_model_info = self.saved_model_info[:self.kwargs['max_model_num']]


trainer = BaseTrainer
