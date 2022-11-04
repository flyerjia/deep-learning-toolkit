# -*- encoding: utf-8 -*-
"""
@File    :   run.py
@Time    :   2022/07/27 11:41:19
@Author  :   jiangjiajia
"""
import argparse
import importlib
import os

import yaml

from . import __name__ as package_name


def get_args():
    cmd_choices = ['training', 'training_cv', 'inference', 'service']
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', choices=cmd_choices, required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    return args


def read_yaml(file_path):
    """
    read yaml file

    Args:
        file_path (str): file path
    """
    if not os.path.exists(file_path):
        raise ValueError('{} not exit'.format(file_path))
    with open(file_path, 'r', encoding='utf-8') as fn:
        data = yaml.safe_load(fn)
    return data


def main(cmd=None, config=None, config_path=None):
    if not cmd and not config and not config_path:
        args = get_args()
        cmd = args.command
        config_path = args.config
        config = read_yaml(config_path)

    if cmd and not config and config_path:
        config = read_yaml(config_path)

    if config.get('use_gpu', False):
        gpu_ids = config.get('gpu_ids', 0)
        if isinstance(gpu_ids, str):
            gpu_ids = ','.join([gpu_id.strip() for gpu_id in gpu_ids.split(',')])
        else:
            gpu_ids = str(gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    try:
        import torch
        import torch.multiprocessing as mp
    except Exception as ex:
        raise ex
    n_gpus = torch.cuda.device_count()
    # 暂时只支持训练阶段并行
    if cmd in ['training', 'training_cv'] and n_gpus >= 2:
        ddp_flag = True
        mp.spawn(run, args=(n_gpus, ddp_flag, cmd, config), nprocs=n_gpus, join=True)
    else:
        ddp_flag = False
        run(0, n_gpus, ddp_flag, cmd, config)


def run(rank, n_gpus, ddp_flag, cmd, config):
    if ddp_flag:
        try:
            import torch.distributed as dist
        except Exception as ex:
            raise ex
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=n_gpus)

    try:
        from .utils.common_utils import init_logger, logger_output, set_seed
    except Exception as ex:
        raise ex

    init_logger(config['log_file'], package_name)
    set_seed(config.get('random_seed', 2022))
    logger_output('info', 'configs: {}'.format(config), rank)

    controller_type = config.get('controller', None)
    if not controller_type:
        controller_type = 'base_controller'
    try:
        controller = importlib.import_module('..controller.' + controller_type, __name__).controller
    except Exception as ex:
        logger_output('error', '{} controller not existed or import error'.format(controller_type), rank)
        raise ex
    controller = controller(rank, ddp_flag, cmd, config)
    controller.run()

    if ddp_flag:
        dist.destroy_process_group()
