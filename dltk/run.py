# -*- encoding: utf-8 -*-
"""
@File    :   run.py
@Time    :   2022/07/27 11:41:19
@Author  :   jiangjiajia
"""

import importlib
import logging
from multiprocessing import cpu_count

import torch
import transformers

from . import __name__ as package_name
from .utils.common_utils import get_args, init_logger, read_yaml, set_seed

logger = logging.getLogger(__name__)


def run(cmd=None, config=None, config_path=None):

    # 获取CPU核数
    # cpu_nums = cpu_count()
    # Mac下默认为CPU核数一半，而服务器默认为CPU核数总数
    # torch.set_num_threads(int(cpu_nums / 2))
    transformers.logging.set_verbosity_error()

    if not cmd and not config and not config_path:
        args = get_args()
        cmd = args.command
        config_path = args.config
        config = read_yaml(config_path)

    if cmd and not config and config_path:
        config = read_yaml(config_path)

    init_logger(config['log_file'], package_name)
    set_seed(config.get('random_seed', 2022))

    logger.info('configs: {}'.format(config))

    controller_type = config.get('controller', None)
    if not controller_type:
        controller_type = 'base_controller'
    try:
        controller = importlib.import_module('..controller.' + controller_type, __name__).controller
    except Exception as ex:
        logger.error('{} controller not existed or import error'.format(controller_type))
        logger.error(ex)
        raise ex
    controller = controller(cmd, config)
    controller.run()
