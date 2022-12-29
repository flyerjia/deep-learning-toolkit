# -*- encoding: utf-8 -*-
"""
@File    :   common_utils.py
@Time    :   2022/07/12 20:05:10
@Author  :   jiangjiajia
"""
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.special import expit, softmax
from torch.optim import AdamW, Adam, SGD
from transformers import (BertModel, BertTokenizer, DebertaV2Model, ErnieModel,
                          MegatronBertModel, NezhaModel, T5Tokenizer, T5ForConditionalGeneration)

logger = logging.getLogger(__name__)

ENCODERS = {
    'bert': BertModel,
    'nezha': NezhaModel,
    'erlangshen': MegatronBertModel,
    'deberta-v2': DebertaV2Model,
    'ernie': ErnieModel,
    't5': T5ForConditionalGeneration,
    'lstm': nn.LSTM
}

TOKENIZERS = {
    'bert_tokenizer': BertTokenizer,
    't5_tokenizer': T5Tokenizer
}

OPTIMIZERS = {
    'adamw': AdamW,
    'adam': Adam,
    'sgd': SGD
}


def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_logger(log_file, name='dltk'):
    time_ = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s  - %(message)s',
                                   datefmt='%Y/%m/%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    file_handler = logging.FileHandler(log_file + '_' + time_ + '.log',
                                       encoding='UTF-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)


def logger_output(msg_type, msg, rank=0, only_ms_out=True):
    """
    日志输出封装

    Args:
        msg_type (str): 日志类型 info error warning
        msg (str): 输出信息
        rank (int, optional): rank id. Defaults to 0.
        only_ms_out (boolean, optional): 是否只有rank=0的进程输入信息. Defaults to True.
    """
    if (only_ms_out and rank == 0) or not only_ms_out:
        if msg_type == 'info':
            logger.info(msg)
        elif msg_type == 'error':
            logger.error(msg)
        elif msg_type == 'warning':
            logger.warning(msg)
        else:
            logger.warning(msg)


def get_device(use_gpu=False, rank=0, gpu_ids=[]):
    if use_gpu:
        if not torch.cuda.is_available():
            logger_output('warning', 'GPU not available and load CPU device', rank)
            device = torch.device('cpu')
        else:
            gpu_nums = torch.cuda.device_count()
            if 0 <= rank < gpu_nums:
                logger_output('info', 'use GPU: {}'.format(str(gpu_ids[rank])), rank, False)
                device = torch.device('cuda:' + str(rank))
            else:
                logger_output('warning', 'argument gpu_id not correct and load CPU device', rank, False)
                device = torch.device('cpu')

    else:
        device = torch.device('cpu')
    return device


def read_json(file_path):
    """
    read a json from file

    Args:
        file_path (str): file path
    """
    if not os.path.exists(file_path):
        logger_output('error', '{} not exit'.format(file_path))
        raise ValueError('{} not exit'.format(file_path))
    with open(file_path, 'r', encoding='utf-8') as fn:
        data = json.load(fn)
    return data


def write_json(file_path, data):
    """
    write a json to file

    Args:
        file_path (str): file path
        data (json): writed data
    """
    with open(file_path, 'w', encoding='utf-8') as fn:
        fn.write(json.dumps(data, ensure_ascii=False, indent=2))


def read_jsons(file_path):
    """
    read jsons from file. each line is a json

    Args:
        file_path (str): file path
    """
    if not os.path.exists(file_path):
        logger_output('error', '{} not exit'.format(file_path))
        raise ValueError('{} not exit'.format(file_path))
    data = []
    with open(file_path, 'r', encoding='utf-8') as fn:
        for line in fn.readlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsons(file_path, data):
    """
    write jsons to file

    Args:
        file_path (str): file paht
        data (list): json list
    """
    with open(file_path, 'w', encoding='utf-8') as fn:
        for each_json in data:
            fn.write(json.dumps(each_json, ensure_ascii=False) + '\n')


def read_text(file_path):
    """
    read text

    Args:
        file_path (str): file path
    """
    if not os.path.exists(file_path):
        logger_output('error', '{} not exit'.format(file_path))
        raise ValueError('{} not exit'.format(file_path))
    data = []
    with open(file_path, 'r', encoding='utf-8') as fn:
        for line in fn.readlines():
            line = line.strip()
            if line:
                data.append(line)
    return data


def write_text(file_path, data):
    """
    write text

    Args:
        file_path (str): file path
        data (list): data
    """
    with open(file_path, 'w', encoding='utf-8') as fn:
        for each_text in data:
            fn.write(each_text + '\n')


def read_yaml(file_path):
    """
    read yaml file

    Args:
        file_path (str): file path
    """
    if not os.path.exists(file_path):
        logger_output('error', '{} not exit'.format(file_path))
        raise ValueError('{} not exit'.format(file_path))
    with open(file_path, 'r', encoding='utf-8') as fn:
        data = yaml.safe_load(fn)
    return data


def write_yaml(file_path, data):
    """
    write yaml data to a file
    """
    with open(file_path, 'w', encoding='utf-8') as fn:
        yaml.safe_dump(data, fn, sort_keys=False, allow_unicode=True)


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[UNK]')
            else:
                tokens.append(_ch)

    return tokens


def numpy_softmax(logits):
    # logits = torch.from_numpy(logits)
    # probs = torch.softmax(logits, dim=-1).numpy()
    # return probs
    return softmax(logits, axis=-1)


def numpy_sigmoid(logits):
    # logits = torch.from_numpy(logits)
    # return torch.sigmoid(logits).numpy()
    return expit(logits)


def numpy_topk(matrix, K, axis=-1):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort.take(-(np.arange(K) + 1), axis=axis)


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring
