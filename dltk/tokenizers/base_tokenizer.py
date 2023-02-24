# -*- encoding: utf-8 -*-
"""
@File    :   base_tokenizer.py
@Time    :   2022/07/21 14:41:29
@Author  :   jiangjiajia
"""
import logging
from typing import List

from ..utils.common_utils import logger_output


class BaseTokenizer:
    def __init__(self, **kwargs):
        """
        基础的分词器，暂时使用Transformers的分词器，后续再补充
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

    def save_pretrained(self, save_path) -> List[str]:
        """
        分词器的保存函数

        Args:
            save_path (str): 保存路径

        Returns:
            List[str]: 保存的文件名
        """
        raise NotImplementedError
