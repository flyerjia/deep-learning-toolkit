# -*- encoding: utf-8 -*-
"""
@File    :   base_tokenizer.py
@Time    :   2022/07/21 14:41:29
@Author  :   jiangjiajia
"""

import logging

logger = logging.getLogger(__name__)


class BaseTokenizer:
    def __init__(self, **kwargs):
        """
        基础的分词器，暂时使用Transformers的分词器，后续再补充
        """
        for name, value in kwargs.items():
            setattr(self, name, value)
        pass
