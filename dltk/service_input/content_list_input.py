# -*- encoding: utf-8 -*-
"""
@File    :   text_list_input.py
@Time    :   2022/11/04 11:45:27
@Author  :   jiangjiajia
"""

from typing import List

from pydantic import BaseModel


class InputData(BaseModel):
    content: List[str]


input_data = InputData
