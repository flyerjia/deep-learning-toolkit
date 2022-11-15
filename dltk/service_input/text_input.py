# -*- encoding: utf-8 -*-
"""
@File    :   text_input.py
@Time    :   2022/11/15 16:29:12
@Author  :   jiangjiajia
"""
from pydantic import BaseModel


class InputData(BaseModel):
    text: str


input_data = InputData
