# -*- encoding: utf-8 -*-
"""
@File    :   base_input.py
@Time    :   2022/11/04 11:43:12
@Author  :   jiangjiajia
"""
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class InputData(BaseModel):
    pid: int
    diagnose: str
    category: str
    ocr_json: List[Dict]
    output_all_keys: bool = False  # 默认值
    name: Union[str, None]  # 可选类型
    debug: Optional[bool] = True  # 可选参数和默认值


input_data = InputData
