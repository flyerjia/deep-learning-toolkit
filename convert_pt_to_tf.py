# -*- encoding: utf-8 -*-
"""
@File    :   convert_pt_to_tf.py
@Time    :   2022/08/15 22:21:51
@Author  :   jiangjiajia
"""
# 把pytorch模型转换成tf模型，暂时只支持CPU环境转换

import random

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
import torch.onnx
from onnx_tf.backend import prepare

from dltk.utils.common_utils import TOKENIZERS


def test_diff(data_a, data_b):
    try:
        np.testing.assert_allclose(data_a, data_b, rtol=1e-03, atol=1e-05)
    except AssertionError as ae:
        print('数据差异超过范围')
        print(ae)
        return False
    return True


torch_model_path = 'model.pth'  # 设置pytorch模型地址
onnx_model_path = 'onnx_model.onnx'  # onnx模型保存地址
tf_model_dir = 'tf_model'  # tf模型保存文件夹路径
# 入参设置
tokenizer_type = 'bert_tokenizer'
vocab_path = 'pretrain_models'
max_len = 512
content = 'test data'
tokenizer = TOKENIZERS[tokenizer_type].from_pretrained(vocab_path)
tokenize_result = tokenizer.encode_plus(text=content, add_special_tokens=True, padding='max_length',
                                        truncation=True, max_length=max_len, return_tensors='np')
input_ids = tokenize_result['input_ids']
token_type_ids = tokenize_result['token_type_ids']
attention_mask = tokenize_result['attention_mask']

# 1. pytorch转onnx
model = torch.load(torch_model_path, map_location=torch.device('cpu'))
model.eval()
print('*'*50)
print('pytorch model:')
print(model)
print('*' * 80)
# 设置pytorch模型参数
input_data = (
    torch.from_numpy(input_ids),
    torch.from_numpy(token_type_ids),
    torch.from_numpy(attention_mask)
)
input_names = ['input_ids', 'token_type_ids', 'attention_mask']
output_names = ['logits']
# 动态维度
dynamic_axes = {
    'input_ids': {0: 'batch'},
    'token_type_ids': {0: 'batch'},
    'attention_mask': {0: 'batch'},
    'logits': {0: 'batch'}
}
torch.onnx.export(model, input_data, onnx_model_path, input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes, verbose=True, opset_version=12)
with torch.no_grad():
    torch_result = model(input_data[0], input_data[1], input_data[2])['logits'].detach().numpy()
del model
torch.cuda.empty_cache()
print('torch result:')
print(torch_result)

print('*' * 50)
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print('onnx model:')
print(onnx.helper.printable_graph(onnx_model.graph))

print('*' * 80)
# 加载onnx模型，并输出结果
session = ort.InferenceSession(onnx_model_path)
# 设置onnx模型参数
onnx_result = session.run(None, {'input_ids': input_ids, 'token_type_ids': token_type_ids,
                          'attention_mask': attention_mask})[0]
print('onnx result:')
print(onnx_result)
if test_diff(torch_result, onnx_result):
    print('pytorch和onnx模型输出差异满足要求')
else:
    print('pytorch和onnx模型输出差异过大，不满足要求')

# 2. onnx转tf，并输出tf结构
print('*' * 80)
tf_model = prepare(onnx_model)
tf_model.export_graph(tf_model_dir)
# 设置tf模型参数
tf_result = tf_model.run({'input_ids': tf.convert_to_tensor(input_ids), 'token_type_ids': tf.convert_to_tensor(token_type_ids),
                          'attention_mask': tf.convert_to_tensor(attention_mask)})[0]
print('tf result:')
print(tf_result)
if test_diff(torch_result, tf_result):
    print('pytorch和tf模型输出差异满足要求')
else:
    print('pytorch和tf模型输出差异过大，不满足要求')

print('Done')
