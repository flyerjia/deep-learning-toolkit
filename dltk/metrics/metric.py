# -*- encoding: utf-8 -*-
"""
@File    :   metric.py
@Time    :   2022/07/29 10:43:07
@Author  :   jiangjiajia
"""


def compute_f1(prediction, target):
    num_predictions = len(prediction)
    num_golds = len(target)
    num_correct = len(set(prediction) & set(target))
    precision = num_correct / num_predictions if num_predictions > 0 else 0.
    recall = num_correct / num_golds if num_golds > 0 else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.
    result = {}
    result['P'] = precision
    result['R'] = recall
    result['F1'] = f1
    return result
