# -*- encoding: utf-8 -*-
"""
@File    :   efficient_globalpointer.py
@Time    :   2022/09/08 20:02:57
@Author  :   jiangjiajia
"""

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(self, output_dim, merge_mode='add'):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode

    def forward(self, inputs):
        seq_len = inputs.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.float)[None]  # [1, seq_len]
        indices = torch.arange(self.output_dim // 2, dtype=torch.float)  # [output_dim // 2]
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)  # [output_dim // 2]
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)  # [1, seq_len, output_dim // 2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)],
                                 dim=-1)  # [1, seq_len, output_dim // 2, 2]
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))  # [1, seq_len, output_dim]
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(self, hidden_size, heads, head_size):
        super(EfficientGlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size, head_size * 2, bias=True)
        self.linear_2 = nn.Linear(head_size * 2, heads * 2, bias=True)

    def forward(self, inputs):
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, head_size * 2]
        inputs = self.linear_1(inputs)
        # qw, kw: [batch_size, seq_len, head_size]
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # [1, seq_len, output_dim]
        pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
        # [1, seq_len, output_dim]
        cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
        # [1, seq_len, output_dim]
        sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
        # [batch_size, seq_len, head_size//2, 2]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
        # [batch_size, seq_len, head_size]
        qw2 = torch.reshape(qw2, qw.shape)
        # [batch_size, seq_len, head_size]
        qw = qw * cos_pos + qw2 * sin_pos
        # [batch_size, seq_len, head_size//2, 2]
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
        # [batch_size, seq_len, head_size]
        kw2 = torch.reshape(kw2, kw.shape)
        # [batch_size, seq_len, head_size]
        kw = kw * cos_pos + kw2 * sin_pos
        # [batch_size, seq_len, seq_len]
        logits = torch.einsum('bmd , bnd -> bmn', qw, kw) / self.head_size**0.5
        # linear_2(inputs): [batch_size, seq_len, head_size * 2] -> [batch_size, seq_len, heads * 2]
        # bias: [batch_size, heads * 2, seq_len]
        bias = torch.einsum('bnh -> bhn', self.linear_2(inputs)) / 2
        # logits: [batch_size, 1, seq_len, seq_len] + [batch_size, heads, 1, seq_len] + [batch_size, heads, seq_len, 1]
        # logits: [batch_size, heads, seq_len, seq_len]
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        # logits: [batch_size, seq_len, seq_len, heads]
        logits = torch.permute(logits, (0, 2, 3, 1))
        return logits
