import torch
import torch.nn as nn
from common import clone_module
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_n, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.heads_n = heads_n
        assert self.d_model % self.heads_n == 0
        self.dim_k = self.d_model // self.heads_n

        #  Q、K、V、O四个线性转换权重矩阵，这里K、V的维度相同
        self.liners = clone_module(nn.Linear(self.d_model, self.d_model), 4)
        #  注意力分数
        self.atten = None

    def forward(self, query, key, value, mask=None):
        """
        多头注意力下，所有头共享一个mask，要保证维度相同，因此扩展一个维度。这里掩码的是token
        mask初始维度（batch_size， sequence_len）,要与attention_score（batch_size, heads_num, sequence_len, sequence_len）维度对应，
        需要增加一个维度，size变为（batch_size， 1，sequence_len），然后在scores.masked_fill(mask==0, value=float("-inf"))时根据pytorch的广播机制，mask形状进一步变为（batch_size, heads_num, sequence_len, sequence_len）

        为什么不直接扩展成与scores相同维度？为了节省内存不然每批次每个头都要大幅度提高mask维度；提高计算效率
        为什么要扩展一次而不是全部给广播机制做？ 明确扩展目标是头，避免潜在的广播错误
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        #  线性转换
        _list = []
        for l, x in zip(self.liners, (query, key, value)):
            _list.append(l(x).view(batch_size, -1, self.heads_n, self.dim_k).transpose(1, 2))
        query, key, value = _list[0], _list[1], _list[2]

        attention_scores, self.atten = attention(query, key, value, mask=mask, dropout=self.dropout)
        #  多头注意力拼接
        attention_scores = attention_scores.transpose(1, 2).contiguous().view(batch_size, -1, self.heads_n * self.dim_k)

        #  再次线性转换
        return self.liners[-1](attention_scores)


def attention(query, key, value, mask=None, dropout=None):
    """
    计算注意力分数
    QKV形状(batch_size, heads_num, sequence_len, head_dim)
    """

    #  Q与K的转置(batch_size, heads_num, head_dim，sequence_len)点积，然后缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1)) # (batch_size, heads_num, sequence_len, sequence_len)

    #  decoder中的第一个attention， mask为下三角矩阵,在归一化前掩码
    if mask is not None:
        scores = scores.masked_fill(mask==0, value=float("-inf"))
    #  在sequence_len维度上转换成概率
    percent_att = F.softmax(scores, dim=-1)

    #  正则化，随机丢弃部分神经元，防止模型过拟合
    if dropout is not None:
        percent_att = dropout(percent_att)

    #  (batch_size, heads_num, sequence_len, head_dim), (sequence_len, sequence_len)
    return torch.matmul(percent_att, value), percent_att
