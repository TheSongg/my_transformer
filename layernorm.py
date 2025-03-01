import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """归一化层"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        #  缩放参数，可学习
        self.a_2 = nn.Parameter(torch.ones(features))
        #  平移参数，可学习
        self.b_2 = nn.Parameter(torch.zeros(features))
        #  很小的数，防止后续计算时被除数为0
        self.eps = eps

    def forward(self, x):
        #  计算x最后一维的均值
        mean = x.mean(-1, keepdim=True)
        #  计算x最后一维的标准差
        std = x.std(-1, keepdim=True)
        #  对标准化后的输出缩放、平移
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
