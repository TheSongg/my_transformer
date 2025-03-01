import torch.nn as nn
import copy
from layernorm import LayerNorm


def clone_module(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class SubLayerConnection(nn.Module):
    """norm-->attention-->dropout-->feed forward"""
    def __init__(self, features, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """这里的sublayer是一个函数，有一个入参，但是当sublayer是有5个入参attention函数时，
        使用lambda调用attention，避免入参问题，参数self.norm(x)不影响attention函数
        """
        return x + self.dropout(sublayer(self.norm(x)))
