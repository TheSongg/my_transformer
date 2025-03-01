import torch
import torch.nn as nn
import numpy as np
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


#  positional embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        #  在训练阶段随机将部分数值置为0，其他数据除以（1-dropout），防止模型依赖于部分数据，避免过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #  x size(batch_size, sequence_len, d_model)
        seq_len = x.size(1)
        embedding_dim = x.size(2)
        positional_encoding = np.zeros((seq_len, embedding_dim))
        _k = [0, embedding_dim//2]
        #  偶数位置为sin，奇数位置为cos
        for row in range(positional_encoding.shape[0]):
            for col in _k:
                if 2*col > len(positional_encoding[0]) or 2*col+1 >= len(positional_encoding[0]):
                    break
                positional_encoding[row][2*col] = math.sin(row/(10000**(2*col/self.d_model)))
                positional_encoding[row][(2*col)+1] = math.cos(row/(10000**(2*col/self.d_model)))
        pe = torch.from_numpy(positional_encoding) + x
        return self.dropout(pe)
