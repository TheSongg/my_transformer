import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        #  第一个线性转换层，将维度放大到2048
        self.w1 = nn.Linear(d_model, d_ff)
        #  第二个线性转换层，恢复维度
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
