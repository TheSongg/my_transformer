import torch.nn as nn
from common import clone_module, SubLayerConnection
from layernorm import LayerNorm


class Encoder(nn.Module):
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        #  N个encoder拼接
        self.layers = clone_module(layer, n)
        #  Add & Norm层
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for _layer in self.layers:
            x = _layer(x, mask)
        #  最后一层也要残差连接，因为残差连接放在了最前面
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.attn = attn
        self.feed_forward = feed_forward
        #  每层由一个attention和一个feed forward组成
        self.sublayerconn = clone_module(SubLayerConnection(self.size, dropout), 2)

    def forward(self, x, mask):
        #  第一个是attention，需要使用lambda
        x = self.sublayerconn[0](x, lambda y: self.attn(x, x, x, mask))
        #  第二个是前馈网络层，不需要lambda，调用FeedForward实例的forward方法，有个一入参
        x = self.sublayerconn[1](x, self.feed_forward)
        return x
