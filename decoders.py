import torch.nn as nn
from common import clone_module, SubLayerConnection
from layernorm import LayerNorm


class Decoder(nn.Module):
    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clone_module(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for _layer in self.layers:
            x = _layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.size = size
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        #  self attention、cross attention、feed forward
        self.sublayerconn = clone_module(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        #  self attention 这里需要根据decoder的输入掩码
        x = self.sublayerconn[0](x, lambda y: self.attn(x, x, x, tgt_mask))

        #  cross attention query是上一个decoder的输出，key、value是encoder的输出，mask是encoder和decoder的合并mask
        x = self.sublayerconn[1](x, lambda y: self.src_attn(x, memory, memory, src_mask))

        #  前馈层
        x = self.sublayerconn[2](x, self.feed_forward)
        return x
