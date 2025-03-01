import torch.nn as nn
from attention import MultiHeadAttention
from embedding import Embedding, PositionalEncoding
from feedforward import FeedForward
from encoders import Encoder, EncoderLayer
from decoders import Decoder, DecoderLayer
from generator import Generator
import copy


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embedding, tgt_embedding, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        #  先完成这一批次的encoder
        enc = self.encode(src, src_mask)
        #  在进入decoder处理
        output = self.decode(enc, src_mask, tgt, tgt_mask)
        return output

    def encode(self, src, src_mask):
        return self.encoder(self.src_embedding(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        tgt：token id
        memory：来自encoder
        """
        return self.decoder(self.tgt_embedding(tgt), memory, src_mask, tgt_mask)



def create_model(src_vocab, tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    atten = MultiHeadAttention(h, d_model, dropout)
    pe = PositionalEncoding(d_model, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    en_layer = EncoderLayer(d_model, copy.deepcopy(atten), copy.deepcopy(ff), dropout)
    de_layer = DecoderLayer(d_model, copy.deepcopy(atten), copy.deepcopy(atten), copy.deepcopy(ff), dropout)
    src_emb = nn.Sequential(Embedding(src_vocab, d_model), copy.deepcopy(pe))
    tgt_emb = nn.Sequential(Embedding(tgt_vocab, d_model), copy.deepcopy(pe))

    model = EncoderDecoder(Encoder(en_layer, n), Decoder(de_layer, n), src_emb, tgt_emb, Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

