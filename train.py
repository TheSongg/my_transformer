from masked import gen_subsequence_mask
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json


class Batch:
    def __init__(self, src, trg=None, pad=0):
        """
        src：输入的token id矩阵，已经按照最长的token padding，size(batch_size, seq_len)
        trg：目标token id矩阵，预测试时为None, [BOS, w1, w2, w3, EOS], 在收尾添加BOS、EOS作为起止、结束标识
        pad：掩码的标记，不一定是0，要根据实际全部词元表映射

        src: torch.Size([2, 3])
            tensor([[1., 2., 5.],
                    [2., 3., 0.]])

        src_mask: torch.Size([2, 1, 3])
                tensor([[[ True,  True,  True]],

                            [[ True,  True, False]]])
        """
        self.src = src
        #  生成形状和src相同的掩码布尔值矩阵，不等于padding的地方为True，并在倒数第二处增加一维, size(batch_size, 1, seq_len)
        self.src_mask = (self.src != pad).unsqueeze(1)
        if trg is not None:
            #  对所有行切片处理，截断最后一个元素，作为decoder的输入
            #  模拟预测时的decoder输入，第一个输入为[BOS]，输出是w1，拼接后[BOS，w1]作为输入预测w2，依次类推，直到预测到EOS结束，因此去除EOS标识，输入为[BOS, w1, w2, w3]
            self.trg = trg[:, :-1]

            #  对所有行切片处理，去除第一个元素
            #  模拟作为decoder的输出，第一个输出为w1，第二个输出为w2，直到EOS预测结束，因此去除BOS，输出为[w1, w2, w3, EOS]
            self.trg_y = trg[:, 1:]

            #  得到padding mask和sequence mask合并后的mask
            self.trg_mask = self.make_mask(self.trg, pad)

            #  计算目标序列中非填充符号的数量，用于损失计算
            self.n_tokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_mask(tgt, pad):
        """
        tgt_mask:tensor([[[ True,  True,  True]],

                    [[ True,  True, False]]])
                 size(2, 1, 3)

        seq_mask:tensor([[[ True, False, False],
                         [ True,  True, False],
                         [ True,  True,  True]]])
                 size(1, 3, 3)


        合并后：tensor([[[ True, False, False],
                     [ True,  True, False],
                     [ True,  True,  True]],

                    [[ True, False, False],
                     [ True,  True, False],
                     [ True,  True, False]]])
              size(2, 3, 3)
        """
        #  句子的长度掩码矩阵，size（batch_size, 1, seq_len）
        tgt_mask = (tgt != pad).unsqueeze(-2)

        #  decoder的上三角矩阵，size（1, seq_len, seq_len）
        seq_mask = gen_subsequence_mask(tgt.size(-1))

        #  两个矩阵合并，先广播成相同形状（batch_size, squ_len. seq_len），然后相同位置如果有False，则合并后位置为False，在掩盖未来词元的同时掩盖padding的词元
        #  tgt_mask类型为FloatTensor，seq_mask类型为ByteTensor，type_as将类型统一为FloatTensor, size(batch_size, seq_len. seq_len）
        tgt_mask = tgt_mask & seq_mask.type_as(tgt_mask)
        return tgt_mask

def get_tokenizer():
    return AutoTokenizer.from_pretrained('./')


def get_text_encode(tokenizer, text):
    """根据输入文字转换成token id"""
    return tokenizer.encode(text, add_special_tokens=True)


def get_text_decode(tokenizer, token_ids):
    """根据输入的token is转换成文字"""
    return tokenizer.decode(token_ids)


class LoadData(Dataset):
    def __init__(self, file_path, vocab):
        self.vocab = vocab
        self.file_path = file_path
        self.tokenizer = get_tokenizer()


    def __getitem__(self, item):
        src = get_text_encode(self.tokenizer, self.src_sentences)
        tgt = get_text_encode(self.tokenizer, self.tgt_sentences)

        src = [self.vocab for word in self.src_sentences[item]]

    def synthetic_data(self):
        """将输入和输出分别整理成列表"""
        if os.path.isfile(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                _data = json.loads(f)
        else:
            raise Exception('data file not found!!!')
