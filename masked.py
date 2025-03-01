import torch
import numpy as np

def gen_subsequence_mask(dim):
    """这里是子句子的sequence mask"""
    #  生成全为1的矩阵
    matrix = np.ones((1, dim, dim))
    #  返回值为上三角矩阵布尔张量，True的位置表示不掩码，False掩码,size(1, seq_len, seq_len)
    return torch.from_numpy(np.triu(matrix, k=1)) == 0
