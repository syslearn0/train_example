import math

import torch
import torch.nn.functional as F
from torch import nn


class SDPA(nn.Module):
    def __init__(self, drop_rate=0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, Q, K, V, mask=None):
        # Q: (batch, seq_q, d_k)
        # K: (batch, seq_k, d_k)
        # V: (batch, seq_k, d_v)
        d_k = Q.size(-1) 
        Q_K_product = Q @ K.transpose(-2, -1) / math.sqrt(d_k) # (batch, seq_q, seq_k)

        if mask is not None:
            Q_K_product = Q_K_product.masked_fill(mask == 0, torch.finfo(Q_K_product.dtype).min) # マスクする（CAUTION : mask==0のところをマスクすることとする）

        Q_K_softmax = F.softmax(Q_K_product, dim=-1) # dim=-1 : seq_k方向に合計1,  (batch, seq_q, seq_k)
        Q_K_softmax = self.dropout(Q_K_softmax)
        Q_K_V_attention = Q_K_softmax @ V # (batch, seq_q, d_v)
        return Q_K_V_attention