import torch
from sdpa import SDPA
from torch import nn

class MHA(nn.Module):
    def __init__(self, d_model, num_heads, drop_rate=0.0, is_masked=False, max_len=512): # is_maskedは未来マスクを担当
        super().__init__()
        assert d_model % num_heads == 0 # headで綺麗に分割される必要がある
        self.d_model = d_model
        self.num_heads = num_heads
        self.is_masked = is_masked
        self.d_k = d_model//num_heads
        self.sdpa = SDPA(drop_rate)

        # 最初にQ, K, V にかける行列
        self.W_Q = nn.Linear(d_model, d_model) # Q:(batch, seq_q, d_model) 
        self.W_K = nn.Linear(d_model, d_model) # K:(batch, seq_k, d_model)
        self.W_V = nn.Linear(d_model, d_model) # V:(batch, seq_k, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # 未来マスクの三角形を作っておき、あとで切り出して使う
        # register_bufferで勝手にdeviceに送られるようにする
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_len, max_len, dtype=bool)), persistent=False)

    # num_headsに分割する関数
    def _num_heads_division(self, X):
        X = X.view(X.size(0), X.size(1), self.num_heads, self.d_k)
        X = X.transpose(1, 2) # (batch, num_heads, seq, d_k)　
        # この転置は行うのは、SDPAが[-1]が特徴量、[-2]がseqという前提で書かれているから
        return X

    def forward(self, Q, K, V, mask=None): # maskはpadトークンなどを担当

        # Q, K, V を線形変換し、分割に対する調整を行う
        Q = self.W_Q(Q) # Q:(batch, seq_q, d_model) 
        K = self.W_K(K) # K:(batch, seq_k, d_model)
        V = self.W_V(V) # V:(batch, seq_k, d_model)

        # num_headsに分割
        Q = self._num_heads_division(Q) # Q:(batch, num_heads, seq_q, self.d_k)　
        K = self._num_heads_division(K) # K:(batch, num_heads, seq_k, self.d_k)
        V = self._num_heads_division(V) # V:(batch, num_heads, seq_k, self.d_k)

        # maskを統合
        if self.is_masked:
            assert Q.size(2) == K.size(2), "seq_q == seq_k が必要"
            assert Q.size(2) <= self.causal_mask.size(0), "max_len を超えています"
            future_mask = self.causal_mask[:Q.size(2), :Q.size(2)]
            if mask is None:
                mask = future_mask
            else:
                mask = mask.bool() & future_mask.bool()
            
        # SDPAを実行
        score = self.sdpa(Q, K, V, mask) # (batch, num_heads, seq_q, self.d_k)

        # 分割を結合する
        score = score.transpose(1, 2) # (batch, seq_q, num_heads, self.d_k)
        score = score.reshape(score.size(0), score.size(1), self.d_model) # (batch, seq_q, d_model)

        # 仕上げの線形変換
        score = self.W_O(score) # (batch, seq_q, d_model)

        return score
