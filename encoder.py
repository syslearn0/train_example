from torch import nn
from mha import MHA
from pe import PE

class FF(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ff(x)

class Encoder_block(nn.Module):
    def __init__(self, d_model, num_heads, drop_rate, max_len=512, d_ff=2048):
        super().__init__()
        self.mha = MHA(d_model, num_heads, drop_rate, is_masked=False, max_len=max_len)
        self.ff = FF(d_model, d_ff=d_ff)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, mask=None): # x: (batch, seq_q, d_model)
        x1 = self.norm_1(self.dropout(self.mha(x, x, x, mask)) + x)  # (batch, seq_q, d_model)
        x2 = self.norm_2(self.dropout(self.ff(x1)) + x1)
        return x2

class Encoder(nn.Module):
    def __init__(self, num_block, d_model, num_heads, drop_rate, d_ff=2048, max_len=512):
        super().__init__()
        self.pe = PE(d_model, max_len)
        self.dropout = nn.Dropout(drop_rate)
        self.encoder_blocks = nn.ModuleList([Encoder_block(d_model, num_heads, drop_rate, max_len, d_ff) for _ in range(num_block)])

    def forward(self, x, mask=None):
        x = self.pe(x)
        x = self.dropout(x)
        for block in self.encoder_blocks:
            x = block(x, mask)
        return x
