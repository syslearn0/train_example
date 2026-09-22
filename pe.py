import torch
from torch import nn

class PE(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos = torch.arange(max_len).unsqueeze(1) # pos: (max_len, 1)
        self.d_i = torch.arange(d_model) # d_i: (d_model,)
        self.d_i[1::2] -= 1
        self.d_i = self.d_i.unsqueeze(0) # d_i: (1, d_model)
        self.d_i = self.d_i/self.d_model 
        self.d_i = 10000 ** self.d_i
        self.register_buffer("pe", self.pos/self.d_i) # pe: (max_len, d_model) # GPUに自動で送られるように
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])
        
    def forward(self, x): 
        assert x.size(1) <= self.max_len, f"seq={x.size(1)} が max_len={self.max_len} を超えています"
        return x + self.pe[:x.size(1)] # 系列長ぶんだけ切り出して加算
