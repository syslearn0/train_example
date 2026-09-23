import torch
from torch import nn
from encoder import Encoder

class Encoder_Model(nn.Module): # input_ids　-> logits
    def __init__(self, vocab_size, d_model, pad_id, num_block, num_heads, drop_rate, d_ff=2048, max_len=512): # input_ids: (batch, len)
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.encoder = Encoder(num_block, d_model, num_heads, drop_rate, d_ff=d_ff, max_len=max_len)
        self.W1 = nn.Linear(d_model, 5)
        self.pad_id = pad_id

    def forward(self, input_ids):
        inputs = self.embedding(input_ids) # (batch, len, d_model)
        mask = (input_ids != self.pad_id) # (batch, len)
        attn_mask = mask[:, None, None, :] # (batch, 1, 1, len) MHAのスコアが(batch, num_heads, seq_q, seq_k)の4次元なので合わせる
        inputs = self.encoder(inputs, attn_mask) # (batch, len, d_model)
        mask_num = mask.float().unsqueeze(-1) # (batch, len, 1)
        inputs = inputs * mask_num # (batch, len, d_model)
        inputs = inputs.sum(dim=1) # (batch, d_model)
        inputs = inputs/(mask.float().sum(dim=1, keepdim=True).clamp(min=1)) # (batch, d_model)/(batch, 1) = (batch, d_model)
        outputs = self.W1(inputs) # (batch, 5)
        return outputs

if __name__ == "__main__":
    vocab_size = 1000
    d_model = 8
    pad_id = 0
    batch = 64
    seq_len = 256
    input_ids = torch.randint(0, vocab_size, size=(batch, seq_len))
    model = Encoder_Model(vocab_size, d_model, pad_id, num_block=2, num_heads=2, drop_rate=0.1, d_ff=32, max_len=512)
    out = model(input_ids)
    print(out.shape)
