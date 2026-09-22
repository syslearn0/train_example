import torch
from torch import nn

class Basic_Model(nn.Module): # input_ids　-> logits
    def __init__(self, vocab_size, d_model, pad_id): # input_ids: (batch, len)
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.W1 = nn.Linear(d_model, 5)
        self.pad_id = pad_id

    def forward(self, input_ids):
        inputs = self.embedding(input_ids) # (batch, len, d_model)
        mask = (input_ids != self.pad_id) # (batch, len)
        mask_num = mask.float().unsqueeze(-1) # (batch, len, 1)
        inputs = inputs * mask_num # (batch, len, d_model)
        inputs = inputs.sum(dim=1) # (batch, d_model)
        inputs = inputs/(mask.float().sum(dim=1, keepdim=True).clamp(min=1)) # (batch, d_model)/(batch, 1) = (batch, d_model)
        outputs = self.W1(inputs) # (batch, 5)
        return outputs

if __name__ == "__main__":
    vocab_size=1000
    d_model = 8
    pad_id = 0
    batch = 64
    seq_len = 256
    input_ids = torch.randint(0, vocab_size, size=(batch, seq_len))
    model = Basic_Model(vocab_size, d_model, pad_id)
    out = model(input_ids)
    print(out.shape)