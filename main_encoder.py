import torch
from main import execute_training
from encoder_model import Encoder_Model
from criterion import basic_criterion
from data_utils import VOCAB_SIZE, PAD_ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_block = 4
d_model = 128
num_heads = 4
drop_rate = 0.1
d_ff = 512
max_len = 512
model = Encoder_Model(VOCAB_SIZE, d_model, PAD_ID, num_block, num_heads, drop_rate, d_ff=d_ff, max_len=max_len).to(device)

num_epoch = 6
criterion = basic_criterion
execute_training(model, num_epoch, criterion, "encoder_soft_label", run_test=False, smoke=False, soft_label=True)
