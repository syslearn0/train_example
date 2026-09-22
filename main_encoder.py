from main import execute_training
from encoder import Encoder
from criterion import basic_criterion

num_block = 4
d_model = 128
num_heads = 4
drop_rate = 0.1
model = Encoder(num_block, d_model, num_heads, drop_rate, d_ff=2048, max_len=512)

num_epoch = 1
criterion = 
execute_training(model, num_epoch, criterion, root, run_test=False, smoke=False)