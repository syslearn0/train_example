import torch
from torch import optim
from data_utils import train_loader, val_loader, test_loader, make_loader
from basic_model import Basic_Model
from train import training, evaluating
from criterion import basic_criterion
from data_utils import VOCAB_SIZE, PAD_ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def execute_training(model, num_epoch, criterion, root, run_test=False, smoke=False, soft_label=False):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if smoke:
        tr = make_loader("train", shuffle=True, limit=10000)
        va = make_loader("validation", limit=1000)
    else:
        tr, va = train_loader, val_loader

    training(num_epoch, tr, va, model, criterion, optimizer, root, soft_label=soft_label)

    if run_test:
        evaluating(test_loader, model, criterion, val=False)

if __name__ == "__main__":
    model = Basic_Model(VOCAB_SIZE, 128, PAD_ID).to(device)
    execute_training(model, 15, basic_criterion, "basic_hard", soft_label=False)
    model2 = Basic_Model(VOCAB_SIZE, 128, PAD_ID).to(device)
    execute_training(model2, 15, basic_criterion, "basic_soft", soft_label=True)
