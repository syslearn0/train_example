from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

def accuracy(pred, label):
    return (pred.argmax(dim=-1) == label).float().sum().item()

def evaluating(loader, model, criterion):
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        num_idx = 0
        for batch in loader:
            text = batch["input_ids"]
            label = batch["label"]
            pred = model(text)
            loss = criterion(pred, label)
            loss_sum += loss * text.size(0)
            acc_sum += accuracy(pred, label)
            num_idx += text.size(0)
        loss_avg = loss_sum/num_idx
        acc_avg = acc_sum/num_idx
        print(f"\n-------VALIDATION----------\nloss_avg:{loss_avg}\nacc_avg:{acc_avg}")
        return loss_avg, acc_avg

def training(num_epoch, train_loader, val_loader, model, criterion, optimizer, root):
    writer = SummaryWriter(f"runs/{root}")
    for epoch in range(num_epoch):
        print(f"=============EPOCH:{epoch}==================")
        for idx, batch in enumerate(tqdm(train_loader)): # batch: (64, 4)
            model.train()
            input_ids = batch["input_ids"]
            label = batch["label"]
            optimizer.zero_grad()
            pred = model(input_ids)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            global_step = epoch * len(train_loader) + idx
            if idx%100 == 0:
                acc = accuracy(pred, label)
                print(f"\n--------TRAIN(epoch:{epoch}, batch:{idx})--------\nloss:{loss.item()}\nacc:{acc}")
                writer.add_scalar("loss/train", loss, global_step)
                writer.add_scalar("acc_rate/train", acc/input_ids.size(0), global_step)

        val_loss, val_acc = evaluating(val_loader, model, criterion)
        writer.add_scalar("loss/val", val_loss, global_step)
        writer.add_scalar("acc_rate/val", val_acc, global_step)
        print("""⠀∧,,,∧
( ̳- ω - ̳) zzZ
/    づ～""")
    writer.close()
