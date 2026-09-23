from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import subprocess

try:
    frame = subprocess.run(
        ["chafa", "--size=90x45", "image.png"],
        capture_output=True, text=True, check=True
    ).stdout
except (subprocess.CalledProcessError, FileNotFoundError):
    frame = ""      # chafaが無い/失敗したら飾りなしで続行

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(pred, label):
    return (pred.argmax(dim=-1) == label).float().sum().item()

def mae(pred, label):
    return (pred.argmax(dim=-1) - label).abs().float().sum().item()

def evaluating(loader, model, criterion, val=True):
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        num_idx = 0
        mae_sum = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)
            pred = model(input_ids)
            loss = criterion(pred, label)
            loss_sum += loss * input_ids.size(0)
            acc_sum += accuracy(pred, label)
            num_idx += input_ids.size(0)
            mae_sum += mae(pred, label)
        loss_avg = loss_sum/num_idx
        acc_avg = acc_sum/num_idx
        mae_avg = mae_sum/num_idx
        if val:
            print(f"\n-------VALIDATION----------\nloss_avg:{loss_avg}\nacc_avg:{acc_avg}\nmae_avg:{mae_avg}")
        if not val:
            print(f"\n=========TEST===========\nloss_avg:{loss_avg}\nacc_avg:{acc_avg}\nmae_avg:{mae_avg}")
        return loss_avg, acc_avg, mae_avg

def training(num_epoch, train_loader, val_loader, model, criterion, optimizer, root, soft_label=False):
    best_loss = 1e10
    writer = SummaryWriter(f"runs/{root}")
    for epoch in range(num_epoch):
        print(f"=============EPOCH:{epoch}==================")
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)): # batch: (64, 4)
            input_ids = batch["input_ids"].to(device)
            if soft_label:
                label = batch["soft_label"].to(device)
            else:
                label = batch["label"].to(device)
            optimizer.zero_grad()
            pred = model(input_ids)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            global_step = epoch * len(train_loader) + idx
            if idx%100 == 0:
                acc = accuracy(pred, batch["label"])
                mae_sum = mae(pred, batch["label"])
                tqdm.write(f"\n--------TRAIN(epoch:{epoch}, batch:{idx})--------\nloss:{loss.item()}\nacc:{acc/input_ids.size(0)}\nmae:{mae_sum/input_ids.size(0)}")
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("acc_rate/train", acc/input_ids.size(0), global_step)
                writer.add_scalar("mae/train", mae_sum/input_ids.size(0), global_step)

        val_loss, val_acc, val_mae_avg = evaluating(val_loader, model, criterion)
        writer.add_scalar("loss/val", val_loss, global_step)
        writer.add_scalar("acc_rate/val", val_acc, global_step)
        writer.add_scalar("mae/val", val_mae_avg, global_step)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"runs/{root}/best.pt")
            print("\n===================\nbest model更新しといたぜ!!!\n==========================")
            print(frame)
    writer.close()
