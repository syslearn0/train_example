from datasets import load_dataset
from torch.utils.data import DataLoader
import subprocess
import sys
import json
import torch
from pathlib import Path

MAX_LEN = 256
SIGMA = 0.7
PAD_ID = 0

# Parquet形式で直接配信されている日本語Amazonレビューデータを読み込む
dataset = load_dataset("SetFit/amazon_reviews_multi_ja")

# トークン対応のvocab_listを取り込む
BASE = Path(__file__).resolve().parent
VOCAB_JSON_PATH = BASE / "vocab.json"
VOCAB_SCRIPT_PATH = BASE / "vocab.py"

if not VOCAB_JSON_PATH.exists():
    print("jsonが存在しないため作成します")
    subprocess.run([sys.executable, str(VOCAB_SCRIPT_PATH)], check=True)

with open(VOCAB_JSON_PATH, "r", encoding="utf-8") as f:
    char2id = json.load(f)

VOCAB_SIZE = len(char2id)

def tokenize(data):
    tokenized_data = []
    for sentences in data:
        id_list = []
        for idx, char in enumerate(list(sentences)):
            if idx == MAX_LEN:
                break
            elif char in char2id:
                id_list.append(char2id[char])
            else:
                id_list.append(char2id["<unk>"])
        tokenized_data.append(id_list)
    return tokenized_data


# soft label
idx = torch.arange(5).float()
d = (idx.view(-1, 1) - idx.view(1, -1)) ** 2 
SOFT_TABLE = torch.exp(-d/(2*SIGMA**2))
SOFT_TABLE = SOFT_TABLE/SOFT_TABLE.sum(dim=1, keepdim=True)

def collate_fn(batch):
    texts = [i["text"] for i in batch]
    labels = [i["label"] for i in batch]
    input_ids = tokenize(texts)
    seqs = [torch.tensor(input_id, dtype=torch.long) for input_id in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        seqs,
        batch_first=True,
        padding_value=PAD_ID
    )
    mask = (input_ids != PAD_ID)
    labels = torch.tensor(labels, dtype=torch.long)

    return {"input_ids":input_ids, "mask":mask, "label":labels, "soft_label":SOFT_TABLE[labels]}

# loaderの追加
def make_loader(split, batch_size=64, shuffle=False, limit=None):
    ds = dataset[split]
    if limit is not None:
        ds = ds.shuffle(seed=42).select(range(limit))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

train_loader = make_loader("train", shuffle=True)
val_loader = make_loader("validation")
test_loader = make_loader("test")

if __name__ == "__main__":
    batch = next(iter(train_loader))
    print(batch["input_ids"].shape)    # (64, 何か) になっているか
    print(batch["input_ids"].dtype)    # torch.int64 か
    print(batch["label"].shape)        # (64,) か
    print(batch["label"].min(), batch["label"].max())   # 0〜4 の範囲か
    print(batch["mask"].sum(dim=1)[:5])  # 各文の実際の長さ。元の文と一致するか
