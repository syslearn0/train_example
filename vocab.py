import json
from collections import Counter
from datasets import load_dataset
from pathlib import Path

BASE = Path(__file__).resolve().parent
VOCAB_JSON_PATH = BASE / "vocab.json"

dataset = load_dataset("SetFit/amazon_reviews_multi_ja")

# 語彙のsetを作成
char_count = Counter()
for text in dataset["train"]["text"]:
    char_count.update(text)

# 5回以上登場したものを抽出
vocab_set = set()
MIN_COUNT = 5
for char, count in char_count.items():
    if count >= MIN_COUNT:
        vocab_set.add(char)

# 特殊トークンを追加
special_tokens = ["<pad>", "<unk>"]
vocab_list = special_tokens + sorted(vocab_set)

# トークン化
char2id = {}
for idx, char in enumerate(vocab_list):
    char2id[char] = idx

with open("VOCAB_JSON_PATH", "w", encoding="utf-8") as f:
    json.dump(char2id, f, ensure_ascii=False, indent=2)

print("jsonに保存しました")