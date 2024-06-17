import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
train_input_file_path = os.path.join(os.path.dirname(__file__), 'TinyStoriesV2-GPT4-train-cleaned.txt')
val_input_file_path = os.path.join(os.path.dirname(__file__), 'TinyStoriesV2-GPT4-valid-cleaned.txt')

with open(train_input_file_path, 'r', encoding='utf-8') as f:
    train_data = f.read()

with open(val_input_file_path, 'r', encoding='utf-8') as f:
    val_data = f.read()

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
