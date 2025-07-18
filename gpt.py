import torch
import torch.nn as nn
from torch.nn import functional as F
import glob
import os

text = ""
data_dir = './data/'

if not os.path.exists(data_dir):
    print(f"Directory '{data_dir}' not found.")
    exit()

txt_files = glob.glob(os.path.join(data_dir, '*.txt'))
if not txt_files:
    print(f"No .txt files found in '{data_dir}'. Please place your .txt files inside.")
    exit()

for filepath in txt_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        text += f.read()

#print(text[:500])
# print(len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# mapping string to integer value, reverse for itos
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# entirety of text data encoded to integer format
data = torch.tensor(encode(text), dtype=torch.long)






