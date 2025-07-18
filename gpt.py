import torch
import torch.nn as nn
from torch.nn import functional as F
import glob
import os

torch.manual_seed(8647)
batch_size = 8
context_length = 16

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

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size, )) # batch_size number of random number for offset in data
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix]) # targets beginning one index after first token
    return x,y

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape, xb) # batch size x context length dim

# yb is essentially xb shifted up one index

for b in range(batch_size):
    for t in range(context_length): # time dim
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"{context.tolist()}\t{target}")

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target):
        logits = self.token_embedding_table(idx) # (B, T, C)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        loss = F.cross_entropy(logits, target) # requires 2dim
        return logits, loss

m = BigramLanguageModel(vocab_size)
out, loss = m(xb, yb)
print(out.shape)