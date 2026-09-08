import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

batch_size = 32
block_size = 256
n_blocks = 12
n_embd = 768
n_heads = 12

device = "cuda" if torch.cuda.is_available() else "cpu"

text = "" 
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda x: [i for i in stoi[x]]
decode = lambda x: "".join(i for i in itos[x]) 

data = torch.tensor(encode(text), dtype = torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
test_data = data[n:]

def get_batch(split = "train"):
    data = train_data if split == "train" else test_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

class LayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(n_embd)
    def forward(self, x):
        return self.ln(x)

class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax = nn.Softmax(dim = -1)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", tensor = torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(n_embd, dim = -1)
        q = q.view(B, T, n_heads, C // n_heads).transpose(1, 2)
        k = k.view(B, T, n_heads, C // n_heads).transpose(1, 2)
        v = v.view(B, T, n_heads, C // n_heads).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        else:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(k.size(-1))
            scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            scores = self.softmax(scores)
            y = scores @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln1 = LayerNorm()
        self.ln2 = LayerNorm()
        self.mlp = MLP()
        self.attn = Attention()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    