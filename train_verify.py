"""
Train the same tiny transformer with PyTorch SGD, export initial weights
and per-step losses for comparison with transformer_train.cpp.

Usage:
  1. python train_verify.py          -- exports weights.bin, trains, writes py_losses.txt
  2. g++ -O2 -o transformer_train transformer_train.cpp -lm
  3. ./transformer_train             -- loads weights.bin, trains, compares against py_losses.txt
"""

import torch
import torch.nn as nn
import numpy as np

# ─── Dimensions (must match transformer_train.cpp) ───────────────────────────
SEQ_LEN  = 16
N_LAYERS = 4
D_MODEL  = 32
N_HEADS  = 4
D_HEAD   = D_MODEL // N_HEADS
D_MLP    = D_MODEL * 4
VOCAB    = 256

# ─── Model definition (identical to train_and_export.py) ─────────────────────

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_HEAD, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_HEAD, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_HEAD, bias=False)

    def forward(self, x):
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        scale = D_HEAD ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        seq_len = x.shape[0]
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead() for _ in range(N_HEADS)])
        self.W_O   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ln_attn = nn.LayerNorm(D_MODEL)
        self.ln_mlp  = nn.LayerNorm(D_MODEL)
        self.mlp1  = nn.Linear(D_MODEL, D_MLP, bias=False)
        self.mlp2  = nn.Linear(D_MLP, D_MODEL, bias=False)

    def forward(self, x):
        normed = self.ln_attn(x)
        head_outputs = torch.cat([h(normed) for h in self.heads], dim=-1)
        x = x + self.W_O(head_outputs)
        normed = self.ln_mlp(x)
        hidden = torch.relu(self.mlp1(normed))
        x = x + self.mlp2(hidden)
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, D_MODEL)
        self.layers    = nn.ModuleList([TransformerLayer() for _ in range(N_LAYERS)])
        self.ln_final  = nn.LayerNorm(D_MODEL)
        self.unembed   = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.unembed(x)

# ─── Initialize model ───────────────────────────────────────────────────────
torch.manual_seed(42)
model = Transformer()

# ─── Export initial weights to weights.bin ───────────────────────────────────
def write_array(f, t):
    arr = t.detach().numpy().astype(np.float32)
    f.write(arr.tobytes())

print("Exporting initial weights to weights.bin ...")
with open("weights.bin", "wb") as f:
    write_array(f, model.embedding.weight)
    for layer_idx, layer in enumerate(model.layers):
        for h in range(N_HEADS):
            write_array(f, layer.heads[h].W_Q.weight)
            write_array(f, layer.heads[h].W_K.weight)
            write_array(f, layer.heads[h].W_V.weight)
        write_array(f, layer.W_O.weight)
        write_array(f, layer.mlp1.weight)
        write_array(f, layer.mlp2.weight)
        write_array(f, layer.ln_attn.weight)
        write_array(f, layer.ln_attn.bias)
        write_array(f, layer.ln_mlp.weight)
        write_array(f, layer.ln_mlp.bias)
    write_array(f, model.ln_final.weight)
    write_array(f, model.ln_final.bias)
    write_array(f, model.unembed.weight)
print("Done.\n")

# ─── Train with SGD ─────────────────────────────────────────────────────────
tokens  = torch.tensor([i + 1 for i in range(SEQ_LEN)], dtype=torch.long)
targets = torch.tensor([i + 2 for i in range(SEQ_LEN)], dtype=torch.long)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

N_STEPS = 50
losses = []

print(f"Training for {N_STEPS} steps with SGD (lr=0.01)\n")
for step in range(N_STEPS):
    optimizer.zero_grad()
    logits = model(tokens)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 5 == 0 or step == N_STEPS - 1:
        print(f"  step {step:3d}: loss {loss.item():.6f}")

# ─── Save losses ────────────────────────────────────────────────────────────
with open("py_losses.txt", "w") as f:
    for l in losses:
        f.write(f"{l:.6f}\n")
print(f"\nPyTorch losses written to py_losses.txt")
print(f"Initial loss: {losses[0]:.6f}, final loss: {losses[-1]:.6f}")
print(f"\nNext: compile and run transformer_train to compare.")
