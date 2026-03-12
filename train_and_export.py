import torch
import torch.nn as nn
import numpy as np
import struct

# ─── Dimensions (must match transformer_forward.cpp) ─────────────────────────
SEQ_LEN  = 16
N_LAYERS = 4
D_MODEL  = 32
N_HEADS  = 4
D_HEAD   = D_MODEL // N_HEADS
D_MLP    = D_MODEL * 4
VOCAB    = 256

# ─── Model definition using explicit separate Q, K, V matrices ───────────────

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_HEAD, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_HEAD, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_HEAD, bias=False)

    def forward(self, x):
        # x: [seq_len, d_model]
        q = self.W_Q(x)  # [seq_len, d_head]
        k = self.W_K(x)
        v = self.W_V(x)
        scale = D_HEAD ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [seq_len, seq_len]
        # Causal mask
        seq_len = x.shape[0]
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)  # [seq_len, d_head]

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
        # Attention sublayer
        normed = self.ln_attn(x)
        head_outputs = torch.cat([h(normed) for h in self.heads], dim=-1)
        x = x + self.W_O(head_outputs)
        # MLP sublayer
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
        x = self.embedding(tokens)  # [seq_len, d_model]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.unembed(x)      # [seq_len, vocab]

# ─── Train briefly on random data ─────────────────────────────────────────────

torch.manual_seed(42)
model = Transformer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("Training...")
for step in range(200):
    tokens = torch.randint(0, VOCAB, (SEQ_LEN,))
    targets = torch.randint(0, VOCAB, (SEQ_LEN,))
    logits = model(tokens)
    loss = loss_fn(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"  step {step}: loss {loss.item():.4f}")

print("Training done.\n")

# ─── Run inference on a fixed input and print logits ─────────────────────────

TEST_TOKENS = list(range(1, SEQ_LEN + 1))
tokens_tensor = torch.tensor(TEST_TOKENS, dtype=torch.long)

model.eval()
with torch.no_grad():
    ref_logits = model(tokens_tensor)

print("PyTorch logits (first position, first 8 vocab entries):")
print(ref_logits[0, :8].numpy())
print()

# ─── Export weights to binary file ───────────────────────────────────────────
# Layout matches exactly what the C++ loader expects (see load_weights() there).
# All weights are written as float32, row-major.

def write_array(f, t):
    """Write a tensor to file as raw float32."""
    arr = t.detach().numpy().astype(np.float32)
    f.write(arr.tobytes())

print("Exporting weights to weights.bin ...")
with open("weights.bin", "wb") as f:
    # Token embedding: [VOCAB, D_MODEL]
    write_array(f, model.embedding.weight)

    for layer_idx, layer in enumerate(model.layers):
        # Q, K, V for each head: [D_HEAD, D_MODEL] each
        for h in range(N_HEADS):
            write_array(f, layer.heads[h].W_Q.weight)
            write_array(f, layer.heads[h].W_K.weight)
            write_array(f, layer.heads[h].W_V.weight)
        # W_O: [D_MODEL, D_MODEL]
        write_array(f, layer.W_O.weight)
        # MLP weights: [D_MLP, D_MODEL] and [D_MODEL, D_MLP]
        write_array(f, layer.mlp1.weight)
        write_array(f, layer.mlp2.weight)
        # Layer norm attn: scale [D_MODEL], bias [D_MODEL]
        write_array(f, layer.ln_attn.weight)
        write_array(f, layer.ln_attn.bias)
        # Layer norm mlp
        write_array(f, layer.ln_mlp.weight)
        write_array(f, layer.ln_mlp.bias)

    # Final layer norm
    write_array(f, model.ln_final.weight)
    write_array(f, model.ln_final.bias)

    # Unembed: [VOCAB, D_MODEL]
    write_array(f, model.unembed.weight)

print("Weights written.\n")

# Also save reference logits for all positions to a text file for comparison
with open("ref_logits.txt", "w") as f:
    for pos in range(SEQ_LEN):
        for v in range(VOCAB):
            f.write(f"{ref_logits[pos, v].item():.6f}\n")

print("Reference logits written to ref_logits.txt")
