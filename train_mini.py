“””
train_mini.py — Versione leggera per test del loop agente.

Differenze rispetto a train.py:

- Import da prepare_mini (dati sintetici, no COCO)
- Modello più piccolo (default: 64-dim, 2 layer)
- TIME_BUDGET 2 minuti → giri completi in ~3 min totali
- Nessun torch.compile (startup istantaneo)

L’agente modifica questo file esattamente come farebbe con train.py.
“””

import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_mini import (
IMAGE_EMBED_DIM, MAX_CAPTION_LEN, N_VISUAL_TOKENS,
PAD_TOKEN_ID, TIME_BUDGET, VOCAB_SIZE,
evaluate_val_loss, get_batch,
)

# ── HYPERPARAMETERS — modify freely ──────────────────────────────────────────

LM_DIM       = 64      # molto piccolo per test rapidi
DEPTH        = 2
N_HEADS      = 2
N_VIS_USED   = 1
BATCH_SIZE   = 64
LR           = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50

# ─────────────────────────────────────────────────────────────────────────────

DEVICE = “cuda” if torch.cuda.is_available() else “cpu”

class CausalSelfAttention(nn.Module):
def **init**(self):
super().**init**()
assert LM_DIM % N_HEADS == 0
self.n_heads  = N_HEADS
self.head_dim = LM_DIM // N_HEADS
self.qkv  = nn.Linear(LM_DIM, 3 * LM_DIM, bias=False)
self.proj = nn.Linear(LM_DIM, LM_DIM, bias=False)

```
def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.qkv(x).split(LM_DIM, dim=2)
    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
```

class MLP(nn.Module):
def **init**(self):
super().**init**()
self.fc1 = nn.Linear(LM_DIM, 4 * LM_DIM, bias=False)
self.fc2 = nn.Linear(4 * LM_DIM, LM_DIM, bias=False)

```
def forward(self, x):
    return self.fc2(F.gelu(self.fc1(x)))
```

class Block(nn.Module):
def **init**(self):
super().**init**()
self.ln1  = nn.LayerNorm(LM_DIM)
self.attn = CausalSelfAttention()
self.ln2  = nn.LayerNorm(LM_DIM)
self.mlp  = MLP()

```
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x
```

class VLM(nn.Module):
def **init**(self):
super().**init**()
seq_len = N_VIS_USED + MAX_CAPTION_LEN
self.vis_proj = nn.Linear(IMAGE_EMBED_DIM, LM_DIM, bias=False)
self.tok_emb  = nn.Embedding(VOCAB_SIZE, LM_DIM)
self.pos_emb  = nn.Embedding(seq_len, LM_DIM)
self.blocks   = nn.ModuleList([Block() for _ in range(DEPTH)])
self.ln_f     = nn.LayerNorm(LM_DIM)
self.head     = nn.Linear(LM_DIM, VOCAB_SIZE, bias=False)
self.head.weight = self.tok_emb.weight
self._init_weights()

```
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

def forward(self, img_features, caption_tokens):
    vis = self.vis_proj(img_features[:, :N_VIS_USED, :])
    txt = self.tok_emb(caption_tokens)
    x   = torch.cat([vis, txt], dim=1)
    x   = x + self.pos_emb(torch.arange(x.shape[1], device=x.device))
    for block in self.blocks:
        x = block(x)
    x = self.ln_f(x)
    logits = self.head(x[:, N_VIS_USED - 1 : N_VIS_USED + MAX_CAPTION_LEN - 1, :])
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        caption_tokens.reshape(-1),
        ignore_index=PAD_TOKEN_ID,
    )
    return loss
```

def make_lr_lambda(warmup, total):
def fn(step):
if step < warmup:
return step / max(warmup, 1)
t = (step - warmup) / max(total - warmup, 1)
return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
return fn

def main():
torch.cuda.reset_peak_memory_stats()
t_total_start = time.time()

```
model = VLM().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"VLM-mini | params={n_params:.2f}M | depth={DEPTH} | dim={LM_DIM} | device={DEVICE}")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR,
    weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, make_lr_lambda(WARMUP_STEPS, 5000)
)

step = 0
total_tokens = 0
running_loss = 0.0
LOG_EVERY    = 50
t_train_start = None

try:
    while True:
        img, cap = get_batch("train", BATCH_SIZE, DEVICE)
        if t_train_start is None:
            t_train_start = time.time()
        if time.time() - t_train_start >= TIME_BUDGET:
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            loss = model(img, cap)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss  += loss.item()
        total_tokens  += BATCH_SIZE * MAX_CAPTION_LEN
        step          += 1

        if step % LOG_EVERY == 0:
            print(f"step {step:5d} | loss {running_loss/LOG_EVERY:.4f} | {time.time()-t_train_start:.0f}s")
            running_loss = 0.0

except torch.cuda.OutOfMemoryError:
    print("CUDA OutOfMemoryError — run counts as crash.")
    print("val_loss: OOM")
    sys.exit(1)

t_end = time.time()
training_seconds = t_end - (t_train_start or t_end)
total_seconds    = t_end - t_total_start
peak_vram_mb     = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0.0

print("Evaluating …")
res = evaluate_val_loss(model, DEVICE)

print("---")
print(f"val_loss:         {res['val_loss']:.6f}")
print(f"val_bpb:          {res['val_bpb']:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.2f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {n_params:.2f}")
print(f"depth:            {DEPTH}")
print(f"lm_dim:           {LM_DIM}")
print(f"n_vis_used:       {N_VIS_USED}")
```

if **name** == “**main**”:
main()
