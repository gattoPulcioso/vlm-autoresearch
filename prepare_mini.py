“””
prepare_mini.py — Versione leggera per test. DO NOT MODIFY.

Nessun download. Genera dati sintetici in ~30 secondi:

- 500 immagini sintetiche (rumore + forme geometriche semplici)
- Caption generate proceduralmente (“a red circle on white background”, ecc.)
- Features CLIP simulate con PCA-noise realistiche

Usare per testare il loop agente senza aspettare prepare.py.

python prepare_mini.py

API identica a prepare.py — train.py funziona senza modifiche.
“””

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── FIXED CONSTANTS ───────────────────────────────────────────────────────────

CACHE_DIR        = Path(”~/.cache/vlm_autoresearch_mini”).expanduser()
IMAGE_EMBED_DIM  = 512
N_VISUAL_TOKENS  = 50
MAX_CAPTION_LEN  = 32     # più corto del full (64) → train più veloce
VOCAB_SIZE       = 4096   # vocabolario ridotto, caratteri ASCII + parole comuni
PAD_TOKEN_ID     = 0
TIME_BUDGET      = 120    # 2 minuti invece di 5 → iterazioni più rapide
TRAIN_SIZE       = 400
EVAL_SIZE        = 100
EVAL_BATCH_SIZE  = 50

# ─────────────────────────────────────────────────────────────────────────────

_FEATURES_FILE = CACHE_DIR / “features.npy”
_CAPTIONS_FILE = CACHE_DIR / “captions.npy”
_SPLIT_FILE    = CACHE_DIR / “split.npz”

# Vocabolario minimale — parole usate nelle caption sintetiche

_WORDS = [
“<pad>”, “<eos>”, “a”, “an”, “the”, “on”, “with”, “and”, “background”,
“white”, “black”, “red”, “blue”, “green”, “yellow”, “gray”, “dark”, “bright”,
“circle”, “square”, “triangle”, “rectangle”, “line”, “dot”, “cross”,
“small”, “large”, “big”, “tiny”, “multiple”, “single”, “two”, “three”,
“centered”, “left”, “right”, “top”, “bottom”, “corner”,
“solid”, “outline”, “filled”, “striped”, “blurry”, “sharp”,
“image”, “shows”, “contains”, “has”, “is”,
]

# Pad to VOCAB_SIZE con token generici

_WORD2ID = {w: i for i, w in enumerate(_WORDS)}
for i in range(len(_WORDS), VOCAB_SIZE):
_WORD2ID[f”__tok{i}”] = i

def _tokenize(caption: str) -> list[int]:
toks = [_WORD2ID.get(w, 1) for w in caption.lower().split()]
toks = toks[:MAX_CAPTION_LEN - 1] + [1]  # 1 = <eos>
toks += [PAD_TOKEN_ID] * (MAX_CAPTION_LEN - len(toks))
return toks

def _make_caption(rng: np.random.Generator) -> str:
color  = rng.choice([“red”, “blue”, “green”, “yellow”, “gray”, “black”])
shape  = rng.choice([“circle”, “square”, “triangle”, “rectangle”, “dot”, “cross”])
bg     = rng.choice([“white”, “black”, “dark”, “bright”])
size   = rng.choice([“small”, “large”, “tiny”, “big”])
pos    = rng.choice([“centered”, “left”, “right”, “top”, “bottom”])
extras = rng.choice([“solid”, “outline”, “filled”, “blurry”, “sharp”])
return f”a {size} {color} {shape} {extras} {pos} on {bg} background”

def _make_features(rng: np.random.Generator, n: int) -> np.ndarray:
“””
Simula features CLIP realistiche:
- CLS token (index 0): media pesata dei patch token + bias
- Patch tokens (1..49): rumore strutturato con correlazione spaziale
“””
feats = rng.standard_normal((n, N_VISUAL_TOKENS, IMAGE_EMBED_DIM)).astype(np.float32)
# correlazione spaziale leggera fra patch vicini
feats[:, 1:, :] = 0.6 * feats[:, 1:, :] + 0.4 * feats[:, :-1, :]
# CLS = media patch + rumore
feats[:, 0, :] = feats[:, 1:, :].mean(axis=1) + 0.1 * rng.standard_normal((n, IMAGE_EMBED_DIM))
# normalizza L2 per token (come CLIP reale)
norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-6)
return (feats / norms).astype(np.float16)

def prepare() -> None:
CACHE_DIR.mkdir(parents=True, exist_ok=True)
N = TRAIN_SIZE + EVAL_SIZE
rng = np.random.default_rng(42)

```
if not _FEATURES_FILE.exists():
    t0 = time.time()
    print(f"Generating {N} synthetic image features …")
    feats = _make_features(rng, N)
    np.save(_FEATURES_FILE, feats)
    print(f"  → {_FEATURES_FILE}  shape={feats.shape}  ({time.time()-t0:.1f}s)")
else:
    print(f"[skip] features already exist.")

if not _CAPTIONS_FILE.exists():
    print(f"Generating {N} synthetic captions …")
    captions = np.array([_tokenize(_make_caption(rng)) for _ in range(N)], dtype=np.int32)
    np.save(_CAPTIONS_FILE, captions)
    print(f"  → {_CAPTIONS_FILE}  shape={captions.shape}")
else:
    print(f"[skip] captions already exist.")

if not _SPLIT_FILE.exists():
    idx = rng.permutation(N)
    np.savez(_SPLIT_FILE, train=idx[:TRAIN_SIZE], val=idx[TRAIN_SIZE:])
    print(f"  → {_SPLIT_FILE}  train={TRAIN_SIZE}  val={EVAL_SIZE}")
else:
    print(f"[skip] split already exists.")

print("prepare_mini() done — ~30s total, no downloads needed.")
```

# ── runtime cache ─────────────────────────────────────────────────────────────

_cache: dict = {}

def _load_data():
if not _cache:
assert _FEATURES_FILE.exists(), “Run `python prepare_mini.py` first.”
_cache[“features”] = np.load(_FEATURES_FILE)
_cache[“captions”] = np.load(_CAPTIONS_FILE)
split = np.load(_SPLIT_FILE)
_cache[“train_idx”] = split[“train”]
_cache[“val_idx”]   = split[“val”]
return _cache

def get_batch(split: str, batch_size: int, device: str):
d = _load_data()
idx = d[f”{split}_idx”]
chosen = np.random.choice(idx, batch_size, replace=True)
img = torch.from_numpy(d[“features”][chosen].astype(np.float32)).to(device)
cap = torch.from_numpy(d[“captions”][chosen].astype(np.int64)).to(device)
return img, cap

def evaluate_val_loss(model: torch.nn.Module, device: str) -> dict:
d = _load_data()
val_idx = d[“val_idx”]
model.eval()
total_loss = 0.0
total_tok  = 0
with torch.no_grad():
for i in range(0, len(val_idx), EVAL_BATCH_SIZE):
batch_idx = val_idx[i : i + EVAL_BATCH_SIZE]
img = torch.from_numpy(d[“features”][batch_idx].astype(np.float32)).to(device)
cap = torch.from_numpy(d[“captions”][batch_idx].astype(np.int64)).to(device)
non_pad = (cap != PAD_TOKEN_ID).sum().item()
loss = model(img, cap)
total_loss += loss.item() * non_pad
total_tok  += non_pad
model.train()
avg = total_loss / max(total_tok, 1)
bpb = avg / math.log(2) / 3.5
return {“val_loss”: avg, “val_bpb”: bpb, “eval_tokens”: total_tok}

if **name** == “**main**”:
prepare()
