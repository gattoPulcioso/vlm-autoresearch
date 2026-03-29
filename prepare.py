“””
prepare.py — Fixed infrastructure. DO NOT MODIFY.

One-time setup:
python prepare.py

Downloads COCO val2017 images + captions (~1 GB), precomputes CLIP ViT-B/32
features for every image, tokenizes captions with tiktoken, saves everything
to ~/.cache/vlm_autoresearch/. Subsequent runs are instant (files exist).

Runtime API used by train.py:
get_batch(split, batch_size, device)  →  (img_features, caption_tokens)
evaluate_val_loss(model, device)      →  {“val_loss”: float, …}
“””

import json
import math
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── FIXED CONSTANTS — train.py can read these but must not change them ────────

CACHE_DIR         = Path(”~/.cache/vlm_autoresearch”).expanduser()
CLIP_MODEL_ID     = “openai/clip-vit-base-patch32”
IMAGE_EMBED_DIM   = 512    # CLIP ViT-B/32 hidden dim
N_VISUAL_TOKENS   = 50     # 1 CLS + 49 spatial patches (7×7 grid)
MAX_CAPTION_LEN   = 64     # tokens per caption (pad/truncate)
VOCAB_SIZE        = 50257  # tiktoken cl100k_base
PAD_TOKEN_ID      = 50256  # <|endoftext|> used as padding
TIME_BUDGET       = 300    # wall-clock seconds of training
TRAIN_SIZE        = 4000
EVAL_SIZE         = 1000
EVAL_BATCH_SIZE   = 64

# ─────────────────────────────────────────────────────────────────────────────

_IMAGES_URL = “http://images.cocodataset.org/zips/val2017.zip”
_ANNOTS_URL = “http://images.cocodataset.org/annotations/annotations_trainval2017.zip”

_IMAGES_DIR     = CACHE_DIR / “val2017”
_FEATURES_FILE  = CACHE_DIR / “features.npy”   # [N, 50, 512]  float16
_CAPTIONS_FILE  = CACHE_DIR / “captions.npy”   # [N, 64]       int32
_SPLIT_FILE     = CACHE_DIR / “split.npz”       # train/val indices

# ── helpers ───────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> None:
if dest.exists():
print(f”  [skip] {dest.name} already downloaded.”)
return
dest.parent.mkdir(parents=True, exist_ok=True)
print(f”  Downloading {url} …”)
urllib.request.urlretrieve(url, str(dest))
print(f”  → {dest}”)

def _extract(zip_path: Path, dest_dir: Path, sentinel: Path) -> None:
if sentinel.exists():
print(f”  [skip] already extracted to {dest_dir.name}/”)
return
print(f”  Extracting {zip_path.name} …”)
with zipfile.ZipFile(zip_path) as zf:
zf.extractall(dest_dir)
print(f”  → {dest_dir}”)

# ── main prepare routine ──────────────────────────────────────────────────────

def prepare() -> None:
CACHE_DIR.mkdir(parents=True, exist_ok=True)

```
# 1. Images
zip_images = CACHE_DIR / "val2017.zip"
_download(_IMAGES_URL, zip_images)
_extract(zip_images, CACHE_DIR, _IMAGES_DIR)

# 2. Annotations
zip_annots = CACHE_DIR / "annotations_trainval2017.zip"
_download(_ANNOTS_URL, zip_annots)
annots_dir = CACHE_DIR / "annotations"
_extract(zip_annots, CACHE_DIR, annots_dir)

# 3. Build (image_path, caption) list
captions_json = annots_dir / "captions_val2017.json"
with open(captions_json) as f:
    raw = json.load(f)
id2file = {img["id"]: img["file_name"] for img in raw["images"]}
id2cap: dict = {}
for ann in raw["annotations"]:
    iid = ann["image_id"]
    if iid not in id2cap:
        id2cap[iid] = ann["caption"]
samples = [
    (_IMAGES_DIR / id2file[iid], cap)
    for iid, cap in id2cap.items()
    if (_IMAGES_DIR / id2file[iid]).exists()
]
samples = samples[: TRAIN_SIZE + EVAL_SIZE]
print(f"  {len(samples)} image-caption pairs loaded.")

# 4. Tokenise captions
if not _CAPTIONS_FILE.exists():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    all_tokens = np.full(
        (len(samples), MAX_CAPTION_LEN), PAD_TOKEN_ID, dtype=np.int32
    )
    for i, (_, cap) in enumerate(samples):
        toks = enc.encode(cap)[:MAX_CAPTION_LEN]
        all_tokens[i, : len(toks)] = toks
    np.save(_CAPTIONS_FILE, all_tokens)
    print(f"  Captions saved → {_CAPTIONS_FILE}")
else:
    print(f"  [skip] captions already tokenised.")

# 5. Pre-compute CLIP features
if not _FEATURES_FILE.exists():
    from PIL import Image as PILImage
    from transformers import CLIPModel, CLIPProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading CLIP on {dev} …")
    clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    N = len(samples)
    feats = np.zeros((N, N_VISUAL_TOKENS, IMAGE_EMBED_DIM), dtype=np.float16)
    BSZ = 64
    t0 = time.time()
    for i in range(0, N, BSZ):
        paths = [s[0] for s in samples[i : i + BSZ]]
        imgs = [PILImage.open(p).convert("RGB") for p in paths]
        inp = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            # last_hidden_state: [B, N_VISUAL_TOKENS, IMAGE_EMBED_DIM]
            out = clip.vision_model(**inp).last_hidden_state
        feats[i : i + len(paths)] = out.cpu().to(torch.float16).numpy()
        if (i // BSZ) % 10 == 0:
            pct = 100 * (i + BSZ) / N
            print(f"    {i+BSZ}/{N}  ({pct:.0f}%)  {time.time()-t0:.0f}s")
    np.save(_FEATURES_FILE, feats)
    print(f"  Features saved → {_FEATURES_FILE}  shape={feats.shape}")
    del clip
else:
    print(f"  [skip] features already computed.")

# 6. Train/val split
if not _SPLIT_FILE.exists():
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    np.savez(_SPLIT_FILE, train=idx[:TRAIN_SIZE], val=idx[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE])
    print(f"  Split saved → {_SPLIT_FILE}  (train={TRAIN_SIZE}, val={EVAL_SIZE})")
else:
    print(f"  [skip] split already saved.")

print("prepare() complete.")
```

# ── runtime cache (populated once per process) ────────────────────────────────

_cache: dict = {}

def _load_data() -> tuple:
if not _cache:
assert _FEATURES_FILE.exists(), “Run `python prepare.py` first.”
_cache[“features”] = np.load(_FEATURES_FILE)   # [N, 50, 512] float16
_cache[“captions”] = np.load(_CAPTIONS_FILE)   # [N, 64]      int32
split = np.load(_SPLIT_FILE)
_cache[“train_idx”] = split[“train”]
_cache[“val_idx”]   = split[“val”]
return _cache

# ── public API ────────────────────────────────────────────────────────────────

def get_batch(split: str, batch_size: int, device: str):
“””
Returns (img_features, caption_tokens) ready for the model.
img_features:    FloatTensor [B, N_VISUAL_TOKENS, IMAGE_EMBED_DIM]
caption_tokens:  LongTensor  [B, MAX_CAPTION_LEN]
“””
d = _load_data()
idx = d[f”{split}_idx”]
chosen = np.random.choice(idx, batch_size, replace=True)
img = torch.from_numpy(d[“features”][chosen].astype(np.float32)).to(device)
cap = torch.from_numpy(d[“captions”][chosen].astype(np.int64)).to(device)
return img, cap

def evaluate_val_loss(model: torch.nn.Module, device: str) -> dict:
“””
Evaluate model on the full validation split.
Model must implement: loss = model(img_features, caption_tokens)
where loss is scalar cross-entropy averaged over non-PAD tokens.

```
Returns:
    val_loss  — cross-entropy in nats/token (lower is better)
    val_bpb   — bits per byte (~val_loss / log2 / 3.5, for display)
"""
d = _load_data()
val_idx = d["val_idx"]
model.eval()
total_loss = 0.0
total_tok  = 0
with torch.no_grad():
    for i in range(0, len(val_idx), EVAL_BATCH_SIZE):
        batch_idx = val_idx[i : i + EVAL_BATCH_SIZE]
        img = torch.from_numpy(d["features"][batch_idx].astype(np.float32)).to(device)
        cap = torch.from_numpy(d["captions"][batch_idx].astype(np.int64)).to(device)
        non_pad = (cap != PAD_TOKEN_ID).sum().item()
        loss = model(img, cap)
        total_loss += loss.item() * non_pad
        total_tok  += non_pad
model.train()
avg = total_loss / max(total_tok, 1)
bpb = avg / math.log(2) / 3.5   # approx 3.5 bytes/token for cl100k_base
return {"val_loss": avg, "val_bpb": bpb, "eval_tokens": total_tok}
```

if **name** == “**main**”:
prepare()
