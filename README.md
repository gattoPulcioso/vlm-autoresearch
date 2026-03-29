# vlm-autoresearch

Autonomous research loop for Vision-Language Models on a single consumer GPU.

Inspired by [@karpathy/autoresearch](https://github.com/karpathy/autoresearch) — which lets an AI agent run overnight experiments on a small LLM training setup — this repo applies the same idea to **Vision-Language Models**: an agent modifies the architecture, trains for a fixed time budget, checks if the metric improved, keeps or discards, and repeats.

-----

## How it works

The core idea is identical to the original autoresearch: give an AI agent a real but minimal training setup and let it experiment autonomously. You go to sleep. You wake up to a log of experiments and (hopefully) a better model.

The key adaptation for VLMs is **pre-computed visual features**: the vision encoder (CLIP ViT-B/32) runs once in `prepare.py` and saves all image embeddings to disk. The agent then works exclusively on the *projector* and the *language model* — keeping each training run within the 5-minute budget even on consumer hardware.

### Repository structure

```
prepare.py       — one-time data prep: downloads COCO, runs CLIP, saves features to disk. Never modified.
train.py         — the only file the agent edits: projector, LM architecture, optimizer, training loop.
program.md       — instructions for the agent.

prepare_mini.py  — lightweight version: generates synthetic data in ~30s, no downloads.
train_mini.py    — lightweight training script (2-min budget, tiny model). Use to test the agent loop.
program_mini.md  — agent instructions for the mini version.
```

Start with the `_mini` files to verify your agent loop works end-to-end, then switch to the full files for real experiments.

-----

## Quick start

**Requirements:** NVIDIA GPU (tested on RTX 3070, 8GB VRAM), Python 3.10+, PyTorch 2.x + CUDA.

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers tiktoken Pillow numpy

# 2a. Mini setup — synthetic data, ~30 seconds, no downloads
python prepare_mini.py
python train_mini.py   # verify a single run works

# 2b. Full setup — downloads COCO val2017 (~1GB) + precomputes CLIP features (~20 min)
python prepare.py
python train.py        # verify a single run works

# 3. Initialize git (required for the agent loop)
git init && git add . && git commit -m "initial"
```

-----

## Running the agent

Point any AI coding agent with shell access at `program_mini.md` (or `program.md` for the full version) and use a prompt like:

```
Have a look at program_mini.md and let's kick off a new experiment!
```

The agent will:

1. Create a branch (`autoresearch/<tag>`)
1. Modify `train_mini.py` with an architectural idea
1. Run training, extract `val_loss` from the log
1. Keep the commit if the metric improved, `git reset` otherwise
1. Repeat indefinitely until you interrupt it

After a night of experiments you get a `results.tsv` with a full log of what was tried.

### Recommended agents

- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — best results, handles the tool loop reliably
- Any coding agent with shell access and file write permissions

#### Running with a local model (Ollama)

```bash
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_AUTH_TOKEN="ollama"
export ANTHROPIC_MODEL="qwen2.5-coder:7b"
claude
```

If the local model struggles with `str_replace`, the `program_mini.md` already instructs the agent to **always rewrite the file from scratch** — this is the most reliable strategy for smaller models.

-----

## Design

### What the agent can modify

Everything in `train.py` / `train_mini.py` is fair game:

- **Projector architecture** — linear, 2-layer MLP, cross-attention, gated
- **Number of visual tokens** — from 1 (CLS only) up to 50 (full 7×7 grid)
- **LM architecture** — depth, width, positional embeddings (learned vs RoPE), activation (GeLU vs SwiGLU)
- **Optimizer** — AdamW, Lion, Muon, learning rate, schedule, weight decay
- **Batch size and gradient accumulation**

### What is fixed

`prepare.py` / `prepare_mini.py` are never modified. They contain the evaluation harness (`evaluate_val_loss`), the data pipeline, and all constants — including `TIME_BUDGET`. This ensures all experiments are directly comparable.

### Metric

**`val_loss`** — cross-entropy in nats/token on the held-out validation split. Lower is better. Since the time budget is fixed, there is no need to account for training speed: every experiment gets the same wall-clock compute.

### VRAM budget (RTX 3070 / 8GB)

|Version|Baseline VRAM|Time budget|Experiments / night|
|-------|-------------|-----------|-------------------|
|mini   |~300 MB      |2 min      |~150               |
|full   |~2–4 GB      |5 min      |~50                |

-----

## Dataset

**Full version:** [COCO val2017](https://cocodataset.org) — 5000 image-caption pairs. CLIP ViT-B/32 features are pre-computed once and saved as `features.npy` (~230MB). The agent never loads raw images.

**Mini version:** 500 synthetic samples generated procedurally — CLIP-like feature vectors (structured Gaussian noise with spatial correlation) paired with template captions like `"a small red circle filled centered on white background"`. No downloads, no external dependencies beyond PyTorch and NumPy.

-----

## Results format

The agent logs every experiment to `results.tsv` (untracked by git):

```
commit    val_loss    memory_gb    status    description
a1b2c3d   3.210000    0.3          keep      baseline
b2c3d4e   3.058000    0.3          keep      2-layer MLP projector
c3d4e5f   3.301000    0.3          discard   SwiGLU activation worse
d4e5f6g   2.991000    0.5          keep      LM_DIM 128 depth 3
e5f6g7h   0.000000    0.0          crash     N_VIS_USED 50 OOM
```

-----

## Acknowledgements

This project is a direct adaptation of **[@karpathy/autoresearch](https://github.com/karpathy/autoresearch)**.
The original repo introduced the idea of giving an AI agent a minimal but real LLM training setup and letting it run autonomous experiments overnight. All credit for the core concept and loop design goes there. This repo extends it to the vision-language domain and adapts it for consumer GPUs with limited VRAM.

-----

## License

MIT
