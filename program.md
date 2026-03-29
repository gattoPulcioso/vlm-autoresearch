# vlm-autoresearch

Autonomous research loop for Vision-Language Models on a single GPU.

-----

## Setup

To start a new experiment session:

1. **Agree on a run tag** with the user (e.g. `apr1`). The branch
   `autoresearch/<tag>` must not already exist.
1. **Create the branch**: `git checkout -b autoresearch/<tag>`
1. **Read the in-scope files** — all three are short, read them fully:
- `prepare.py` — fixed constants, data pipeline, evaluation. **Do not modify.**
- `train.py` — the only file you edit. Model, optimizer, training loop.
- `program.md` — this file.
1. **Verify data exists**: check that `~/.cache/vlm_autoresearch/` contains
   `features.npy`, `captions.npy`, `split.npz`. If not, tell the user to run
   `python prepare.py` (one-time, ~20 min including COCO download).
1. **Initialise results.tsv** with just the header row (see format below).
1. **Confirm** setup looks good, then begin.

-----

## What you are optimising

The task is **image-conditioned caption generation** on COCO val2017.

Each sample is a (image, caption) pair. The model receives pre-computed
CLIP ViT-B/32 features for the image and must predict the caption tokens
left-to-right.

**Metric: `val_loss` (cross-entropy, nats/token). Lower is better.**

Since training always runs for exactly **5 minutes** of wall-clock time,
experiments are directly comparable regardless of architecture.

-----

## Experimentation rules

**You CAN:**

- Modify `train.py` — anything: architecture, optimizer, LR schedule,
  batch size, number of visual tokens used, projector type, etc.

**You CANNOT:**

- Modify `prepare.py`. It is read-only. The eval harness lives there.
- Install new packages. Use only what is already importable in the environment
  (torch, transformers, numpy, tiktoken, math, time, sys).
- Change the evaluation metric or game the validation set.

**Simplicity criterion**: A tiny `val_loss` improvement that adds
50 lines of gnarly code is probably not worth it. Removing code and
maintaining performance is always a win.

-----

## Architectural search space

The baseline uses one CLS token from CLIP → linear projector → small GPT
decoder. Many things are worth trying:

**Projector:**

- 2-layer MLP with GeLU (classic LLaVA)
- Cross-attention: text queries, image keys/values
- Gated projector

**Visual tokens (`N_VIS_USED` in train.py):**

- 1  → only CLS (baseline)
- 7  → pooled 7×1 strip
- 49 → all spatial patches (the full 7×7 grid)
- Learned spatial pooling (e.g. 4 tokens via cross-attention)

**LM architecture:**

- Wider / deeper (watch VRAM — RTX 3070 has 8 GB)
- Mixture of sliding + full attention windows
- Rotary embeddings (RoPE) instead of learned positional embeddings
- SwiGLU activation instead of GeLU

**Training:**

- Higher LR with aggressive warmup
- AdamW → Lion → Muon optimizer
- Gradient accumulation for larger effective batch size
- Freeze projector for first N steps, then unfreeze
- Prefix tuning (freeze LM entirely, only train projector)

**Regularisation:**

- Dropout on attention / MLP
- Label smoothing in cross-entropy

-----

## Experiment loop

LOOP FOREVER (do not stop to ask the user):

1. Look at current git state (branch, last commit).
1. Edit `train.py` with your idea.
1. `git add train.py && git commit -m "<short description>"`
1. Run: `python train.py > run.log 2>&1`
1. Extract result: `grep "^val_loss:\|^peak_vram_mb:" run.log`
1. If grep is empty → run crashed. Check: `tail -n 40 run.log`
- Trivial fix (typo, missing import)? Fix and re-run.
- Fundamental problem (OOM, wrong shape)? Log as crash, discard, move on.
1. Log to `results.tsv`.
1. If `val_loss` improved → keep the commit (advance the branch).
1. If `val_loss` equal or worse → `git reset --hard HEAD~1` (discard).

**Timeout**: if a run exceeds 10 minutes, kill it (`Ctrl-C`) and treat
it as a crash.

**NEVER STOP**: once the loop starts, run indefinitely until the user
interrupts you. The user may be asleep expecting ~30–50 experiments overnight.
Do not ask “should I continue?”. Just continue.

-----

## Output format

After training, `train.py` prints a summary like:

```
---
val_loss:         2.345678
val_bpb:          0.963400
training_seconds: 300.1
total_seconds:    318.4
peak_vram_mb:     5840.2
total_tokens_M:   3.84
num_steps:        7200
num_params_M:     11.2
depth:            4
lm_dim:           256
n_vis_used:       1
```

Extract the key metrics:

```bash
grep "^val_loss:\|^peak_vram_mb:\|^num_params_M:" run.log
```

-----

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).
Do **not** git-commit this file — leave it untracked.

Header + format:

```
commit	val_loss	memory_gb	status	description
```

Columns:

1. `commit` — 7-char git hash
1. `val_loss` — numeric (e.g. `2.345678`); use `0.000000` for crashes
1. `memory_gb` — peak VRAM in GB, 1 decimal (e.g. `5.7`); use `0.0` for crashes
1. `status` — one of: `keep`, `discard`, `crash`
1. `description` — short plain-text description (no tabs!)

Example:

```
commit	val_loss	memory_gb	status	description
a1b2c3d	2.345678	5.7	keep	baseline
b2c3d4e	2.301200	5.8	keep	2-layer MLP projector
c3d4e5f	2.389000	5.7	discard	SwiGLU activation (worse)
d4e5f6g	0.000000	0.0	crash	49 visual tokens OOM
e5f6g7h	2.288500	6.1	keep	49 visual tokens with grad accumulation
```

-----

## VRAM budget (RTX 3070, 8 GB)

- Stay under ~7.5 GB to leave headroom.
- If an experiment hits OOM, log as crash and try a leaner variant next.
- Useful levers: reduce `LM_DIM`, reduce `DEPTH`, reduce `BATCH_SIZE`,
  use gradient accumulation instead of large `BATCH_SIZE`.
