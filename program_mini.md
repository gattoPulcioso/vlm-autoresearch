# vlm-autoresearch-mini

Autonomous VLM research loop — lightweight version for testing the agent loop.
Synthetic data, no downloads required, full cycle in ~2.5 minutes.

-----

## Setup

1. **Agree on a run tag** with the user (e.g. `test1`). The branch
   `autoresearch/<tag>` must not already exist.
1. **Create the branch**: `git checkout -b autoresearch/<tag>`
1. **Read the in-scope files** — read all three in full:
- `prepare_mini.py` — synthetic data, pipeline, evaluation harness. **Do not modify.**
- `train_mini.py` — the only file you edit. Model, optimizer, training loop.
- `program_mini.md` — this file.
1. **Verify data exists**: check that `~/.cache/vlm_autoresearch_mini/` contains
   `features.npy`, `captions.npy`, `split.npz`. If not, tell the user to run
   `python prepare_mini.py` (takes ~30 seconds, no downloads).
1. **Initialize results.tsv** with the header row only.
1. **Confirm** setup looks good, then begin.

-----

## What you are optimizing

Task: image-conditioned caption generation on synthetic data.

Each sample is a pair of (image features, tokenized caption). The model receives
pre-computed synthetic CLIP-like features and must predict caption tokens
left-to-right.

**Metric: `val_loss` (cross-entropy, nats/token). Lower is better.**

Training always runs for exactly **2 minutes** of wall-clock time — experiments
are directly comparable regardless of architecture.

Note: data is synthetic, so absolute `val_loss` values carry no semantic meaning.
What matters is **direction** — which architecture learns more effectively within
this budget.

-----

## Experimentation rules

**You CAN:**

- Modify `train_mini.py` — anything: architecture, optimizer, LR schedule,
  batch size, number of visual tokens, projector type, etc.

**You CANNOT:**

- Modify `prepare_mini.py`. It is read-only. The evaluation harness lives there.
- Install new packages. Only use `torch`, `numpy`, `math`, `time`, `sys`.
- Change the metric or manipulate the validation set in any way.

**File editing**: always rewrite `train_mini.py` from scratch using the Write
File tool. Do NOT use str_replace or patch tools. Read the current file, plan
your changes, then write the complete new file in a single operation.

**Simplicity criterion**: a tiny improvement that adds complex code is not worth
it. Removing code while maintaining performance is always a win.

-----

## Search space

The baseline uses 1 CLS token → linear projector → 2-layer GPT with dim=64.
Things worth trying:

**Projector:**

- 2-layer MLP with GeLU (linear → nonlinear)
- Projector with explicit bias

**Visual tokens (`N_VIS_USED`):**

- 1 → CLS only (baseline)
- 4, 9, 16 → subset of patch tokens
- 50 → all tokens (watch VRAM)

**LM architecture:**

- Wider: `LM_DIM` 128, 256
- Deeper: `DEPTH` 3, 4
- RoPE instead of learned positional embeddings
- SwiGLU instead of GeLU in MLP

**Training:**

- Higher LR (1e-3) or lower (1e-4)
- Larger batch size (128, 256) with gradient accumulation
- More aggressive weight decay
- Longer or shorter warmup

**Key insight**: with synthetic data and a small model, underfitting is the main
risk, not overfitting. Slightly larger models tend to do better here.

-----

## Experiment loop

LOOP FOREVER — do not stop to ask the user whether to continue:

1. Check current git state (branch, last commit).
1. Edit `train_mini.py` with your idea — **always rewrite the whole file**.
1. `git add train_mini.py && git commit -m "<short description>"`
1. Run: `python train_mini.py > run.log 2>&1`
1. Extract results: `grep "^val_loss:\|^peak_vram_mb:" run.log`
1. If grep is empty → crash. Check: `tail -n 30 run.log`
- Trivial fix (typo, missing import)? Fix and re-run.
- Fundamental problem (OOM, wrong shape)? Log as crash, discard, move on.
1. Log to `results.tsv`.
1. If `val_loss` improved → keep the commit (advance the branch).
1. If equal or worse → `git reset --hard HEAD~1`.

**Timeout**: each run takes ~2 minutes + a few seconds overhead. If a run
exceeds 5 minutes, kill it and treat it as a crash.

**NEVER STOP**: once the loop starts, run indefinitely until the user manually
interrupts you. Do not ask “should I continue?”. Just continue.

-----

## Output format

`train_mini.py` prints a summary at the end:

```
---
val_loss:         3.210000
val_bpb:          1.320000
training_seconds: 120.1
total_seconds:    121.8
peak_vram_mb:     312.4
total_tokens_M:   0.98
num_steps:        4800
num_params_M:     0.08
depth:            2
lm_dim:           64
n_vis_used:       1
```

Extract key metrics:

```bash
grep "^val_loss:\|^peak_vram_mb:\|^num_params_M:" run.log
```

-----

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).
Do NOT git-commit this file — leave it untracked.

Header:

```
commit	val_loss	memory_gb	status	description
```

Example:

```
commit	val_loss	memory_gb	status	description
a1b2c3d	3.210000	0.3	keep	baseline
b2c3d4e	3.058000	0.3	keep	2-layer MLP projector
c3d4e5f	3.301000	0.3	discard	SwiGLU activation worse
d4e5f6g	2.991000	0.5	keep	LM_DIM 128 depth 3
e5f6g7h	0.000000	0.0	crash	N_VIS_USED 50 OOM
```

-----

## VRAM budget

With the mini model, VRAM is rarely an issue (baseline ~300MB).
If you explore very large `LM_DIM` (512+) or `N_VIS_USED=50` with a large
batch size, you may hit OOM on low-VRAM GPUs.
Primary levers: reduce `BATCH_SIZE` or `LM_DIM`.
