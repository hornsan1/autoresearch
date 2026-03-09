# MutualBase QLoRA autoresearch

Fine-tune Qwen3.5-4B (or Qwen2.5-3B) to replace the 35B model for MutualBase specialist agents.

## Setup

1. **Agree on a run tag**: e.g. `qlora-mar9`
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Read the files**:
   - `prepare_qlora.py` — data loading, tokenization, eval metric. **Do not modify.**
   - `train_qlora.py` — QLoRA config, training loop. **This is what you modify.**
   - `training_data.jsonl` — the training data (407 examples for 3 MutualBase agents)
4. **Prepare data**: `python prepare_qlora.py --data training_data.jsonl`
5. **Initialize results.tsv** with header row
6. **Confirm and go**

## What you CAN modify

Only `train_qlora.py`. Everything in `TrainConfig` is fair game:
- LoRA rank (`lora_r`): 8, 16, 32, 64
- LoRA alpha: usually 2x rank
- Target modules: which layers get LoRA adapters
- Learning rate: try 1e-5 to 5e-4
- Batch size / gradient accumulation
- LR scheduler: cosine, linear, constant
- Warmup steps
- Dropout

You can also modify the training loop itself — different loss functions, curriculum strategies, etc.

## What you CANNOT modify

- `prepare_qlora.py` — fixed data pipeline and eval metric
- `training_data.jsonl` — fixed training data
- The base model (set in prepare_qlora.py)

## The metric

`val_loss` — average cross-entropy on held-out validation examples. **Lower is better.**

Extract from log:
```
grep "^val_loss:" run.log
```

## Logging results

Same format as original autoresearch:

```
commit	val_loss	memory_gb	status	description
a1b2c3d	2.345678	4.2	keep	baseline
b2c3d4e	2.123456	4.3	keep	increase lora_r to 32
```

## The loop

LOOP FOREVER:
1. Check git state
2. Modify `train_qlora.py` with an experimental idea
3. git commit
4. Run: `python train_qlora.py > run.log 2>&1`
5. Read results: `grep "^val_loss:\|^peak_vram_mb:" run.log`
6. If empty → crashed. `tail -n 50 run.log` for stack trace
7. Log to results.tsv
8. If val_loss improved → keep
9. If worse → git reset
10. GOTO 1

**Time budget**: Each run is ~10 minutes. Expect ~6 experiments/hour, ~50 overnight.

**Ideas to try** (in rough priority):
1. Baseline first (always)
2. LoRA rank sweep: 8 → 16 → 32 → 64
3. Learning rate sweep: 5e-5, 1e-4, 2e-4, 5e-4
4. Target modules: try just attention (q,k,v,o) vs attention+MLP
5. Alpha/rank ratio: 1x, 2x, 4x
6. Scheduler: cosine vs linear vs constant
7. Gradient accumulation: 1, 2, 4, 8
8. Dropout: 0, 0.05, 0.1

**NEVER STOP** — run until manually interrupted.
