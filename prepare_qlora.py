"""
Data preparation for MutualBase QLoRA fine-tuning.
Loads chat-format JSONL training data, tokenizes, and creates train/val splits.

Usage:
    python prepare_qlora.py                          # uses default data path
    python prepare_qlora.py --data path/to/data.jsonl

Data is cached in ~/.cache/autoresearch-qlora/.
"""

import json
import os
import random
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600          # training time budget: 10 min (QLoRA is slower than pretraining)
EVAL_SAMPLES = 50          # held-out validation examples
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # base model for fine-tuning (swap for Qwen3.5-4B when available)
MAX_SEQ_LEN = 4096         # max sequence length for training

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-qlora")
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "training_data.jsonl")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(data_path):
    """Load chat-format JSONL. Each line has {"messages": [...], "metadata": {...}}"""
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append(ex)
    return examples


def split_data(examples, val_size=EVAL_SAMPLES, seed=42):
    """Split into train/val, stratified by agent type."""
    random.seed(seed)
    random.shuffle(examples)

    # Group by agent
    by_agent = {}
    for ex in examples:
        agent = ex.get("metadata", {}).get("agent", "unknown")
        by_agent.setdefault(agent, []).append(ex)

    val = []
    train = []
    for agent, agent_examples in by_agent.items():
        # Take proportional val samples from each agent
        n_val = max(1, int(len(agent_examples) * val_size / len(examples)))
        val.extend(agent_examples[:n_val])
        train.extend(agent_examples[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def format_chat(messages, tokenizer):
    """Format messages into tokenizer's chat template."""
    # Use the model's built-in chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def tokenize_examples(examples, tokenizer, max_len=MAX_SEQ_LEN):
    """Tokenize all examples, return list of token ID tensors."""
    tokenized = []
    skipped = 0
    for ex in examples:
        messages = ex["messages"]
        text = format_chat(messages, tokenizer)
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)
        if len(ids) < 10:  # skip trivially short examples
            skipped += 1
            continue
        tokenized.append(torch.tensor(ids, dtype=torch.long))
    if skipped:
        print(f"  Skipped {skipped} examples (too short)")
    return tokenized


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def evaluate_loss(model, val_data, tokenizer, device="cuda"):
    """Average cross-entropy loss on validation set. Lower is better."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for ids in val_data:
            ids = ids.unsqueeze(0).to(device)
            if ids.shape[1] < 2:
                continue
            outputs = model(input_ids=ids[:, :-1], labels=ids[:, 1:])
            n_tokens = ids.shape[1] - 1
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(data_path=None):
    """Download model, tokenize data, save splits."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    data_path = data_path or DEFAULT_DATA
    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        print("Generate it first: python generate_mb_training_data.py --agent all --count 150")
        return

    print(f"Loading training data from {data_path}...")
    examples = load_training_data(data_path)
    print(f"  {len(examples)} examples loaded")

    print(f"\nSplitting data (val_size={EVAL_SAMPLES})...")
    train, val = split_data(examples)
    print(f"  Train: {len(train)}, Val: {len(val)}")

    # Stats by agent
    for split_name, split_data_list in [("Train", train), ("Val", val)]:
        by_agent = {}
        for ex in split_data_list:
            agent = ex.get("metadata", {}).get("agent", "unknown")
            by_agent[agent] = by_agent.get(agent, 0) + 1
        print(f"  {split_name}: {dict(by_agent)}")

    print(f"\nDownloading/loading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)

    print("Tokenizing train split...")
    train_tokens = tokenize_examples(train, tokenizer)
    print(f"  {len(train_tokens)} tokenized examples")

    print("Tokenizing val split...")
    val_tokens = tokenize_examples(val, tokenizer)
    print(f"  {len(val_tokens)} tokenized examples")

    # Token stats
    train_total = sum(len(t) for t in train_tokens)
    val_total = sum(len(t) for t in val_tokens)
    print(f"\nToken counts: train={train_total:,}, val={val_total:,}")
    print(f"Avg tokens/example: train={train_total/len(train_tokens):.0f}, val={val_total/len(val_tokens):.0f}")

    # Save processed data
    torch.save({"train": train_tokens, "val": val_tokens}, os.path.join(CACHE_DIR, "data.pt"))
    print(f"\nSaved tokenized data to {CACHE_DIR}/data.pt")
    print("Done! Ready to train with: python train_qlora.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to training JSONL")
    args = parser.parse_args()
    prepare(args.data)
