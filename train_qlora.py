"""
QLoRA fine-tuning script for MutualBase agents.
Adapted from autoresearch — this is the file the agent modifies.

Usage: python train_qlora.py
"""

import os
import gc
import time
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset

from prepare_qlora import CACHE_DIR, BASE_MODEL, TIME_BUDGET, MAX_SEQ_LEN, evaluate_loss

# ---------------------------------------------------------------------------
# Config — THE AGENT MODIFIES THESE
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # LoRA params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # Training params
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0

    # Schedule
    lr_scheduler: str = "cosine"  # cosine, linear, constant


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChatDataset(Dataset):
    """Simple dataset wrapping pre-tokenized examples."""

    def __init__(self, token_tensors, max_len=MAX_SEQ_LEN):
        self.examples = [t[:max_len] for t in token_tensors]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad batch to same length, create attention masks."""
    max_len = max(len(x) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, ids in enumerate(batch):
        input_ids[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = 1
        labels[i, :len(ids)] = ids  # causal LM: labels = input shifted

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    config = TrainConfig()
    print("=" * 60)
    print("MutualBase QLoRA Fine-Tune")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Config: {asdict(config)}")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    # Load data
    data_path = os.path.join(CACHE_DIR, "data.pt")
    if not os.path.exists(data_path):
        print("ERROR: Run prepare_qlora.py first")
        return
    data = torch.load(data_path)
    train_data = data["train"]
    val_data = data["val"]
    print(f"Data: {len(train_data)} train, {len(val_data)} val examples")

    # Load model in 4-bit
    print(f"\nLoading {BASE_MODEL} in 4-bit...")
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable ({100*trainable/total:.2f}%)")

    # DataLoader
    train_dataset = ChatDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # LR scheduler
    total_steps_estimate = (TIME_BUDGET // 10) * config.gradient_accumulation_steps  # rough estimate
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_estimate)
    elif config.lr_scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps_estimate)
    else:
        scheduler = None

    # Evaluate baseline
    print("\nEvaluating baseline...")
    baseline_loss = evaluate_loss(model, val_data, tokenizer)
    print(f"Baseline val_loss: {baseline_loss:.6f}")

    # Training loop
    print(f"\nTraining (time budget: {TIME_BUDGET}s)...")
    model.train()
    train_start = time.time()
    step = 0
    epoch = 0
    total_tokens = 0
    running_loss = 0.0
    best_loss = baseline_loss

    while True:
        epoch += 1
        for batch in train_loader:
            elapsed = time.time() - train_start
            if elapsed >= TIME_BUDGET:
                break

            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item()
            total_tokens += attention_mask.sum().item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            step += 1

            # Log every 10 steps
            if step % 10 == 0:
                avg_loss = running_loss / 10 * config.gradient_accumulation_steps
                lr = optimizer.param_groups[0]["lr"]
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                print(f"  step {step:4d} | loss {avg_loss:.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | {elapsed:.0f}s/{TIME_BUDGET}s")
                running_loss = 0.0

        elapsed = time.time() - train_start
        if elapsed >= TIME_BUDGET:
            break

    train_time = time.time() - train_start

    # Final evaluation
    print("\nEvaluating fine-tuned model...")
    model.eval()
    final_loss = evaluate_loss(model, val_data, tokenizer)
    improvement = baseline_loss - final_loss

    # Peak VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Save adapter
    adapter_dir = os.path.join(CACHE_DIR, "adapter_latest")
    model.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    # Print summary (autoresearch format)
    print("\n---")
    print(f"val_loss:          {final_loss:.6f}")
    print(f"baseline_loss:     {baseline_loss:.6f}")
    print(f"improvement:       {improvement:.6f}")
    print(f"training_seconds:  {train_time:.1f}")
    print(f"total_steps:       {step}")
    print(f"total_tokens:      {total_tokens}")
    print(f"epochs:            {epoch}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"trainable_params:  {trainable/1e6:.1f}M")
    print(f"lora_r:            {config.lora_r}")
    print(f"lora_alpha:        {config.lora_alpha}")
    print(f"learning_rate:     {config.learning_rate}")
    print(f"batch_size:        {config.batch_size}")


if __name__ == "__main__":
    train()
