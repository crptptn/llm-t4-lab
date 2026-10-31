"""QLoRA: дообучение Qwen2.5-Instruct классифицировать обращения граждан.

База грузится в 4 битах (NF4, двойная квантизация), обучаются только LoRA-адаптеры
во всех линейных слоях внимания и MLP. Loss — только по токенам ответа.
На T4 нет bf16, поэтому вычисления в fp16, оптимизатор paged AdamW 8-bit.

    python -m lab.appeals.train
    python -m lab.appeals.train --save-merged   # дополнительно склеить адаптер с базой
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

from lab.appeals.core import build_training_example, pad_batch, read_appeals_csv
from lab.common import CACHE_DIR, DATA_DIR, OUTPUTS_DIR, RESULTS_DIR, env_info, save_json

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_collator(pad_id: int):
    import torch

    def collate(batch):
        return {k: torch.tensor(v, dtype=torch.long) for k, v in pad_batch(batch, pad_id).items()}

    return collate


def merge_adapter(base_model: str, adapter_dir: str, out_dir: str) -> None:
    """Склейка адаптера с fp16-базой — запасной вариант, если LoRA в vLLM не заведётся."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16, device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA-классификатор обращений")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-csv", default=str(DATA_DIR / "appeals" / "train.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "appeals-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-text-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-merged", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Trainer на двух GPU оборачивает модель в DataParallel, а 4-битной модели это не нужно
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    rows = read_appeals_csv(args.train_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [build_training_example(tokenizer, r["text"], r["label"], args.max_text_tokens) for r in rows]
    random.Random(args.seed).shuffle(examples)
    print(f"примеров: {len(examples)}, макс. длина: {max(len(e['input_ids']) for e in examples)} токенов")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, dtype=torch.float16, device_map={"": 0}
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"обучаемых параметров: {trainable:,} из {total:,} ({100 * trainable / total:.2f}%)")

    steps = math.ceil(len(examples) / (args.batch_size * args.grad_accum) * args.epochs)
    training_args = TrainingArguments(
        output_dir=str(CACHE_DIR / "appeals-trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, steps // 20),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    losses = [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h]
    save_json(
        {
            "base_model": args.model,
            "examples": len(examples),
            "epochs": args.epochs,
            "steps": trainer.state.global_step,
            "batch_size": args.batch_size * args.grad_accum,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "trainable_params": trainable,
            "total_params": total,
            "train_minutes": train_s / 60,
            "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            "loss_curve": losses,
            "env": env_info(),
        },
        RESULTS_DIR / "appeals_train.json",
    )
    print(f"адаптер сохранён в {args.output_dir}, обучение заняло {train_s / 60:.1f} мин")

    if args.save_merged:
        del trainer, model
        torch.cuda.empty_cache()
        merged_dir = args.output_dir.rstrip("/") + "-merged"
        merge_adapter(args.model, args.output_dir, merged_dir)
        print(f"склеенная модель: {merged_dir}")


if __name__ == "__main__":
    main()
