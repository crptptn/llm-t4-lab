"""FP16 против AWQ и GPTQ (4 бита): размер весов, скорость генерации и точность на GSM8K.

Каждый вариант запускается в отдельном процессе: vLLM не всегда полностью отдаёт память GPU,
и следующая модель могла бы не поместиться.

    python -m lab.quant --n 300
    python -m lab.quant --variants fp16=Qwen/Qwen2.5-3B-Instruct awq=Qwen/Qwen2.5-3B-Instruct-AWQ
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from lab.common import CACHE_DIR, RESULTS_DIR, env_info, load_json, print_table, save_json, write_jsonl
from lab.metrics import extract_final_number, gsm8k_gold, numbers_match

DEFAULT_VARIANTS = [
    "fp16=Qwen/Qwen2.5-3B-Instruct",
    "awq=Qwen/Qwen2.5-3B-Instruct-AWQ",
    "gptq=Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
]
SYSTEM_PROMPT = (
    "Solve the math word problem step by step. Write the final answer on the last line in the form: #### <number>"
)


def weights_size_gib(model_id: str) -> tuple[float, str | None]:
    """Размер safetensors-весов на диске и метод квантизации из config.json."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, ignore_patterns=["*.bin", "*.pth", "*.pt", "original/*"]))
    size = sum(f.stat().st_size for f in path.rglob("*.safetensors")) / 2**30
    method = None
    config = path / "config.json"
    if config.exists():
        method = (json.loads(config.read_text()).get("quantization_config") or {}).get("quant_method")
    return size, method


def run_variant(name: str, model: str, n: int, tp: int, max_model_len: int, gpu_util: float) -> dict:
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n, len(ds))))
    size_gib, method = weights_size_gib(model)

    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        dtype="half",
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        seed=0,
    )
    load_s = time.perf_counter() - t0

    conversations = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}] for q in ds["question"]
    ]

    # 1) пакетная генерация — точность и суммарная пропускная способность
    t0 = time.perf_counter()
    outputs = llm.chat(conversations, SamplingParams(temperature=0.0, max_tokens=512))
    batch_s = time.perf_counter() - t0
    texts = [o.outputs[0].text for o in outputs]
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    samples, correct = [], 0
    for question, answer, text in zip(ds["question"], ds["answer"], texts):
        gold, pred = gsm8k_gold(answer), extract_final_number(text)
        ok = numbers_match(pred, gold)
        correct += ok
        samples.append({"question": question, "gold": gold, "pred": pred, "correct": ok, "output": text})

    # 2) одиночные запросы — скорость для одного пользователя (prefill + 256 токенов)
    single = []
    single_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
    for conversation in conversations[:5]:
        t = time.perf_counter()
        out = llm.chat([conversation], single_params, use_tqdm=False)[0]
        single.append(len(out.outputs[0].token_ids) / (time.perf_counter() - t))

    write_jsonl(samples, CACHE_DIR / f"quant_{name}_samples.jsonl")
    return {
        "name": name,
        "model": model,
        "quant_method": method or "fp16",
        "weights_gib": size_gib,
        "n": len(samples),
        "gsm8k_accuracy": correct / len(samples),
        "load_s": load_s,
        "batch_s": batch_s,
        "batch_throughput_tok_s": gen_tokens / batch_s,
        "mean_output_tokens": gen_tokens / len(samples),
        "single_stream_tok_s": sum(single) / len(single),
        "errors_examples": [s for s in samples if not s["correct"]][:5],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сравнение квантизаций на GSM8K")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, help="имя=модель")
    parser.add_argument("--n", type=int, default=300, help="сколько задач GSM8K (из 1319)")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel (2 — обе T4)")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    variants = [v.split("=", 1) for v in args.variants]

    if args.worker:
        ((name, model),) = variants
        result = run_variant(name, model, args.n, args.tp, args.max_model_len, args.gpu_util)
        save_json(result, CACHE_DIR / f"quant_{name}.json")
        return

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}
    results = []
    for name, model in variants:
        print(f"=== {name}: {model}", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "lab.quant",
            "--worker",
            "--variants",
            f"{name}={model}",
            "--n",
            str(args.n),
            "--tp",
            str(args.tp),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-util",
            str(args.gpu_util),
        ]
        proc = subprocess.run(cmd, env=env, check=False)
        out = CACHE_DIR / f"quant_{name}.json"
        if proc.returncode != 0 or not out.exists():
            # вариант не завёлся (например, ядро квантизации не поддерживает GPU) — фиксируем и идём дальше
            results.append({"name": name, "model": model, "error": f"код возврата {proc.returncode}"})
            continue
        results.append(load_json(out))
        out.unlink()

    path = save_json({"variants": results, "env": env_info()}, RESULTS_DIR / "quant.json")
    print_table(
        ["вариант", "веса, ГиБ", "GSM8K", "пакет, ток/с", "1 запрос, ток/с"],
        [
            [r["name"], "—", "—", "—", r["error"]]
            if "error" in r
            else [
                r["name"],
                f"{r['weights_gib']:.2f}",
                f"{r['gsm8k_accuracy']:.3f}",
                f"{r['batch_throughput_tok_s']:.0f}",
                f"{r['single_stream_tok_s']:.1f}",
            ]
            for r in results
        ],
    )
    print(f"сохранено: {path}")


if __name__ == "__main__":
    main()
