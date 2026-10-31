"""Оценка классификатора обращений через OpenAI-совместимый API vLLM.

Режимы (все с одним и тем же промптом):
  base       — исходная модель, свободный ответ;
  base_json  — исходная модель, ответ ограничен JSON-схемой с перечнем категорий
               (показывает, сколько ошибок базы — просто неверный формат ответа);
  lora       — модель с LoRA-адаптером, который vLLM раздаёт под отдельным именем.

    python -m lab.server start Qwen/Qwen2.5-1.5B-Instruct -- \\
        --enable-lora --lora-modules appeals=outputs/appeals-lora --max-lora-rank 16
    python -m lab.appeals.evaluate
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from lab.appeals.core import LABELS, build_messages, category_schema, read_appeals_csv
from lab.common import (
    CACHE_DIR,
    DATA_DIR,
    RESULTS_DIR,
    env_info,
    json_schema_format,
    make_async_client,
    parse_json_object,
    print_table,
    save_json,
    write_jsonl,
)
from lab.metrics import classification_metrics, parse_label

MODES = ("base", "base_json", "lora")


async def classify(client, model: str, text: str, json_mode: bool) -> tuple[str | None, str]:
    kwargs = {"response_format": json_schema_format("appeal_category", category_schema())} if json_mode else {}
    resp = await client.chat.completions.create(
        model=model,
        messages=build_messages(text, json_mode=json_mode),
        temperature=0.0,
        max_tokens=32,
        **kwargs,
    )
    raw = resp.choices[0].message.content or ""
    if json_mode:
        data = parse_json_object(raw) or {}
        return parse_label(str(data.get("category", "")), LABELS), raw
    return parse_label(raw, LABELS), raw


async def classify_all(client, model: str, rows: list[dict], json_mode: bool, concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        async with semaphore:
            try:
                pred, raw = await classify(client, model, row["text"], json_mode)
            except Exception as exc:  # noqa: BLE001 — ошибка запроса = невалидный ответ
                pred, raw = None, f"ERROR {type(exc).__name__}: {exc}"[:200]
        return {"text": row["text"], "label": row["label"], "pred": pred, "raw": raw}

    return await asyncio.gather(*(one(r) for r in rows))


def top_confusions(results: list[dict], n: int = 10) -> list[dict]:
    pairs = Counter((r["label"], r["pred"] or "нераспознано") for r in results if r["pred"] != r["label"])
    return [{"true": t, "pred": p, "count": c} for (t, p), c in pairs.most_common(n)]


async def evaluate(args: argparse.Namespace) -> dict:
    rows = read_appeals_csv(args.test_csv)
    if args.limit:
        rows = rows[: args.limit]
    client = make_async_client(args.base_url, max_retries=2)
    report = {}
    for mode in args.modes:
        model = args.lora_name if mode == "lora" else args.base_model
        results = await classify_all(client, model, rows, json_mode=(mode == "base_json"), concurrency=args.concurrency)
        metrics = classification_metrics([r["label"] for r in results], [r["pred"] for r in results], LABELS)
        metrics["top_confusions"] = top_confusions(results)
        report[mode] = metrics
        write_jsonl([r for r in results if r["pred"] != r["label"]], CACHE_DIR / f"appeals_errors_{mode}.jsonl")
        print(f"{mode}: accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}", flush=True)
    await client.close()
    return {
        "test_csv": str(args.test_csv),
        "n": len(rows),
        "base_model": args.base_model,
        "modes": report,
        "env": env_info(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка классификатора обращений")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-name", default="appeals")
    parser.add_argument("--test-csv", default=str(DATA_DIR / "appeals" / "test.csv"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    result = asyncio.run(evaluate(parse_args(argv)))
    save_json(result, RESULTS_DIR / "appeals_eval.json")
    print_table(
        ["режим", "accuracy", "macro-F1", "нераспознано"],
        [
            [mode, f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}", f"{m['invalid_rate']:.1%}"]
            for mode, m in result["modes"].items()
        ],
    )
    return result


if __name__ == "__main__":
    main()
