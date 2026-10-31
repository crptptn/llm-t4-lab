"""Синтетический датасет обращений граждан через OpenAI-совместимый API (vLLM).

Реальные обращения содержат персональные данные, поэтому датасет генерируется.
Чтобы тест не повторял трейн по форме, у тестовой выборки свои стили текста,
которых нет в обучении. Дубликаты (после нормализации) убираются, в том числе между выборками.

    python -m lab.server start Qwen/Qwen2.5-7B-Instruct-AWQ
    python -m lab.appeals.generate --model Qwen/Qwen2.5-7B-Instruct-AWQ
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
from collections import Counter

from lab.appeals.core import LABEL_HINTS, LABELS, write_appeals_csv
from lab.common import (
    DATA_DIR,
    RESULTS_DIR,
    env_info,
    json_schema_format,
    make_async_client,
    parse_json_object,
    save_json,
)
from lab.metrics import normalize_text

OUT_DIR = DATA_DIR / "appeals"

TRAIN_STYLES = [
    "официальное письмо: обращение «Уважаемый…», полные предложения, канцелярские обороты",
    "короткое сообщение с телефона: без приветствия, разговорный стиль, одна-две опечатки",
    "эмоциональная жалоба человека, который обращается уже не первый раз",
    "вопрос-уточнение: человек не жалуется, а спрашивает, куда обратиться или как получить услугу",
]
TEST_STYLES = [
    "пост или комментарий в городском паблике, с хештегом и обращением к администрации",
    "сбивчивое обращение пожилого человека с лишними подробностями о своей жизни",
]
PERSONAS = [
    "пенсионерка",
    "студент",
    "мама двоих детей",
    "водитель такси",
    "индивидуальный предприниматель",
    "учитель",
    "житель частного сектора",
    "председатель совета дома",
    "человек с инвалидностью",
    "молодой отец",
]
SCHEMA = {
    "type": "object",
    "properties": {"appeals": {"type": "array", "items": {"type": "string"}}},
    "required": ["appeals"],
    "additionalProperties": False,
}


def build_prompt(label: str, style: str, persona: str, k: int, rng: random.Random) -> str:
    mix = ""
    if rng.random() < 0.3:
        other = rng.choice([x for x in LABELS if x != label])
        mix = (
            f"- в двух-трёх обращениях мимоходом упомяни тему «{other}», "
            f"но главная проблема должна оставаться в теме «{label}»;\n"
        )
    return (
        f"Сгенерируй {k} разных обращений граждан в региональные органы власти.\n"
        f"Тема: «{label}» — {LABEL_HINTS[label]}.\n"
        f"Стиль: {style}.\n"
        f"Автор: {persona}.\n"
        "Требования:\n"
        "- каждое обращение — от одного до пяти предложений; проблемы, места и детали не повторяются;\n"
        "- не называй тему напрямую и не пиши слово «категория»;\n"
        f"{mix}"
        "- адреса и названия выдумывай, без реальных ФИО и телефонов.\n"
        'Верни JSON вида {"appeals": ["текст", ...]}.'
    )


def plan_jobs(split: str, per_label: int, styles: list[str], per_request: int, rng: random.Random) -> list[dict]:
    # берём с запасом 20%: часть текстов отсеется как дубли или слишком короткие
    batches = math.ceil(per_label * 1.2 / per_request)
    jobs = []
    for label in LABELS:
        for b in range(batches):
            style = styles[b % len(styles)]
            persona = rng.choice(PERSONAS)
            jobs.append(
                {
                    "split": split,
                    "label": label,
                    "style": style,
                    "prompt": build_prompt(label, style, persona, per_request, rng),
                }
            )
    return jobs


def dedupe(rows: list[dict], seen: set[str]) -> list[dict]:
    unique = []
    for row in rows:
        key = normalize_text(row["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def take_per_label(rows: list[dict], per_label: int, rng: random.Random) -> list[dict]:
    rng.shuffle(rows)
    counts: Counter[str] = Counter()
    picked = []
    for row in rows:
        if counts[row["label"]] < per_label:
            counts[row["label"]] += 1
            picked.append(row)
    return picked


async def run_job(client, model: str, job: dict, semaphore: asyncio.Semaphore, temperature: float) -> list[dict]:
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": job["prompt"]}],
                temperature=temperature,
                top_p=0.95,
                max_tokens=2500,
                response_format=json_schema_format("appeals", SCHEMA),
            )
        except Exception as exc:  # noqa: BLE001 — один неудачный батч не должен ронять генерацию
            print(f"ошибка запроса: {type(exc).__name__}: {exc}"[:200], flush=True)
            return []
    data = parse_json_object(resp.choices[0].message.content)
    texts = data.get("appeals") if data else None
    if not isinstance(texts, list):
        return []
    return [
        {"text": t.strip(), "label": job["label"], "style": job["style"], "split": job["split"]}
        for t in texts
        if isinstance(t, str) and len(t.strip()) >= 30
    ]


async def generate(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    jobs = plan_jobs("train", args.train_per_label, TRAIN_STYLES, args.per_request, rng)
    jobs += plan_jobs("test", args.test_per_label, TEST_STYLES, args.per_request, rng)
    client = make_async_client(args.base_url, max_retries=2)
    semaphore = asyncio.Semaphore(args.concurrency)
    batches = await asyncio.gather(*(run_job(client, args.model, j, semaphore, args.temperature) for j in jobs))
    await client.close()

    rows = [row for batch in batches for row in batch]
    seen: set[str] = set()
    train = take_per_label(dedupe([r for r in rows if r["split"] == "train"], seen), args.train_per_label, rng)
    test = take_per_label(dedupe([r for r in rows if r["split"] == "test"], seen), args.test_per_label, rng)
    write_appeals_csv(train, OUT_DIR / "train.csv")
    write_appeals_csv(test, OUT_DIR / "test.csv")

    stats = {
        "generator_model": args.model,
        "requests": len(jobs),
        "failed_requests": sum(1 for b in batches if not b),
        "generated_texts": len(rows),
        "train": dict(Counter(r["label"] for r in train)),
        "test": dict(Counter(r["label"] for r in test)),
        "train_styles": TRAIN_STYLES,
        "test_styles": TEST_STYLES,
        "env": env_info(),
    }
    save_json(stats, RESULTS_DIR / "appeals_data.json")
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация синтетических обращений")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-per-label", type=int, default=200)
    parser.add_argument("--test-per-label", type=int, default=60)
    parser.add_argument("--per-request", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    stats = asyncio.run(generate(parse_args(argv)))
    print(
        f"train: {sum(stats['train'].values())}, test: {sum(stats['test'].values())}, "
        f"неудачных запросов: {stats['failed_requests']} из {stats['requests']}"
    )


if __name__ == "__main__":
    main()
