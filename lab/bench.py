"""Нагрузочный бенчмарк OpenAI-совместимого сервера vLLM.

На каждом уровне конкурентности отправляет пачку стриминговых запросов и считает:
  TTFT  — время от отправки запроса до первого токена (в основном prefill и ожидание в очереди);
  TPOT  — среднее время на каждый следующий токен (скорость декодирования);
  throughput — выходные токены и запросы в секунду по всей пачке.

Сценарии:
  unique         — у каждого запроса свой текст, общих префиксов почти нет;
  shared-prefix  — у всех запросов общий системный промпт на ~2 тыс. токенов. Запусти дважды:
                   с --enable-prefix-caching и с --no-enable-prefix-caching, чтобы увидеть эффект кэша.

    python -m lab.bench --model Qwen/Qwen2.5-1.5B-Instruct --scenario unique --tag unique
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass

from lab.common import RESULTS_DIR, env_info, http_get_json, make_async_client, print_table, save_json
from lab.metrics import summarize
from lab.prompts import QUESTIONS, build_shared_prefix

SCENARIOS = ("unique", "shared-prefix")


@dataclass
class RequestResult:
    ok: bool
    ttft: float | None = None
    e2e: float | None = None
    output_tokens: int = 0
    prompt_tokens: int | None = None
    error: str | None = None

    @property
    def tpot(self) -> float | None:
        if self.ttft is None or self.e2e is None or self.output_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)


def build_messages(scenario: str, index: int, prefix: str) -> list[dict]:
    question = QUESTIONS[index % len(QUESTIONS)]
    if scenario == "unique":
        # номер запроса в самом начале: общим остаётся только служебный кусок шаблона чата
        return [{"role": "user", "content": f"Запрос №{index}. {question}"}]
    if scenario == "shared-prefix":
        return [
            {"role": "system", "content": prefix},
            {"role": "user", "content": f"{question} (запрос №{index})"},
        ]
    raise ValueError(f"неизвестный сценарий: {scenario}")


async def run_request(client, model: str, messages: list[dict], max_tokens: int) -> RequestResult:
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    completion_tokens = None
    prompt_tokens = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            # фиксированная длина ответа, чтобы уровни были сравнимы
            extra_body={"ignore_eos": True},
        )
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                completion_tokens = usage.completion_tokens
                prompt_tokens = usage.prompt_tokens
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    except Exception as exc:  # noqa: BLE001 — упавший запрос считаем в статистику, прогон не роняем
        return RequestResult(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
    e2e = time.perf_counter() - t0
    if ttft is None:
        return RequestResult(ok=False, e2e=e2e, error="пустой ответ")
    tokens = completion_tokens if completion_tokens is not None else chunks
    return RequestResult(ok=True, ttft=ttft, e2e=e2e, output_tokens=tokens, prompt_tokens=prompt_tokens)


async def run_level(
    client,
    model: str,
    concurrency: int,
    num_requests: int,
    scenario: str,
    max_tokens: int,
    prefix: str = "",
    start_index: int = 0,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(i: int) -> RequestResult:
        async with semaphore:
            return await run_request(client, model, build_messages(scenario, start_index + i, prefix), max_tokens)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(worker(i) for i in range(num_requests)))
    wall = time.perf_counter() - t0
    ok = [r for r in results if r.ok]
    errors = [r.error for r in results if not r.ok]
    total_tokens = sum(r.output_tokens for r in ok)
    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "errors": len(errors),
        "error_examples": errors[:3],
        "wall_s": wall,
        "output_throughput_tok_s": total_tokens / wall if wall > 0 else 0.0,
        "request_throughput_req_s": len(ok) / wall if wall > 0 else 0.0,
        "ttft_s": summarize(r.ttft for r in ok),
        "tpot_s": summarize(r.tpot for r in ok),
        "e2e_s": summarize(r.e2e for r in ok),
        "prompt_tokens": summarize(r.prompt_tokens for r in ok),
        "output_tokens": summarize(r.output_tokens for r in ok),
    }


def server_version(base_url: str) -> str | None:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        return http_get_json(f"{root}/version").get("version")
    except (OSError, ValueError, AttributeError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTFT / TPOT / throughput vLLM-сервера")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="unique")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--num-requests", type=int, default=0, help="запросов на уровень; 0 — max(16, 4×c)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prefix-items", type=int, default=16, help="размер общего префикса")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tag", default=None, help="имя файла results/bench_<tag>.json")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> dict:
    client = make_async_client(args.base_url)
    prefix = build_shared_prefix(args.prefix_items) if args.scenario == "shared-prefix" else ""
    for i in range(args.warmup):
        await run_request(client, args.model, build_messages(args.scenario, 900_000 + i, prefix), 16)

    levels = []
    index = 0
    for concurrency in args.concurrency:
        n = args.num_requests or max(16, 4 * concurrency)
        level = await run_level(client, args.model, concurrency, n, args.scenario, args.max_tokens, prefix, index)
        index += n
        levels.append(level)
        ttft = (level["ttft_s"] or {}).get("p50")
        print(
            f"c={concurrency:>3}: {level['output_throughput_tok_s']:8.1f} ток/с, "
            f"TTFT p50 {ttft * 1000 if ttft else float('nan'):7.0f} мс, ошибок {level['errors']}",
            flush=True,
        )
    await client.close()

    prompt_means = [lvl["prompt_tokens"]["mean"] for lvl in levels if lvl["prompt_tokens"]]
    return {
        "tag": args.tag or args.scenario,
        "model": args.model,
        "scenario": args.scenario,
        "max_tokens": args.max_tokens,
        "mean_prompt_tokens": sum(prompt_means) / len(prompt_means) if prompt_means else None,
        "server_version": server_version(args.base_url),
        "levels": levels,
        "env": env_info(),
    }


def _ms(stats: dict | None, key: str = "p50") -> str:
    return f"{stats[key] * 1000:.1f}" if stats else "—"


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    result = asyncio.run(run(args))
    path = save_json(result, RESULTS_DIR / f"bench_{result['tag']}.json")
    print_table(
        ["c", "ток/с", "запр/с", "TTFT p50 мс", "TTFT p90 мс", "TPOT p50 мс", "ошибки"],
        [
            [
                lvl["concurrency"],
                f"{lvl['output_throughput_tok_s']:.1f}",
                f"{lvl['request_throughput_req_s']:.2f}",
                _ms(lvl["ttft_s"]),
                _ms(lvl["ttft_s"], "p90"),
                _ms(lvl["tpot_s"]),
                lvl["errors"],
            ]
            for lvl in result["levels"]
        ],
    )
    print(f"сохранено: {path}")
    return result


if __name__ == "__main__":
    main()
