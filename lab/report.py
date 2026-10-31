"""Собирает results/*.json в RESULTS.md и рисует графики бенчмарка.

python -m lab.report
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from lab.common import RESULTS_DIR, ROOT, load_json


def _f(value, digits: int = 3, scale: float = 1.0) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * scale:.{digits}f}"


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _ms(stats: dict | None, key: str = "p50", digits: int = 0) -> str:
    return _f(stats.get(key) if stats else None, digits, 1000.0)


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _load(results_dir: Path, name: str):
    path = results_dir / name
    return load_json(path) if path.exists() else None


# --- разделы ---------------------------------------------------------------


def bench_plots(runs: dict, out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("bench_throughput.png", "выходных токенов в секунду", lambda lvl: lvl["output_throughput_tok_s"]),
        ("bench_ttft.png", "TTFT p50, мс", lambda lvl: (lvl["ttft_s"] or {}).get("p50", math.nan) * 1000),
    ]
    paths = []
    for filename, ylabel, getter in charts:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for tag, run in runs.items():
            ax.plot(
                [lvl["concurrency"] for lvl in run["levels"]],
                [getter(lvl) for lvl in run["levels"]],
                marker="o",
                label=tag,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("одновременных запросов")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)
        paths.append(out_dir / filename)
    return paths


def bench_section(results_dir: Path, plots: bool, base: Path = ROOT) -> list[str]:
    runs = {p.stem.removeprefix("bench_"): load_json(p) for p in sorted(results_dir.glob("bench_*.json"))}
    if not runs:
        return []
    out = ["## Бенчмарк vLLM", ""]
    for tag, run in runs.items():
        out += [
            f"**{tag}** — {run['model']}, сценарий `{run['scenario']}`, ответ {run['max_tokens']} токенов, "
            f"промпт ≈ {_f(run.get('mean_prompt_tokens'), 0)} токенов, vLLM {run.get('server_version') or '?'}",
            "",
            md_table(
                [
                    "одновременных запросов",
                    "выход, ток/с",
                    "запросов/с",
                    "TTFT p50, мс",
                    "TTFT p90, мс",
                    "TPOT p50, мс",
                    "ошибки",
                ],
                [
                    [
                        lvl["concurrency"],
                        _f(lvl["output_throughput_tok_s"], 1),
                        _f(lvl["request_throughput_req_s"], 2),
                        _ms(lvl["ttft_s"]),
                        _ms(lvl["ttft_s"], "p90"),
                        _ms(lvl["tpot_s"], digits=1),
                        lvl["errors"],
                    ]
                    for lvl in run["levels"]
                ],
            ),
            "",
        ]
    on, off = runs.get("prefix_on"), runs.get("prefix_off")
    if on and off:
        off_by_c = {lvl["concurrency"]: lvl for lvl in off["levels"]}
        rows = []
        for lvl in on["levels"]:
            other = off_by_c.get(lvl["concurrency"])
            if not other or not lvl["ttft_s"] or not other["ttft_s"]:
                continue
            ratio = other["ttft_s"]["p50"] / lvl["ttft_s"]["p50"]
            rows.append([lvl["concurrency"], _ms(other["ttft_s"]), _ms(lvl["ttft_s"]), f"×{ratio:.1f}"])
        out += [
            "**Prefix caching** — TTFT p50 с общим системным промптом",
            "",
            md_table(["одновременных запросов", "кэш выключен, мс", "кэш включён, мс", "ускорение"], rows),
            "",
        ]
    if plots:
        for path in bench_plots(runs, results_dir / "plots"):
            out += [f"![{path.stem}]({Path(os.path.relpath(path, base)).as_posix()})", ""]
    return out


def quant_section(results_dir: Path) -> list[str]:
    data = _load(results_dir, "quant.json")
    if not data:
        return []
    rows = []
    for v in data["variants"]:
        if "error" in v:
            rows.append([v["name"], v["model"], "—", "—", "—", "—", v["error"]])
            continue
        rows.append(
            [
                v["name"],
                v["model"],
                _f(v["weights_gib"], 2),
                _f(v["gsm8k_accuracy"]),
                _f(v["batch_throughput_tok_s"], 0),
                _f(v["single_stream_tok_s"], 1),
                "",
            ]
        )
    n = next((v["n"] for v in data["variants"] if "n" in v), "?")
    return [
        "## Квантизация",
        "",
        f"GSM8K, {n} задач, greedy. «Пакет» — все задачи разом, «1 запрос» — один поток, 256 токенов.",
        "",
        md_table(["вариант", "модель", "веса, ГиБ", "GSM8K", "пакет, ток/с", "1 запрос, ток/с", "ошибка"], rows),
        "",
    ]


def appeals_section(results_dir: Path) -> list[str]:
    data, train, ev = (_load(results_dir, n) for n in ("appeals_data.json", "appeals_train.json", "appeals_eval.json"))
    if not (data or train or ev):
        return []
    out = ["## Классификатор обращений (QLoRA)", ""]
    if data:
        out += [
            f"Синтетический датасет от {data['generator_model']}: train {sum(data['train'].values())}, "
            f"test {sum(data['test'].values())}, {len(data['train'])} категорий. Стили теста в трейне не встречаются.",
            "",
        ]
    if train:
        out += [
            f"Обучение: {train['base_model']}, LoRA r={train['lora_r']}, {train['steps']} шагов, "
            f"{_f(train['train_minutes'], 1)} мин, пик памяти {_f(train['peak_gpu_mem_gib'], 1)} ГиБ, "
            f"обучаемых параметров {train['trainable_params'] / train['total_params']:.2%}.",
            "",
        ]
    if ev:
        out += [
            md_table(
                ["режим", "accuracy", "macro-F1", "нераспознанных ответов"],
                [[m, _f(r["accuracy"]), _f(r["macro_f1"]), _pct(r["invalid_rate"])] for m, r in ev["modes"].items()],
            ),
            "",
        ]
    return out


def rag_section(results_dir: Path) -> list[str]:
    retrieval, faith = _load(results_dir, "rag_retrieval.json"), _load(results_dir, "rag_faithfulness.json")
    if not (retrieval or faith):
        return []
    out = ["## RAG на SberQuAD", ""]
    if retrieval:
        out += [
            f"Корпус {retrieval['n_docs']} абзацев, {retrieval['n_queries']} вопросов. "
            f"Эмбеддинги {retrieval['embedder']}, reranker {retrieval['reranker']} (top-{retrieval['rerank_top']}).",
            "",
            md_table(
                ["метод", "recall@1", "recall@5", "recall@10", "MRR@10", "мс/запрос"],
                [
                    [
                        n,
                        _f(m["recall@1"]),
                        _f(m["recall@5"]),
                        _f(m["recall@10"]),
                        _f(m["mrr@10"]),
                        _f(m["ms_per_query"], 1),
                    ]
                    for n, m in retrieval["methods"].items()
                ],
            ),
            "",
        ]
    if faith:
        m = faith["metrics"]
        hit, miss = m["when_context_hit"], m["when_context_miss"]
        out += [
            f"Генерация {faith['generator_model']}, судья {faith['judge_model']}, "
            f"top-{faith['top_k']} абзаца, {m['n']} вопросов.",
            "",
            md_table(
                ["", "все", "нужный абзац найден", "не найден"],
                [
                    [
                        "доля вопросов",
                        "100%",
                        _pct(m["context_hit_rate"]),
                        _pct(1 - m["context_hit_rate"]) if m["context_hit_rate"] is not None else "—",
                    ],
                    ["faithfulness", _f(m["faithfulness"]), _f(hit["faithfulness"]), _f(miss["faithfulness"])],
                    [
                        "ответ содержит эталон",
                        _pct(m["answer_contains_gold"]),
                        _pct(hit["answer_contains_gold"]),
                        _pct(miss["answer_contains_gold"]),
                    ],
                    [
                        "отказ «нет ответа»",
                        _pct(m["abstain_rate"]),
                        _pct(hit["abstain_rate"]),
                        _pct(miss["abstain_rate"]),
                    ],
                ],
            ),
            "",
        ]
    return out


def tools_section(results_dir: Path) -> list[str]:
    data = _load(results_dir, "tools_eval.json")
    if not data:
        return []
    out = ["## Function calling и structured output", "", f"Модель: {data['model']}.", ""]
    if "tools" in data:
        m = data["tools"]["metrics"]
        out += [
            md_table(
                [
                    "кейсов",
                    "пройдено",
                    "выбор инструмента",
                    "аргументы",
                    "лишние вызовы",
                    "пропущенные вызовы",
                    "нарушения схемы",
                ],
                [
                    [
                        m["n_cases"],
                        _pct(m["pass_rate"]),
                        _pct(m["tool_selection_accuracy"]),
                        _pct(m["args_accuracy"]),
                        _pct(m["false_call_rate"]),
                        _pct(m["missed_call_rate"]),
                        f"{m['schema_violations']} из {m['n_calls']}",
                    ]
                ],
            ),
            "",
            "По группам: "
            + ", ".join(f"{g} {_pct(r)}" for g, r in m["pass_rate_by_group"].items())
            + (f". Провалены: {', '.join(m['failed_cases'])}." if m["failed_cases"] else "."),
            "",
        ]
    if "structured" in data:
        s = data["structured"]
        out += [
            f"Извлечение полей из {s['n']} обращений:",
            "",
            md_table(
                ["режим", "чистый JSON", "JSON после очистки", "валиден по схеме", "категория верна", "задержка, с"],
                [
                    [
                        mode,
                        _pct(r["strict_json"]),
                        _pct(r["lenient_json"]),
                        _pct(r["schema_valid"]),
                        _pct(r["category_correct"]),
                        _f(r["latency_s"], 2),
                    ]
                    for mode, r in s["modes"].items()
                ],
            ),
            "",
        ]
    return out


def environment_line(results_dir: Path) -> list[str]:
    for path in sorted(results_dir.glob("*.json")):
        data = load_json(path)
        env = data.get("env") if isinstance(data, dict) else None
        if env:
            packages = ", ".join(f"{k} {v}" for k, v in env["packages"].items())
            gpus = "; ".join(env["gpus"]) or "GPU не найден"
            return [f"Окружение: {gpus}. Python {env['python']}, {packages}.", ""]
    return []


def build_report(results_dir: Path = RESULTS_DIR, plots: bool = True, base: Path = ROOT) -> str:
    sections = [
        bench_section(results_dir, plots, base),
        quant_section(results_dir),
        appeals_section(results_dir),
        rag_section(results_dir),
        tools_section(results_dir),
    ]
    body = [line for section in sections for line in section]
    header = ["# Результаты", ""]
    if not body:
        return "\n".join([*header, "Пока пусто: запусти `notebooks/kaggle.ipynb`, затем `python -m lab.report`.", ""])
    return "\n".join(
        [
            *header,
            "Таблицы собраны скриптом `python -m lab.report` из файлов `results/*.json`.",
            "",
            *environment_line(results_dir),
            *body,
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RESULTS.md из results/*.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "RESULTS.md")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.results_dir, plots=not args.no_plots, base=args.output.resolve().parent)
    args.output.write_text(report, encoding="utf-8")
    print(f"готово: {args.output}")


if __name__ == "__main__":
    main()
