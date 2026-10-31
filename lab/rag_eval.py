"""Оценка RAG на SberQuAD (русский QA-датасет, вопрос + абзац Википедии с ответом).

retrieval     Корпус — все уникальные абзацы train+validation (~10 тыс.), запросы — вопросы
              из validation. Для каждого вопроса известен «правильный» абзац, поэтому
              recall@k и MRR считаются без ручной разметки. Сравниваются BM25, e5,
              гибрид (RRF) и гибрид с reranker'ом.

faithfulness  По top-3 найденным абзацам модель отвечает на вопрос, затем LLM-судья разбивает
              ответ на утверждения и проверяет, следует ли каждое из контекста.
              faithfulness ответа = доля подтверждённых утверждений.

    python -m lab.rag_eval retrieval --n-queries 1000
    python -m lab.server start Qwen/Qwen2.5-7B-Instruct-AWQ
    python -m lab.rag_eval faithfulness --model Qwen/Qwen2.5-7B-Instruct-AWQ --n 150
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from itertools import chain

from lab.common import (
    CACHE_DIR,
    RESULTS_DIR,
    env_info,
    json_schema_format,
    load_json,
    make_async_client,
    parse_json_object,
    print_table,
    read_jsonl,
    save_json,
    write_jsonl,
)
from lab.metrics import answer_contains, mrr_at_k, normalize_text, recall_at_k
from lab.retrieval import BM25, DenseIndex, Reranker, reciprocal_rank_fusion

DATASET = "kuznetsoffandrey/sberquad"
RUNS_PATH = CACHE_DIR / "rag_runs.jsonl"
DOCS_PATH = CACHE_DIR / "rag_docs.json"
KS = (1, 3, 5, 10)
NO_ANSWER = "НЕТ ОТВЕТА"

GEN_SYSTEM = (
    "Ты отвечаешь на вопросы строго по приведённым фрагментам. Не добавляй фактов, которых нет "
    f"во фрагментах. Если ответа во фрагментах нет, напиши ровно: {NO_ANSWER}. "
    "Отвечай одним-двумя предложениями."
)
JUDGE_SYSTEM = (
    "Ты проверяешь ответ на соответствие контексту. Разбей ответ на атомарные утверждения "
    "(одно утверждение — один факт). Для каждого поставь supported=true, только если оно прямо "
    "следует из контекста; общие знания не считаются. Верни JSON."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


# --- retrieval --------------------------------------------------------------


def _answers(value) -> list[str]:
    if isinstance(value, dict):  # Sequence-фича datasets: {"text": [...], "answer_start": [...]}
        return [t for t in value.get("text", []) if t]
    return [a["text"] for a in value if a.get("text")]


def load_sberquad(n_queries: int, seed: int) -> tuple[list[str], list[dict]]:
    from datasets import load_dataset

    train = load_dataset(DATASET, split="train")
    val = load_dataset(DATASET, split="validation")
    doc_ids: dict[str, int] = {}
    for ctx in chain(train["context"], val["context"]):
        doc_ids.setdefault(ctx.strip(), len(doc_ids))
    queries = [
        {"question": q.strip(), "answers": _answers(a), "gold_doc": doc_ids[c.strip()]}
        for q, a, c in zip(val["question"], val["answers"], val["context"])
    ]
    random.Random(seed).shuffle(queries)
    return list(doc_ids), queries[:n_queries]


def ranking_metrics(ranked: list[list[int]], gold: list[int]) -> dict:
    relevant = [{g} for g in gold]
    out = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in KS}
    out["mrr@10"] = mrr_at_k(ranked, relevant, 10)
    return out


def cmd_retrieval(args: argparse.Namespace) -> dict:
    docs, queries = load_sberquad(args.n_queries, args.seed)
    questions = [q["question"] for q in queries]
    gold = [q["gold_doc"] for q in queries]
    print(f"корпус: {len(docs)} абзацев, запросов: {len(queries)}", flush=True)
    per_query: dict[str, float] = {}

    t = time.perf_counter()
    bm25 = BM25().fit(docs)
    bm25_index_s = time.perf_counter() - t
    t = time.perf_counter()
    bm25_runs = [bm25.search(q, args.depth) for q in questions]
    per_query["bm25"] = (time.perf_counter() - t) / len(questions)

    t = time.perf_counter()
    dense = DenseIndex(args.embedder).fit(docs)
    dense_index_s = time.perf_counter() - t
    t = time.perf_counter()
    dense_runs = dense.search(questions, args.depth)
    per_query["e5"] = (time.perf_counter() - t) / len(questions)

    t = time.perf_counter()
    hybrid_runs = [
        reciprocal_rank_fusion([b, d], k=args.rrf_k, limit=args.depth) for b, d in zip(bm25_runs, dense_runs)
    ]
    per_query["hybrid_rrf"] = per_query["bm25"] + per_query["e5"] + (time.perf_counter() - t) / len(questions)

    reranker = Reranker(args.reranker)
    t = time.perf_counter()
    rerank_runs = reranker.rerank_many(questions, [h[: args.rerank_top] for h in hybrid_runs], docs)
    per_query["hybrid_rrf+rerank"] = per_query["hybrid_rrf"] + (time.perf_counter() - t) / len(questions)

    runs = {"bm25": bm25_runs, "e5": dense_runs, "hybrid_rrf": hybrid_runs, "hybrid_rrf+rerank": rerank_runs}
    metrics = {name: {**ranking_metrics(r, gold), "ms_per_query": 1000 * per_query[name]} for name, r in runs.items()}

    save_json(docs, DOCS_PATH)
    write_jsonl(
        [{**q, "reranked": r[:10], "hybrid": h[:10]} for q, r, h in zip(queries, rerank_runs, hybrid_runs)],
        RUNS_PATH,
    )
    result = {
        "dataset": DATASET,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "embedder": args.embedder,
        "reranker": args.reranker,
        "rerank_top": args.rerank_top,
        "rrf_k": args.rrf_k,
        "index_build_s": {"bm25": bm25_index_s, "e5": dense_index_s},
        "methods": metrics,
        "env": env_info(),
    }
    save_json(result, RESULTS_DIR / "rag_retrieval.json")
    print_table(
        ["метод", "R@1", "R@5", "R@10", "MRR@10", "мс/запрос"],
        [
            [
                n,
                f"{m['recall@1']:.3f}",
                f"{m['recall@5']:.3f}",
                f"{m['recall@10']:.3f}",
                f"{m['mrr@10']:.3f}",
                f"{m['ms_per_query']:.1f}",
            ]
            for n, m in metrics.items()
        ],
    )
    return result


# --- faithfulness -----------------------------------------------------------


def format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))


def is_abstained(answer: str) -> bool:
    return normalize_text(NO_ANSWER) in normalize_text(answer)


async def generate_answer(client, model: str, question: str, contexts: list[str]) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"Фрагменты:\n{format_contexts(contexts)}\n\nВопрос: {question}"},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    return (resp.choices[0].message.content or "").strip()


async def judge_answer(client, model: str, contexts: list[str], answer: str) -> list[dict] | None:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Контекст:\n{format_contexts(contexts)}\n\nОтвет:\n{answer}"},
        ],
        temperature=0.0,
        max_tokens=800,
        response_format=json_schema_format("faithfulness_judge", JUDGE_SCHEMA),
    )
    data = parse_json_object(resp.choices[0].message.content)
    claims = data.get("claims") if data else None
    return claims if isinstance(claims, list) else None


def aggregate_faithfulness(rows: list[dict]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    answered = [r for r in rows if not r["abstained"]]
    hit = [r for r in rows if r["context_hit"]]
    miss = [r for r in rows if not r["context_hit"]]
    return {
        "n": len(rows),
        "context_hit_rate": len(hit) / len(rows) if rows else None,
        "abstain_rate": 1 - len(answered) / len(rows) if rows else None,
        "faithfulness": mean(r["faithfulness"] for r in answered),
        "answer_contains_gold": mean(float(r["contains_gold"]) for r in rows),
        "judge_errors": sum(1 for r in answered if r["claims"] is None),
        "when_context_hit": {
            "faithfulness": mean(r["faithfulness"] for r in hit if not r["abstained"]),
            "answer_contains_gold": mean(float(r["contains_gold"]) for r in hit),
            "abstain_rate": mean(float(r["abstained"]) for r in hit),
        },
        "when_context_miss": {
            "faithfulness": mean(r["faithfulness"] for r in miss if not r["abstained"]),
            "answer_contains_gold": mean(float(r["contains_gold"]) for r in miss),
            "abstain_rate": mean(float(r["abstained"]) for r in miss),
        },
    }


async def run_faithfulness(args: argparse.Namespace) -> dict:
    docs = load_json(DOCS_PATH)
    runs = read_jsonl(RUNS_PATH)[: args.n]
    judge_model = args.judge_model or args.model
    client = make_async_client(args.base_url, max_retries=2)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(run: dict) -> dict:
        ids = run["reranked"][: args.top_k]
        contexts = [docs[i] for i in ids]
        async with semaphore:
            answer = await generate_answer(client, args.model, run["question"], contexts)
            abstained = is_abstained(answer)
            claims = None if abstained else await judge_answer(client, judge_model, contexts, answer)
        faithfulness = None
        if claims:
            faithfulness = sum(bool(c.get("supported")) for c in claims) / len(claims)
        return {
            "question": run["question"],
            "answers": run["answers"],
            "answer": answer,
            "abstained": abstained,
            "context_hit": run["gold_doc"] in ids,
            "contains_gold": answer_contains(answer, run["answers"]),
            "claims": claims,
            "faithfulness": faithfulness,
        }

    rows = await asyncio.gather(*(one(r) for r in runs))
    await client.close()
    write_jsonl(rows, CACHE_DIR / "rag_faithfulness_rows.jsonl")
    worst = sorted((r for r in rows if r["faithfulness"] is not None), key=lambda r: r["faithfulness"])[:5]
    return {
        "generator_model": args.model,
        "judge_model": judge_model,
        "top_k": args.top_k,
        "metrics": aggregate_faithfulness(rows),
        "least_faithful_examples": [{k: r[k] for k in ("question", "answer", "faithfulness", "claims")} for r in worst],
        "env": env_info(),
    }


def cmd_faithfulness(args: argparse.Namespace) -> dict:
    if not RUNS_PATH.exists():
        raise SystemExit("сначала запусти: python -m lab.rag_eval retrieval")
    result = asyncio.run(run_faithfulness(args))
    save_json(result, RESULTS_DIR / "rag_faithfulness.json")
    m = result["metrics"]
    print(
        f"faithfulness {m['faithfulness']}, ответ содержит эталон {m['answer_contains_gold']}, "
        f"отказов {m['abstain_rate']}, контекст найден {m['context_hit_rate']}"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка RAG на SberQuAD")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--n-queries", type=int, default=1000)
    r.add_argument("--depth", type=int, default=100, help="глубина выдачи BM25 и e5 перед слиянием")
    r.add_argument("--rerank-top", type=int, default=20)
    r.add_argument("--rrf-k", type=int, default=60)
    r.add_argument("--embedder", default="intfloat/multilingual-e5-base")
    r.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    r.add_argument("--seed", type=int, default=42)
    f = sub.add_parser("faithfulness")
    f.add_argument("--base-url", default="http://localhost:8000/v1")
    f.add_argument("--model", required=True)
    f.add_argument("--judge-model", default=None, help="по умолчанию та же модель")
    f.add_argument("--n", type=int, default=150)
    f.add_argument("--top-k", type=int, default=3)
    f.add_argument("--concurrency", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    return cmd_retrieval(args) if args.cmd == "retrieval" else cmd_faithfulness(args)


if __name__ == "__main__":
    main()
