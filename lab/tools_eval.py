"""Function calling и structured output через OpenAI-совместимый API vLLM.

tools       Набор тест-кейсов (data/tools_cases.json) для ассистента портала обращений:
            выбирает ли модель нужный инструмент, правильно ли заполняет аргументы,
            соответствуют ли аргументы JSON-схеме и не вызывает ли инструмент,
            когда данных не хватает и нужно переспросить.

structured  Извлечение полей из обращения в JSON двумя способами: только инструкцией в промпте
            и с response_format=json_schema (vLLM ограничивает генерацию схемой).
            Сравниваются доля парсящихся и валидных по схеме ответов.

    python -m lab.server start Qwen/Qwen2.5-7B-Instruct-AWQ -- \\
        --enable-auto-tool-choice --tool-call-parser hermes
    python -m lab.tools_eval all --model Qwen/Qwen2.5-7B-Instruct-AWQ
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict

from lab.appeals.core import LABELS, read_appeals_csv
from lab.common import (
    DATA_DIR,
    RESULTS_DIR,
    env_info,
    json_schema_format,
    load_json,
    make_async_client,
    parse_json_object,
    save_json,
)
from lab.metrics import normalize_text

CASES_PATH = DATA_DIR / "tools_cases.json"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_appeal_status",
            "description": "Узнать статус ранее поданного обращения по его номеру",
            "parameters": {
                "type": "object",
                "properties": {
                    "appeal_id": {"type": "string", "description": "Номер обращения, 6 цифр", "pattern": "^[0-9]{6}$"}
                },
                "required": ["appeal_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_appeal",
            "description": "Зарегистрировать новое обращение гражданина о проблеме",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": LABELS},
                    "address": {"type": "string", "description": "Адрес или место проблемы"},
                    "description": {"type": "string", "description": "Суть проблемы"},
                },
                "required": ["category", "address", "description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_mfc",
            "description": "Найти ближайший МФЦ и часы его работы",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "district": {"type": "string", "description": "Район города, если указан"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_utility_debt",
            "description": "Проверить задолженность по оплате ЖКУ по номеру лицевого счёта",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_number": {
                        "type": "string",
                        "description": "Номер лицевого счёта, 10 цифр",
                        "pattern": "^[0-9]{10}$",
                    }
                },
                "required": ["account_number"],
                "additionalProperties": False,
            },
        },
    },
]
TOOL_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

TOOLS_SYSTEM = (
    "Ты ассистент регионального портала обращений граждан. Используй инструменты, когда пользователь "
    "просит действие, для которого они предназначены. Если для вызова не хватает данных (номера "
    "обращения, лицевого счёта, адреса проблемы), не вызывай инструмент и не придумывай значения — "
    "уточни у пользователя. На посторонние вопросы отвечай без инструментов."
)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": LABELS},
        "address": {"type": "string", "description": "адрес или место; пустая строка, если не указан"},
        "urgency": {"type": "string", "enum": ["низкая", "средняя", "высокая"]},
        "summary": {"type": "string", "description": "суть обращения в одном предложении"},
    },
    "required": ["category", "address", "urgency", "summary"],
    "additionalProperties": False,
}
EXTRACT_SYSTEM = (
    "Извлеки из обращения гражданина поля и верни только JSON без пояснений:\n"
    f"category — одна из: {', '.join(LABELS)};\n"
    "address — адрес или место проблемы, пустая строка, если не указан;\n"
    "urgency — низкая, средняя или высокая;\n"
    "summary — суть обращения в одном предложении."
)
FALLBACK_SAMPLES = [
    {"text": "На Ленина 15 вторую неделю течёт крыша, вода уже в подъезде.", "label": "ЖКХ"},
    {"text": "Возле школы на Садовой огромная яма, машины бьют колёса.", "label": "Дороги и транспорт"},
    {"text": "Во дворе дома 8 по проспекту Мира сломали качели.", "label": "Благоустройство"},
    {"text": "В поликлинике на Гвардейской невозможно записаться к терапевту.", "label": "Здравоохранение"},
    {"text": "Детский сад на Фучика закрыли на ремонт, родителей не предупредили.", "label": "Образование"},
    {"text": "Мне третий месяц не приходит пособие на ребёнка.", "label": "Социальная поддержка"},
    {"text": "Завод за рекой опять дымит по ночам, дышать нечем.", "label": "Экология"},
    {"text": "Не могу подтвердить учётную запись на Госуслугах, сайт выдаёт ошибку.", "label": "Цифровые госуслуги"},
]


# --- оценка вызова инструмента ---------------------------------------------


def check_call(call: dict) -> dict:
    """Аргументы — валидный JSON? Соответствуют схеме инструмента?"""
    from jsonschema import Draft202012Validator

    raw = call.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {"args": None, "json_valid": False, "schema_valid": False, "schema_errors": ["невалидный JSON"]}
    if not isinstance(args, dict):
        return {"args": None, "json_valid": False, "schema_valid": False, "schema_errors": ["аргументы не объект"]}
    schema = TOOL_SCHEMAS.get(call.get("name"))
    if schema is None:
        return {"args": args, "json_valid": True, "schema_valid": False, "schema_errors": ["неизвестный инструмент"]}
    errors = [e.message for e in Draft202012Validator(schema).iter_errors(args)]
    return {"args": args, "json_valid": True, "schema_valid": not errors, "schema_errors": errors}


def args_match(args: dict, case: dict) -> bool:
    """expected_args — точное совпадение (список = любое из), expected_contains — подстрока без учёта регистра."""
    for key, expected in case.get("expected_args", {}).items():
        options = expected if isinstance(expected, list) else [expected]
        if str(args.get(key, "")).strip() not in [str(o) for o in options]:
            return False
    for key, expected in case.get("expected_contains", {}).items():
        options = expected if isinstance(expected, list) else [expected]
        value = normalize_text(str(args.get(key, "")))
        if not any(normalize_text(o) in value for o in options):
            return False
    return True


def score_case(case: dict, calls: list[dict]) -> dict:
    checked = [{**c, **check_call(c)} for c in calls]
    expected = case.get("expected_tool")
    if expected is None:
        selection_ok, args_ok = not calls, None
    else:
        matching = [c for c in checked if c["name"] == expected]
        selection_ok = bool(matching)
        args_ok = any(c["args"] is not None and args_match(c["args"], case) for c in matching)
    return {
        "id": case["id"],
        "group": case.get("group", ""),
        "expected_tool": expected,
        "called": [c["name"] for c in calls],
        "selection_ok": selection_ok,
        "args_ok": args_ok,
        "passed": selection_ok and args_ok is not False,
        "calls": checked,
    }


def aggregate_tools(scored: list[dict]) -> dict:
    def rate(items, key):
        items = list(items)
        return sum(bool(i[key]) for i in items) / len(items) if items else None

    with_tool = [s for s in scored if s["expected_tool"] is not None]
    without_tool = [s for s in scored if s["expected_tool"] is None]
    all_calls = [c for s in scored for c in s["calls"]]
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        groups[s["group"]].append(s)
    return {
        "n_cases": len(scored),
        "pass_rate": rate(scored, "passed"),
        "tool_selection_accuracy": rate(scored, "selection_ok"),
        "args_accuracy": rate((s for s in with_tool if s["selection_ok"]), "args_ok"),
        "missed_call_rate": sum(1 for s in with_tool if not s["called"]) / len(with_tool) if with_tool else None,
        "false_call_rate": sum(1 for s in without_tool if s["called"]) / len(without_tool) if without_tool else None,
        "n_calls": len(all_calls),
        "invalid_json_calls": sum(1 for c in all_calls if not c["json_valid"]),
        "schema_violations": sum(1 for c in all_calls if not c["schema_valid"]),
        "pass_rate_by_group": {g: rate(items, "passed") for g, items in sorted(groups.items())},
        "failed_cases": [s["id"] for s in scored if not s["passed"]],
    }


async def run_tools(client, model: str, cases: list[dict], concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: dict) -> dict:
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": TOOLS_SYSTEM},
                        {"role": "user", "content": case["message"]},
                    ],
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.0,
                    max_tokens=300,
                )
            except Exception as exc:  # noqa: BLE001 — ошибка запроса = проваленный кейс
                result = score_case(case, [])
                result.update(passed=False, selection_ok=False, error=f"{type(exc).__name__}: {exc}"[:200])
                return result
        message = resp.choices[0].message
        calls = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in (message.tool_calls or [])]
        result = score_case(case, calls)
        result["content"] = (message.content or "")[:300]
        return result

    scored = await asyncio.gather(*(one(c) for c in cases))
    return {"metrics": aggregate_tools(scored), "cases": scored}


# --- structured output -----------------------------------------------------


def evaluate_extraction(content: str, label: str) -> dict:
    from jsonschema import Draft202012Validator

    try:
        strict = json.loads(content.strip())
        strict_ok = isinstance(strict, dict)
    except json.JSONDecodeError:
        strict_ok = False
    data = parse_json_object(content)
    schema_ok = data is not None and not list(Draft202012Validator(EXTRACT_SCHEMA).iter_errors(data))
    return {
        "strict_json": strict_ok,
        "lenient_json": data is not None,
        "schema_valid": schema_ok,
        "category_correct": bool(data) and data.get("category") == label,
    }


async def run_structured(client, model: str, samples: list[dict], concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(sample: dict, constrained: bool) -> dict:
        kwargs = {"response_format": json_schema_format("appeal_fields", EXTRACT_SCHEMA)} if constrained else {}
        t0 = time.perf_counter()
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": EXTRACT_SYSTEM},
                        {"role": "user", "content": sample["text"]},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                    **kwargs,
                )
                content = resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                content = f"ERROR {type(exc).__name__}: {exc}"
        return {**evaluate_extraction(content, sample["label"]), "latency_s": time.perf_counter() - t0}

    report = {}
    for mode, constrained in (("prompt_only", False), ("json_schema", True)):
        rows = await asyncio.gather(*(one(s, constrained) for s in samples))
        report[mode] = {
            key: sum(r[key] for r in rows) / len(rows)
            for key in ("strict_json", "lenient_json", "schema_valid", "category_correct", "latency_s")
        }
    return {"n": len(samples), "modes": report}


def load_samples(n: int) -> list[dict]:
    path = DATA_DIR / "appeals" / "test.csv"
    if path.exists():
        return read_appeals_csv(path)[:n]
    return FALLBACK_SAMPLES


async def run_all(args: argparse.Namespace) -> dict:
    client = make_async_client(args.base_url, max_retries=2)
    result: dict = {"model": args.model, "env": env_info()}
    if args.cmd in ("tools", "all"):
        result["tools"] = await run_tools(client, args.model, load_json(CASES_PATH), args.concurrency)
    if args.cmd in ("structured", "all"):
        result["structured"] = await run_structured(client, args.model, load_samples(args.n), args.concurrency)
    await client.close()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function calling и structured output")
    parser.add_argument("cmd", choices=["tools", "structured", "all"])
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=100, help="обращений для structured")
    parser.add_argument("--concurrency", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    result = asyncio.run(run_all(parse_args(argv)))
    save_json(result, RESULTS_DIR / "tools_eval.json")
    if "tools" in result:
        m = result["tools"]["metrics"]
        print(
            f"tools: пройдено {m['pass_rate']:.1%} кейсов, выбор инструмента {m['tool_selection_accuracy']:.1%}, "
            f"лишние вызовы {m['false_call_rate']:.1%}; провалены: {', '.join(m['failed_cases']) or '—'}"
        )
    if "structured" in result:
        for mode, m in result["structured"]["modes"].items():
            print(
                f"structured/{mode}: JSON {m['strict_json']:.1%}, по схеме {m['schema_valid']:.1%}, "
                f"категория {m['category_correct']:.1%}"
            )
    return result


if __name__ == "__main__":
    main()
