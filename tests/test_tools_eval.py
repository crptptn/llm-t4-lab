import asyncio
import json
import re

import pytest
from jsonschema import Draft202012Validator

from lab.common import load_json, make_async_client
from lab.tools_eval import CASES_PATH, TOOL_SCHEMAS, aggregate_tools, evaluate_extraction, run_tools, score_case

CASES = load_json(CASES_PATH)


def call(name, **args):
    return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def case(**kw):
    return {"id": "c", "group": "g", "message": "", **kw}


def test_correct_call_passes():
    c = case(expected_tool="get_appeal_status", expected_args={"appeal_id": "482913"})
    s = score_case(c, [call("get_appeal_status", appeal_id="482913")])
    assert s["passed"] and s["calls"][0]["schema_valid"]


def test_wrong_argument_fails():
    c = case(expected_tool="get_appeal_status", expected_args={"appeal_id": "482913"})
    s = score_case(c, [call("get_appeal_status", appeal_id="111111")])
    assert s["selection_ok"] and not s["args_ok"] and not s["passed"]


def test_contains_and_any_of():
    c = case(
        expected_tool="create_appeal",
        expected_args={"category": ["ЖКХ", "Экология"]},
        expected_contains={"address": "Пушкина"},
    )
    ok = call("create_appeal", category="Экология", address="ул. Пушкина, 3", description="мусор")
    assert score_case(c, [ok])["passed"]


def test_call_when_clarification_needed_fails():
    s = score_case(case(expected_tool=None), [call("get_utility_debt", account_number="77712")])
    assert not s["passed"]
    assert not s["calls"][0]["schema_valid"]  # 5 цифр не проходят pattern


def test_invalid_json_arguments():
    s = score_case(case(expected_tool="find_mfc"), [{"name": "find_mfc", "arguments": "{city: Казань"}])
    assert not s["calls"][0]["json_valid"] and not s["passed"]


def test_aggregate():
    scored = [
        score_case(case(id="a", expected_tool=None), []),
        score_case(case(id="b", expected_tool=None), [call("find_mfc", city="Казань")]),
        score_case(case(id="c", expected_tool="find_mfc"), []),
    ]
    m = aggregate_tools(scored)
    assert m["pass_rate"] == pytest.approx(1 / 3)
    assert m["false_call_rate"] == 0.5 and m["missed_call_rate"] == 1.0
    assert m["failed_cases"] == ["b", "c"]


def test_cases_file_is_consistent():
    """Тест-кейсы сами по себе корректны: инструменты существуют, точные аргументы проходят схему."""
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    for c in CASES:
        tool = c["expected_tool"]
        if tool is None:
            assert not c.get("expected_args") and not c.get("expected_contains")
            continue
        schema = TOOL_SCHEMAS[tool]
        for key in [*c.get("expected_args", {}), *c.get("expected_contains", {})]:
            assert key in schema["properties"], (c["id"], key)
        for key, value in c.get("expected_args", {}).items():
            prop = schema["properties"][key]
            for option in value if isinstance(value, list) else [value]:
                if "pattern" in prop:
                    assert re.fullmatch(prop["pattern"].strip("^$"), option), (c["id"], option)
                if "enum" in prop:
                    assert option in prop["enum"], (c["id"], option)
    Draft202012Validator.check_schema(TOOL_SCHEMAS["create_appeal"])


def test_extraction_scoring():
    good = '{"category": "ЖКХ", "address": "Ленина 1", "urgency": "высокая", "summary": "Течёт крыша"}'
    r = evaluate_extraction(good, "ЖКХ")
    assert r == {"strict_json": True, "lenient_json": True, "schema_valid": True, "category_correct": True}
    fenced = evaluate_extraction(f"```json\n{good}\n```", "ЖКХ")
    assert not fenced["strict_json"] and fenced["lenient_json"] and fenced["schema_valid"]
    wrong_enum = evaluate_extraction(good.replace("высокая", "срочно"), "ЖКХ")
    assert not wrong_enum["schema_valid"]


def test_run_tools_against_fake_server(fake_openai):
    def responder(body):
        assert body["tools"] and body["tool_choice"] == "auto"
        text = body["messages"][-1]["content"]
        if "482913" in text:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "get_appeal_status", "arguments": '{"appeal_id": "482913"}'},
                    }
                ],
            }
        return {"content": "Уточните, пожалуйста, детали."}

    fake_openai.set_responder(responder)
    cases = [c for c in CASES if c["id"] in {"status_simple", "status_no_id", "chitchat", "mfc_city"}]

    async def go():
        client = make_async_client(fake_openai.url)
        try:
            return await run_tools(client, "fake", cases, concurrency=2)
        finally:
            await client.close()

    m = asyncio.run(go())["metrics"]
    assert m["n_cases"] == 4
    assert m["failed_cases"] == ["mfc_city"]
    assert m["pass_rate"] == 0.75
