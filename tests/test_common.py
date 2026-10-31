from lab.common import json_schema_format, parse_json_object, read_jsonl, write_jsonl


def test_parse_json_object_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_object('Вот ответ: {"a": 3}. Готово') == {"a": 3}
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("не json") is None
    assert parse_json_object(None) is None


def test_json_schema_format():
    fmt = json_schema_format("x", {"type": "object"})
    assert fmt == {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}


def test_jsonl_roundtrip(tmp_path):
    rows = [{"text": "привет"}, {"n": 2}]
    assert read_jsonl(write_jsonl(rows, tmp_path / "a" / "b.jsonl")) == rows
