import asyncio
import random

import pytest

from lab.appeals.core import (
    LABELS,
    build_messages,
    build_training_example,
    pad_batch,
    read_appeals_csv,
    write_appeals_csv,
)
from lab.appeals.evaluate import classify_all
from lab.appeals.generate import build_prompt, dedupe, plan_jobs, take_per_label
from lab.common import make_async_client
from lab.metrics import classification_metrics


class CharTokenizer:
    """Токен = символ. Достаточно, чтобы проверить маскирование промпта."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def test_training_example_masks_prompt():
    tok = CharTokenizer()
    ex = build_training_example(tok, "Течёт крыша", "ЖКХ")
    target = [ord(c) for c in "ЖКХ"] + [0]
    assert ex["input_ids"][-len(target) :] == target
    assert ex["labels"][-len(target) :] == target
    assert set(ex["labels"][: -len(target)]) == {-100}
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_training_example_truncates_long_text():
    tok = CharTokenizer()
    short = build_training_example(tok, "а" * 10, "ЖКХ", max_text_tokens=10)
    long = build_training_example(tok, "а" * 500, "ЖКХ", max_text_tokens=10)
    assert len(long["input_ids"]) == len(short["input_ids"])


def test_pad_batch():
    batch = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [8], "labels": [8]}]
    out = pad_batch(batch, pad_id=0)
    assert out["input_ids"] == [[5, 6, 7], [8, 0, 0]]
    assert out["labels"] == [[-100, 6, 7], [8, -100, -100]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


def test_messages_list_all_categories():
    user = build_messages("текст")[1]["content"]
    assert all(f"- {label}" in user for label in LABELS)


def test_csv_roundtrip_and_label_check(tmp_path):
    rows = [{"text": "Нет воды", "label": "ЖКХ", "style": "s"}]
    path = write_appeals_csv(rows, tmp_path / "x.csv")
    assert read_appeals_csv(path) == rows
    bad = write_appeals_csv([{"text": "?", "label": "Космос"}], tmp_path / "bad.csv")
    with pytest.raises(ValueError):
        read_appeals_csv(bad)


def test_generation_plan_and_dedupe():
    rng = random.Random(0)
    jobs = plan_jobs("train", per_label=20, styles=["a", "b"], per_request=10, rng=rng)
    assert len(jobs) == len(LABELS) * 3  # 20 × 1.2 / 10 → 3 запроса на категорию
    assert "Тема: «ЖКХ»" in build_prompt("ЖКХ", "стиль", "студент", 5, rng)
    seen: set[str] = set()
    rows = [
        {"text": "Нет воды!", "label": "ЖКХ"},
        {"text": "нет воды", "label": "ЖКХ"},
        {"text": "Яма", "label": "ЖКХ"},
    ]
    assert len(dedupe(rows, seen)) == 2
    assert len(take_per_label(rows * 3, per_label=2, rng=rng)) == 2


def test_classify_all_against_fake_server(fake_openai):
    fake_openai.set_responder(lambda body: {"content": "Категория: ЖКХ"})
    rows = [{"text": "Течёт крыша", "label": "ЖКХ"}, {"text": "Яма", "label": "Дороги и транспорт"}]

    async def run():
        client = make_async_client(fake_openai.url)
        try:
            return await classify_all(client, "fake", rows, json_mode=False, concurrency=2)
        finally:
            await client.close()

    results = asyncio.run(run())
    assert [r["pred"] for r in results] == ["ЖКХ", "ЖКХ"]
    assert (
        classification_metrics([r["label"] for r in results], [r["pred"] for r in results], LABELS)["accuracy"] == 0.5
    )
    assert fake_openai.requests[0]["temperature"] == 0.0


def test_classify_json_mode_sends_schema(fake_openai):
    fake_openai.set_responder(lambda body: {"content": '{"category": "Экология"}'})

    async def run():
        client = make_async_client(fake_openai.url)
        try:
            return await classify_all(
                client, "fake", [{"text": "Дым", "label": "Экология"}], json_mode=True, concurrency=1
            )
        finally:
            await client.close()

    assert asyncio.run(run())[0]["pred"] == "Экология"
    schema = fake_openai.requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["category"]["enum"] == LABELS
