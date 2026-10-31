import math

import pytest

from lab.metrics import (
    INVALID,
    answer_contains,
    classification_metrics,
    extract_final_number,
    gsm8k_gold,
    mrr_at_k,
    normalize_text,
    numbers_match,
    parse_label,
    recall_at_k,
    summarize,
)

LABELS = ["ЖКХ", "Дороги и транспорт", "Образование"]


def test_recall_and_mrr_single_relevant():
    ranked = [[3, 1, 2], [5, 6, 7], [9, 8]]
    relevant = [{1}, {7}, {4}]
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert mrr_at_k(ranked, relevant, 10) == pytest.approx((1 / 2 + 1 / 3 + 0) / 3)


def test_recall_counts_fraction_of_several_relevant():
    assert recall_at_k([[1, 2, 3]], [{1, 9}], 3) == 0.5


def test_ranking_metrics_reject_length_mismatch():
    with pytest.raises(ValueError):
        recall_at_k([[1]], [], 1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Step 1... so she has 18 eggs.\n#### 18", 18.0),
        ("The answer is $1,250.50.", 1250.5),
        ("#### 42\nThat's it, 7 steps", 42.0),
        ("Ответ: -3", -3.0),
        ("no digits here", None),
        ("", None),
    ],
)
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


def test_gsm8k_gold_and_matching():
    assert gsm8k_gold("Natalia sold 48/2 = 24 clips.\n#### 72") == 72.0
    assert numbers_match(72.0, 72.0000000001)
    assert not numbers_match(None, 72.0)
    assert not numbers_match(71.0, 72.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ЖКХ", "ЖКХ"),
        ("Категория: Дороги и транспорт.", "Дороги и транспорт"),
        ("  образование ", "Образование"),
        ("Это про ЖКХ, а не про образование", "ЖКХ"),
        ("Образованием займётся школа", None),  # не путаем словоформу с названием категории
        ("не знаю", None),
        (None, None),
    ],
)
def test_parse_label(text, expected):
    assert parse_label(text, LABELS) == expected


def test_classification_metrics_counts_invalid_as_error():
    y_true = ["ЖКХ", "ЖКХ", "Образование", "Дороги и транспорт"]
    y_pred = ["ЖКХ", None, "Образование", "ЖКХ"]
    m = classification_metrics(y_true, y_pred, LABELS)
    assert m["accuracy"] == 0.5
    assert m["invalid_rate"] == 0.25
    assert m["per_class_f1"]["Образование"] == 1.0
    assert m["per_class_f1"]["Дороги и транспорт"] == 0.0
    assert m["confusion"]["labels"][-1] == INVALID
    assert sum(map(sum, m["confusion"]["matrix"])) == 4


def test_normalize_and_answer_contains():
    assert normalize_text("Ёлка, «зелёная»!") == "елка зеленая"
    assert answer_contains("Это произошло в 1812 году.", ["1812 году"])
    assert not answer_contains("Не знаю", ["1812"])
    assert not answer_contains("", ["1812"])


def test_summarize_ignores_none():
    s = summarize([1.0, None, 3.0])
    assert s["n"] == 2 and s["mean"] == 2.0 and s["p50"] == 2.0
    assert summarize([None]) is None
    assert not math.isnan(summarize([5.0])["p99"])
