"""Метрики и разбор ответов моделей. Без GPU-зависимостей, всё покрыто тестами."""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Iterable, Sequence

import numpy as np

INVALID = "__invalid__"

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def summarize(values: Iterable[float | None]) -> dict | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "n": len(vals),
    }


# --- ранжирование -----------------------------------------------------------


def recall_at_k(ranked: Sequence[Sequence[Hashable]], relevant: Sequence[set], k: int) -> float:
    """Средняя по запросам доля релевантных документов, попавших в top-k."""
    if len(ranked) != len(relevant):
        raise ValueError("ranked и relevant разной длины")
    scores = [len(rel.intersection(docs[:k])) / len(rel) for docs, rel in zip(ranked, relevant) if rel]
    return float(np.mean(scores)) if scores else 0.0


def mrr_at_k(ranked: Sequence[Sequence[Hashable]], relevant: Sequence[set], k: int) -> float:
    """Mean Reciprocal Rank: 1/позиция первого релевантного документа в top-k (0, если его нет)."""
    if len(ranked) != len(relevant):
        raise ValueError("ranked и relevant разной длины")
    scores = []
    for docs, rel in zip(ranked, relevant):
        if not rel:
            continue
        rr = 0.0
        for pos, doc in enumerate(docs[:k], start=1):
            if doc in rel:
                rr = 1.0 / pos
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


# --- GSM8K ------------------------------------------------------------------


def parse_number(raw: str) -> float | None:
    cleaned = raw.replace(",", "").replace(" ", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_final_number(text: str | None) -> float | None:
    """Число после последнего '####', иначе последнее число в тексте."""
    if not text:
        return None
    marker = text.rfind("####")
    if marker != -1:
        found = _NUMBER.search(text[marker:])
        if found:
            return parse_number(found.group())
    numbers = _NUMBER.findall(text)
    return parse_number(numbers[-1]) if numbers else None


def gsm8k_gold(answer: str) -> float:
    marker = answer.rfind("####")
    value = extract_final_number(answer[marker:] if marker != -1 else answer)
    if value is None:
        raise ValueError(f"в эталонном ответе нет числа: {answer[-80:]!r}")
    return value


def numbers_match(pred: float | None, gold: float | None, tol: float = 1e-6) -> bool:
    return pred is not None and gold is not None and math.isclose(pred, gold, rel_tol=tol, abs_tol=tol)


# --- классификация ----------------------------------------------------------


def parse_label(text: str | None, labels: Sequence[str]) -> str | None:
    """Находит категорию в ответе модели: точное совпадение или первое упоминание целым словом."""
    if not text:
        return None
    norm = normalize_text(text)
    by_norm = {normalize_text(label): label for label in labels}
    if norm in by_norm:
        return by_norm[norm]
    best: tuple[int, int, str] | None = None
    for key, label in by_norm.items():
        match = re.search(rf"(?<!\w){re.escape(key)}(?!\w)", norm)
        if match is None:
            continue
        # раньше в тексте — лучше; при равной позиции — более длинное название
        candidate = (match.start(), -len(key), label)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best else None


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str]) -> dict:
    """accuracy, macro-F1, F1 по классам и матрица ошибок. Нераспознанный ответ = ошибка."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    if not y_true:
        raise ValueError("пустая выборка")
    labels = list(labels)
    pred = [p if p is not None else INVALID for p in y_pred]
    per_class = f1_score(y_true, pred, labels=labels, average=None, zero_division=0)
    matrix_labels = [*labels, INVALID]
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "invalid_rate": sum(p is None for p in y_pred) / len(y_pred),
        "per_class_f1": {label: float(score) for label, score in zip(labels, per_class)},
        "confusion": {
            "labels": matrix_labels,
            "matrix": confusion_matrix(y_true, pred, labels=matrix_labels).tolist(),
        },
    }


# --- RAG --------------------------------------------------------------------


def answer_contains(answer: str | None, golds: Iterable[str]) -> bool:
    """Есть ли эталонный ответ (после нормализации) внутри ответа модели."""
    if not answer:
        return False
    norm = normalize_text(answer)
    return any(normalize_text(g) and normalize_text(g) in norm for g in golds)
