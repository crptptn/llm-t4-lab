"""Категории, промпт классификатора и подготовка примеров для обучения.

Один и тот же промпт используется при обучении и при оценке через vLLM —
иначе адаптер учится на одном формате, а проверяется на другом.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

LABELS = [
    "ЖКХ",
    "Дороги и транспорт",
    "Благоустройство",
    "Здравоохранение",
    "Образование",
    "Социальная поддержка",
    "Экология",
    "Цифровые госуслуги",
]

LABEL_HINTS = {
    "ЖКХ": "отопление, вода, канализация, лифты, протечки, управляющая компания, счета за коммуналку",
    "Дороги и транспорт": "ямы, светофоры, разметка, остановки, маршруты автобусов, парковки",
    "Благоустройство": "дворы, детские площадки, освещение улиц, лавочки, озеленение, уборка снега во дворах",
    "Здравоохранение": "поликлиники, запись к врачу, скорая помощь, лекарства, больницы",
    "Образование": "школы, детские сады, очереди в сад, кружки, питание в школе, учителя",
    "Социальная поддержка": "пособия, выплаты, льготы, пенсии, помощь многодетным и инвалидам, соцзащита",
    "Экология": "свалки, выбросы предприятий, загрязнение рек, вырубка деревьев, запахи",
    "Цифровые госуслуги": "портал Госуслуг, электронная запись, ошибки сайтов, подтверждение учётной записи",
}

SYSTEM_PROMPT = (
    "Ты классифицируешь обращения граждан в органы власти. "
    "Выбери одну категорию из списка и ответь только её названием, без пояснений."
)
SYSTEM_PROMPT_JSON = (
    "Ты классифицируешь обращения граждан в органы власти. "
    "Выбери одну категорию из списка и верни JSON с полем category."
)


def build_messages(text: str, labels: Sequence[str] = LABELS, json_mode: bool = False) -> list[dict]:
    categories = "\n".join(f"- {label}" for label in labels)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_JSON if json_mode else SYSTEM_PROMPT},
        {"role": "user", "content": f"Категории:\n{categories}\n\nОбращение:\n{text.strip()}"},
    ]


def category_schema(labels: Sequence[str] = LABELS) -> dict:
    return {
        "type": "object",
        "properties": {"category": {"type": "string", "enum": list(labels)}},
        "required": ["category"],
        "additionalProperties": False,
    }


def build_training_example(tokenizer, text: str, label: str, max_text_tokens: int = 320) -> dict:
    """input_ids = промпт + ответ; labels = -100 на промпте, токены ответа и EOS — на ответе.

    Loss считается только по названию категории: модель не тратит ёмкость на запоминание промпта.
    """
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(text_ids) > max_text_tokens:
        text = tokenizer.decode(text_ids[:max_text_tokens])
    prompt = tokenizer.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(label, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}


def pad_batch(batch: Sequence[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Паддинг справа: input_ids — pad_id, labels — -100, attention_mask — 0."""
    width = max(len(x["input_ids"]) for x in batch)
    out: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for x in batch:
        gap = width - len(x["input_ids"])
        out["input_ids"].append(x["input_ids"] + [pad_id] * gap)
        out["labels"].append(x["labels"] + [-100] * gap)
        out["attention_mask"].append([1] * len(x["input_ids"]) + [0] * gap)
    return out


def read_appeals_csv(path: str | Path) -> list[dict]:
    """CSV с колонками text и label. Подойдёт и свой датасет вместо синтетики."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("text") and row.get("label")]
    unknown = sorted({r["label"] for r in rows} - set(LABELS))
    if unknown:
        raise ValueError(f"в {path} есть метки не из LABELS: {unknown}")
    return rows


def write_appeals_csv(rows: Sequence[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "style"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
