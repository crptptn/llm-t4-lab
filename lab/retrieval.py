"""Поиск для RAG: BM25 с русским стеммером, плотный поиск (multilingual-e5 + FAISS),
слияние рангов (RRF) и переранжирование cross-encoder'ом."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1)
def _stemmer():
    import snowballstemmer

    return snowballstemmer.stemmer("russian")


def tokenize_ru(text: str) -> list[str]:
    """Нижний регистр, ё→е, слова длиннее одной буквы (числа — любые), стемминг Snowball."""
    words = [w for w in _WORD.findall(text.lower().replace("ё", "е")) if len(w) > 1 or w.isdigit()]
    return _stemmer().stemWords(words)


class BM25:
    """Okapi BM25 на разреженной матрице.

    Веса термов считаются один раз при индексации, поэтому поиск — это сумма столбцов матрицы:
        score(d, q) = Σ_t∈q idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
        idf(t) = ln((N − df + 0.5) / (df + 0.5) + 1)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: Callable[[str], list[str]] = tokenize_ru):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer

    def fit(self, docs: Sequence[str]) -> BM25:
        from scipy import sparse
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(analyzer=self.tokenizer)
        tf = vectorizer.fit_transform(docs).tocoo()
        self.n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean()) if self.n_docs and doc_len.mean() > 0 else 1.0
        df = np.bincount(tf.col, minlength=tf.shape[1])
        self.idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        norm = self.k1 * (1 - self.b + self.b * doc_len[tf.row] / avgdl)
        weights = self.idf[tf.col] * tf.data * (self.k1 + 1) / (tf.data + norm)
        self.weights = sparse.csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)
        self.vocabulary = vectorizer.vocabulary_
        return self

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocabulary[t] for t in self.tokenizer(query) if t in self.vocabulary]
        if not cols:
            return np.zeros(self.n_docs)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()

    def search(self, query: str, k: int) -> list[int]:
        scores = self.scores(query)
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top], kind="stable")].tolist()


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60, limit: int | None = None) -> list[int]:
    """RRF: score(d) = Σ 1/(k + rank). Не требует нормировать оценки разных поисковиков."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores, key=lambda d: (-scores[d], d))
    return fused[:limit] if limit else fused


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class DenseIndex:
    """multilingual-e5: у запросов префикс «query: », у документов «passage: » — так модель обучалась."""

    def __init__(
        self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None, batch_size: int = 64
    ):
        from sentence_transformers import SentenceTransformer

        self.device = _device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device.startswith("cuda"):
            self.model.half()
        self.batch_size = batch_size

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.ascontiguousarray(emb, dtype="float32")

    def fit(self, passages: Sequence[str]) -> DenseIndex:
        import faiss

        emb = self._encode([f"passage: {p}" for p in passages])
        # векторы нормированы, поэтому скалярное произведение = косинусная близость
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        return self

    def search(self, queries: Sequence[str], k: int) -> list[list[int]]:
        _, ids = self.index.search(self._encode([f"query: {q}" for q in queries]), k)
        return [[i for i in row if i >= 0] for row in ids.tolist()]


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None, batch_size: int = 32):
        from sentence_transformers import CrossEncoder

        self.device = _device(device)
        self.model = CrossEncoder(model_name, device=self.device)
        if self.device.startswith("cuda") and hasattr(self.model, "half"):
            self.model.half()
        self.batch_size = batch_size

    def rerank_many(
        self, queries: Sequence[str], candidates: Sequence[Sequence[int]], docs: Sequence[str]
    ) -> list[list[int]]:
        """Все пары (запрос, кандидат) одним прогоном — так GPU загружен лучше, чем по запросу."""
        pairs = [(q, docs[d]) for q, cand in zip(queries, candidates) for d in cand]
        scores = np.asarray(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=len(pairs) > 2000))
        out, offset = [], 0
        for cand in candidates:
            chunk = scores[offset : offset + len(cand)]
            offset += len(cand)
            out.append([cand[i] for i in np.argsort(-chunk, kind="stable")])
        return out
