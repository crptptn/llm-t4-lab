import math

import pytest

from lab.retrieval import BM25, reciprocal_rank_fusion, tokenize_ru

DOCS = [
    "Кошка сидит на крыше дома",
    "Собака бежит по дороге к реке",
    "Кошка и собака живут дружно",
    "Ремонт дорог в городе начнётся весной",
]


def test_tokenizer_stems_and_drops_single_letters():
    tokens = tokenize_ru("Дороги и дорога, ёлки")
    assert tokens[0] == tokens[1]  # одна основа для разных форм
    assert "и" not in tokens
    assert all("ё" not in t for t in tokens)


def test_bm25_ranks_relevant_documents_first():
    bm25 = BM25().fit(DOCS)
    assert set(bm25.search("кошка", 2)) == {0, 2}
    assert bm25.search("дороги", 2) == [1, 3] or bm25.search("дороги", 2) == [3, 1]
    assert bm25.search("ремонт дорог весной", 1) == [3]


def test_bm25_score_matches_formula():
    docs = ["яблоко груша", "яблоко", "слива"]
    bm25 = BM25(k1=1.5, b=0.75, tokenizer=str.split).fit(docs)
    n, df, avgdl = 3, 2, (2 + 1 + 1) / 3
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    expected_doc0 = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / avgdl))
    assert bm25.scores("яблоко")[0] == pytest.approx(expected_doc0)
    assert bm25.scores("слива")[0] == 0.0


def test_bm25_unknown_query_returns_zeros():
    bm25 = BM25().fit(DOCS)
    assert bm25.scores("абракадабра").sum() == 0.0
    assert len(bm25.search("абракадабра", 10)) == len(DOCS)


def test_rrf_combines_rankings():
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60) == [1, 3, 2, 4]
    assert reciprocal_rank_fusion([[1, 2], [2, 1]], k=60, limit=1) == [1]  # ничья — по id
