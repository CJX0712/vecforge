"""eval 层测试：检索指标 + 聚类指标。作者：晨星"""
import numpy as np
import pytest

from vecforge.core.errors import EvalError
from vecforge.eval.metrics import (
    clustering_scores,
    mean_reciprocal_rank,
    precision_at_k,
    random_baseline_retrieval,
    recall_at_k,
)
from vecforge.eval.benchmark import Benchmark


def test_recall_at_k_perfect():
    relevant = {"a", "b", "c"}
    retrieved = ["a", "b", "c", "d", "e"]
    assert recall_at_k(retrieved, relevant, 5) == 1.0


def test_recall_at_k_partial():
    relevant = {"a", "b", "c", "d"}
    retrieved = ["a", "x", "y"]
    assert recall_at_k(retrieved, relevant, 3) == 0.25


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a"], set(), 3) == 0.0


def test_precision_at_k():
    relevant = {"a", "b"}
    assert precision_at_k(["a", "x"], relevant, 2) == 0.5


def test_mrr():
    ranked = [["x"], [], ["y"]]  # 第一查询命中 rank1, 第二未命中, 第三命中
    assert mean_reciprocal_rank(ranked) == (1.0 + 0.0 + 1.0) / 3


def test_clustering_perfect():
    true = [0, 0, 1, 1, 2, 2]
    pred = [0, 0, 1, 1, 2, 2]
    s = clustering_scores(true, pred)
    assert s["nmi"] == 1.0
    assert s["ari"] == 1.0


def test_clustering_random_low():
    true = [0, 0, 0, 1, 1, 1]
    pred = [0, 1, 0, 1, 0, 1]
    s = clustering_scores(true, pred)
    assert s["nmi"] < 1.0


def test_random_baseline_monotonic():
    assert random_baseline_retrieval(100, 5, 10) > random_baseline_retrieval(100, 1, 10)


def test_benchmark_retrieval(documents, embeddings_matrix, sklearn_store):
    sklearn_store.add(embeddings_matrix, [d.doc_id for d in documents])
    bench = Benchmark(documents, embeddings_matrix, sklearn_store, k=5)
    rep = bench.evaluate_retrieval()
    assert rep["recall@k"] > rep["random_recall@k"]
    assert 0.0 <= rep["mrr"] <= 1.0


def test_benchmark_clustering(documents, embeddings_matrix, sklearn_store):
    sklearn_store.add(embeddings_matrix, [d.doc_id for d in documents])
    bench = Benchmark(documents, embeddings_matrix, sklearn_store, k=5)
    rep = bench.evaluate_clustering()
    assert rep["n_clusters_true"] == len({d.topic for d in documents})
    assert 0.0 <= rep["nmi"] <= 1.0


def test_benchmark_clustering_requires_topic(documents, embeddings_matrix, sklearn_store):
    for d in documents:
        d.topic = None
    sklearn_store.add(embeddings_matrix, [d.doc_id for d in documents])
    bench = Benchmark(documents, embeddings_matrix, sklearn_store, k=5)
    with pytest.raises(EvalError):
        bench.evaluate_clustering()
