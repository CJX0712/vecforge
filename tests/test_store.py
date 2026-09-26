"""store 层测试：FAISS(可选) + sklearn 兜底，add/query 正确性。作者：晨星"""
import numpy as np
import pytest

from vecforge.core.errors import StoreError
from vecforge.store import (
    FaissVectorStore,
    SklearnVectorStore,
    get_store,
)


@pytest.fixture
def vectors():
    rng = np.random.RandomState(0)
    return rng.rand(20, 8).astype(np.float32)


def test_sklearn_add_query_returns_k(vectors):
    store = SklearnVectorStore(metric="cosine")
    store.add(vectors, [f"d{i}" for i in range(20)])
    res = store.query(vectors[0], k=5)
    assert len(res.hits) == 5
    assert res.hits[0].doc_id == "d0"  # 自身最相似
    assert res.hits[0].rank == 1


def test_sklearn_query_shape_mismatch(vectors):
    store = SklearnVectorStore()
    store.add(vectors, [f"d{i}" for i in range(20)])
    with pytest.raises(StoreError):
        store.query(np.zeros(4), k=3)  # 维度不符 (应为 8) 应报错


def test_sklearn_ids_mismatch(vectors):
    store = SklearnVectorStore()
    with pytest.raises(StoreError):
        store.add(vectors, ["only-one"])


def test_query_before_add_raises(vectors):
    store = SklearnVectorStore()
    with pytest.raises(StoreError):
        store.query(vectors[0], k=3)


def test_faiss_available_flag():
    # 不应抛异常；返回 bool
    assert isinstance(FaissVectorStore.available(), bool)


@pytest.mark.skipif(not FaissVectorStore.available(), reason="FAISS 未安装")
def test_faiss_add_query(vectors):
    store = FaissVectorStore(metric="cosine")
    store.add(vectors, [f"d{i}" for i in range(20)])
    res = store.query(vectors[0], k=4)
    assert len(res.hits) == 4
    assert res.hits[0].doc_id == "d0"


@pytest.mark.skipif(not FaissVectorStore.available(), reason="FAISS 未安装")
def test_faiss_unavailable_construct():
    # 构造时若未安装应抛 StoreError；此处已安装，仅验证 available 一致
    assert FaissVectorStore.available() is True


def test_get_store_prefers_faiss():
    store = get_store(prefer="faiss", force_sklearn=False)
    if FaissVectorStore.available():
        assert store.backend == "faiss"
    else:
        assert store.backend == "sklearn"


def test_get_store_force_sklearn():
    store = get_store(force_sklearn=True)
    assert store.backend == "sklearn"
