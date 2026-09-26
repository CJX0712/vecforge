"""core 层测试：类型 / 配置 / 错误 / 接口。作者：晨星"""
import os

import pytest

from vecforge.core.config import VecForgeConfig
from vecforge.core.errors import (
    ConfigError,
    DataError,
    EmbeddingError,
    StoreError,
    VecForgeError,
)
from vecforge.core.interfaces import DataSource, Embedder, VectorStore
from vecforge.core.types import ClusterResult, Document, QueryResult, RetrievalHit


def test_document_dataclass_defaults():
    d = Document(doc_id="x", text="hello")
    assert d.topic is None
    assert d.metadata == {}


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ENV_VECFORGE_K", "9")
    monkeypatch.setenv("ENV_VECFORGE_N_TOPICS", "4")
    cfg = VecForgeConfig()
    assert cfg.k == 9
    assert cfg.n_topics == 4


def test_config_defaults_sane():
    cfg = VecForgeConfig()
    assert cfg.n_topics >= 1
    assert cfg.k >= 1
    assert cfg.embedding_dim >= 2
    assert isinstance(cfg.as_dict(), dict)


def test_error_hierarchy_and_code():
    err = StoreError("boom", code="E400")
    assert isinstance(err, VecForgeError)
    assert err.code == "E400"
    assert "E400" in str(err)
    # 子类默认 code 正确
    assert ConfigError().code == "E100"
    assert DataError().code == "E200"
    assert EmbeddingError().code == "E300"
    assert StoreError().code == "E400"


def test_interfaces_are_runtime_checkable():
    class DummySrc:
        def load(self):
            return []

    assert isinstance(DummySrc(), DataSource)
    assert not isinstance(object(), Embedder)
    assert not isinstance(object(), VectorStore)


def test_types_construction():
    hit = RetrievalHit(doc_id="a", score=0.9, rank=1)
    assert hit.rank == 1
    q = QueryResult(query="q", hits=[hit])
    assert len(q.hits) == 1
    c = ClusterResult(labels=[0, 1], n_clusters=2, method="kmeans")
    assert c.n_clusters == 2
