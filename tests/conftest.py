"""pytest 共享夹具。作者：晨星"""
import numpy as np
import pytest

from vecforge.core.config import VecForgeConfig
from vecforge.data.synthetic import SyntheticDataSource
from vecforge.embeddings import get_embedder
from vecforge.store import FaissVectorStore, SklearnVectorStore


@pytest.fixture
def config() -> VecForgeConfig:
    return VecForgeConfig()


@pytest.fixture
def documents():
    return SyntheticDataSource(VecForgeConfig()).load()


@pytest.fixture
def doc_texts(documents):
    return [d.text for d in documents]


@pytest.fixture(params=["tfidf", "lsa", "nmf"])
def embedder(request, config):
    return get_embedder(request.param, config)


@pytest.fixture
def embeddings_matrix(doc_texts, embedder):
    return embedder.fit_transform(doc_texts)


@pytest.fixture
def sklearn_store():
    return SklearnVectorStore(metric="cosine")


@pytest.fixture
def faiss_store():
    if not FaissVectorStore.available():
        pytest.skip("FAISS 未安装")
    return FaissVectorStore(metric="cosine")
