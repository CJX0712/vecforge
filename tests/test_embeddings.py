"""embeddings 层测试：TF-IDF / LSA / NMF 形状与确定性。作者：晨星"""
import numpy as np
import pytest

from vecforge.core.errors import EmbeddingError
from vecforge.embeddings import LsaEmbedder, NmfEmbedder, TfidfEmbedder, get_embedder


def test_tfidf_shape_and_nonempty(doc_texts, config):
    emb = TfidfEmbedder(config).fit(doc_texts)
    X = emb.transform(doc_texts)
    assert X.shape[0] == len(doc_texts)
    assert X.shape[1] >= 1
    assert np.isfinite(X).all()


def test_lsa_dim(doc_texts, config):
    emb = LsaEmbedder(config)
    X = emb.fit_transform(doc_texts)
    assert X.shape == (len(doc_texts), config.embedding_dim)
    # L2 归一化：每行模长≈1
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_nmf_dim(doc_texts, config):
    emb = NmfEmbedder(config)
    X = emb.fit_transform(doc_texts)
    assert X.shape == (len(doc_texts), config.embedding_dim)
    assert (X >= 0).all()  # NMF 非负


def test_transform_before_fit_raises(doc_texts, config):
    with pytest.raises(EmbeddingError):
        TfidfEmbedder(config).transform(doc_texts)


def test_empty_corpus_raises(config):
    with pytest.raises(EmbeddingError):
        TfidfEmbedder(config).fit([])


def test_determinism(doc_texts, config):
    a = TfidfEmbedder(config).fit(doc_texts).transform(doc_texts)
    b = TfidfEmbedder(config).fit(doc_texts).transform(doc_texts)
    assert np.allclose(a, b)


def test_get_embedder_unknown(config):
    with pytest.raises(KeyError):
        get_embedder("nope", config)
