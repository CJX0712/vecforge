"""降维嵌入：LSA (TruncatedSVD) 与 NMF，均基于 TF-IDF 之上。作者：晨星"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from ..core.config import VecForgeConfig
from ..core.errors import EmbeddingError
from .base import BaseEmbedder


class LsaEmbedder(BaseEmbedder):
    """潜在语义分析：TF-IDF 后接 TruncatedSVD 降维到稠密低维空间。"""

    name = "lsa"

    def __init__(self, config: VecForgeConfig | None = None):
        self.config = config or VecForgeConfig()
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.config.max_features,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
                ("svd", TruncatedSVD(n_components=self.config.embedding_dim)),
            ]
        )
        self._fitted = False

    def fit(self, corpus):
        if not corpus:
            raise EmbeddingError("空语料无法拟合 LSA", code="E300")
        self.pipeline.fit(corpus)
        self._fitted = True
        return self

    def transform(self, docs):
        if not self._fitted:
            raise EmbeddingError("尚未 fit", code="E300")
        X = self.pipeline.transform(docs)
        return self._l2_normalize(np.asarray(X, dtype=np.float32))

    @property
    def dim(self) -> int:
        return self.config.embedding_dim


class NmfEmbedder(BaseEmbedder):
    """非负矩阵分解：得到基于「主题」的加性表示，可解释性强。"""

    name = "nmf"

    def __init__(self, config: VecForgeConfig | None = None):
        self.config = config or VecForgeConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self.nmf = NMF(
            n_components=self.config.embedding_dim,
            init="nndsvda",
            max_iter=200,
            random_state=self.config.seed,
        )
        self._fitted = False

    def fit(self, corpus):
        if not corpus:
            raise EmbeddingError("空语料无法拟合 NMF", code="E300")
        self.vectorizer.fit(corpus)
        tfidf = self.vectorizer.transform(corpus)
        self.nmf.fit(tfidf)
        self._fitted = True
        return self

    def transform(self, docs):
        if not self._fitted:
            raise EmbeddingError("尚未 fit", code="E300")
        tfidf = self.vectorizer.transform(docs)
        X = self.nmf.transform(tfidf)
        return self._l2_normalize(np.asarray(X, dtype=np.float32))

    @property
    def dim(self) -> int:
        return self.config.embedding_dim
