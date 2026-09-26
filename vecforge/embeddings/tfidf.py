"""TF-IDF 嵌入器（scikit-learn）。强制兜底路径，零额外依赖。作者：晨星"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..core.config import VecForgeConfig
from ..core.errors import EmbeddingError
from .base import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    """经典词频-逆文档频嵌入。高维稀疏，主题词区分度强。"""

    name = "tfidf"

    def __init__(self, config: VecForgeConfig | None = None):
        self.config = config or VecForgeConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self._fitted = False

    def fit(self, corpus):
        if not corpus:
            raise EmbeddingError("空语料无法拟合 TF-IDF", code="E300")
        self.vectorizer.fit(corpus)
        self._fitted = True
        return self

    def transform(self, docs):
        if not self._fitted:
            raise EmbeddingError("尚未 fit，请先调用 fit 或 fit_transform", code="E300")
        X = self.vectorizer.transform(docs)
        # 转稠密便于后续向量库统一处理；维度通常可控
        return np.asarray(X.todense(), dtype=np.float32)

    @property
    def dim(self) -> int:
        return len(self.vectorizer.vocabulary_)
