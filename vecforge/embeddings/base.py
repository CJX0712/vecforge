"""嵌入层基类。统一接口与 L2 归一化工具。作者：晨星"""
from __future__ import annotations

import numpy as np


class BaseEmbedder:
    """所有嵌入器的基类，实现 Embedder 契约的公共部分。"""

    name = "base"

    def fit(self, corpus):
        raise NotImplementedError

    def transform(self, docs):
        raise NotImplementedError

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """按行 L2 归一化，便于余弦相似度用内积表达。"""
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
