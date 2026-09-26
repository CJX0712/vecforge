"""scikit-learn 向量库（强制兜底）。NearestNeighbors 暴力精确检索。作者：晨星"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..core.errors import StoreError
from ..core.types import QueryResult, RetrievalHit


class SklearnVectorStore:
    """基于 sklearn NearestNeighbors 的精确 KNN 检索。中小规模等价于 FAISS。"""

    backend = "sklearn"

    def __init__(self, metric: str = "cosine"):
        if metric not in ("cosine", "euclidean", "l2"):
            raise StoreError(f"不支持的度量: {metric}", code="E400")
        # NearestNeighbors 接受 "cosine" / "euclidean"
        self.metric = "euclidean" if metric in ("l2",) else metric
        self._metric_name = metric
        self.nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        self.ids: list[str] = []
        self._fitted = False

    @staticmethod
    def available() -> bool:
        return True

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2:
            raise StoreError("向量必须是二维 (n, d)", code="E400")
        if len(ids) != X.shape[0]:
            raise StoreError("ids 数量与向量行数不一致", code="E400")
        n_neigh = max(1, min(len(ids), 1))
        self.nn = NearestNeighbors(n_neighbors=n_neigh, metric=self.metric)
        self.nn.fit(X)
        self.ids = list(ids)
        self._dim = X.shape[1]
        self._fitted = True

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        if not self._fitted:
            raise StoreError("尚未 add 向量", code="E400")
        q = np.asarray(vector, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self._dim:
            raise StoreError(
                f"查询向量维度 {q.shape} 与索引维度 {self._dim} 不一致", code="E400"
            )
        q = q.reshape(1, -1)
        k = min(k, len(self.ids))
        # 重新构建以返回 k 个近邻（n_neighbors 在 add 时未定）
        self.nn.set_params(n_neighbors=k)
        dist, idx = self.nn.kneighbors(q)
        hits: list[RetrievalHit] = []
        for rank, (i, d) in enumerate(zip(idx[0], dist[0]), start=1):
            # 将距离转为相似度：cosine 距离 -> 相似度=1-d；euclidean -> 1/(1+d)
            if self._metric_name == "cosine":
                score = float(1.0 - d)
            else:
                score = float(1.0 / (1.0 + d))
            hits.append(RetrievalHit(doc_id=self.ids[i], score=score, rank=rank))
        return QueryResult(query="<vector>", hits=hits)
