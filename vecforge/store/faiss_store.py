"""FAISS 向量库（可选后端）。缺失时 available() 返回 False，流水线自动降级。作者：晨星"""
from __future__ import annotations

import numpy as np

from ..core.errors import StoreError
from ..core.types import QueryResult, RetrievalHit

try:  # FAISS 为可选依赖，缺失不致命
    import faiss

    HAVE_FAISS = True
except Exception:  # pragma: no cover - 仅在未安装时触发
    faiss = None
    HAVE_FAISS = False


class FaissVectorStore:
    """基于 faiss.IndexFlat 的精确向量检索；cosine 走内积（先 L2 归一化）。"""

    backend = "faiss"

    def __init__(self, metric: str = "cosine"):
        if not HAVE_FAISS:
            raise StoreError(
                "FAISS 未安装，请改用 SklearnVectorStore 或 pip install faiss-cpu",
                code="E400",
            )
        if metric not in ("cosine", "l2"):
            raise StoreError(f"不支持的度量: {metric}", code="E400")
        self.metric = metric
        self.index = None
        self.ids: list[str] = []
        self._dim: int | None = None

    @staticmethod
    def available() -> bool:
        return HAVE_FAISS

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        X = np.asarray(vectors, dtype=np.float32)
        if X.ndim != 2:
            raise StoreError("向量必须是二维 (n, d)", code="E400")
        if len(ids) != X.shape[0]:
            raise StoreError("ids 数量与向量行数不一致", code="E400")
        self._dim = X.shape[1]
        if self.metric == "cosine":
            X = self._normalize(X)
            self.index = faiss.IndexFlatIP(self._dim)
        else:
            self.index = faiss.IndexFlatL2(self._dim)
        self.index.add(X)
        self.ids = list(ids)

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        if self.index is None:
            raise StoreError("尚未 add 向量", code="E400")
        q = np.asarray(vector, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self._dim:
            raise StoreError(
                f"查询向量维度 {q.shape} 与索引维度 {self._dim} 不一致", code="E400"
            )
        q = q.reshape(1, -1)
        if self.metric == "cosine":
            q = self._normalize(q)
        k = min(k, len(self.ids))
        scores, idx = self.index.search(q, k)
        hits: list[RetrievalHit] = []
        for rank, (i, s) in enumerate(zip(idx[0], scores[0]), start=1):
            if i < 0:
                continue
            hits.append(RetrievalHit(doc_id=self.ids[i], score=float(s), rank=rank))
        return QueryResult(query="<vector>", hits=hits)
