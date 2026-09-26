"""VecForge 接口契约（Protocol）。

用 ``typing.Protocol`` + ``runtime_checkable`` 解耦各模块，任何实现只要
方法签名一致即可热插拔；流水线只依赖契约而非具体类。
作者：晨星
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np

from .types import Document, QueryResult


@runtime_checkable
class DataSource(Protocol):
    """数据来源：产出文档列表。"""

    def load(self) -> List[Document]: ...


@runtime_checkable
class Embedder(Protocol):
    """文本 -> 向量。"""

    name: str

    def fit(self, corpus: List[str]) -> "Embedder": ...

    def transform(self, docs: List[str]) -> np.ndarray: ...

    def fit_transform(self, docs: List[str]) -> np.ndarray: ...


@runtime_checkable
class VectorStore(Protocol):
    """向量库的契约。"""

    backend: str

    @staticmethod
    def available() -> bool: ...

    def add(self, vectors: np.ndarray, ids: List[str]) -> None: ...

    def query(self, vector: np.ndarray, k: int) -> QueryResult: ...


@runtime_checkable
class Metric(Protocol):
    """评测指标契约。"""

    name: str

    def evaluate(self, *args, **kwargs) -> float: ...
