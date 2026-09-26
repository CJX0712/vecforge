"""VecForge core 包：类型 / 配置 / 错误 / 接口契约。作者：晨星"""
from .config import VecForgeConfig
from .errors import (
    ConfigError,
    DataError,
    EmbeddingError,
    EvalError,
    StoreError,
    VecForgeError,
)
from .interfaces import DataSource, Embedder, Metric, VectorStore
from .types import ClusterResult, Document, EvalReport, QueryResult, RetrievalHit

__all__ = [
    "VecForgeConfig",
    "VecForgeError",
    "ConfigError",
    "DataError",
    "EmbeddingError",
    "StoreError",
    "EvalError",
    "DataSource",
    "Embedder",
    "VectorStore",
    "Metric",
    "Document",
    "QueryResult",
    "RetrievalHit",
    "ClusterResult",
    "EvalReport",
]
