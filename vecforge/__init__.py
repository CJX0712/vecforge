"""VecForge —— 文档向量检索与聚类系统。

整合 FAISS（可选）与 scikit-learn（强制兜底），提供 TF-IDF/LSA/NMF 嵌入、
精确向量检索、检索(recall/MRR)与聚类(NMI/ARI)评测，及一键可复现流水线。
作者：晨星
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "晨星"

from .core import (
    ClusterResult,
    Document,
    EvalReport,
    QueryResult,
    RetrievalHit,
    VecForgeConfig,
)
from .data import CsvDataSource, FileDataSource, SyntheticDataSource
from .embeddings import LsaEmbedder, NmfEmbedder, TfidfEmbedder
from .pipeline import VecForgePipeline
from .store import FaissVectorStore, SklearnVectorStore, get_store

__all__ = [
    "__version__",
    "__author__",
    "VecForgeConfig",
    "Document",
    "QueryResult",
    "RetrievalHit",
    "ClusterResult",
    "EvalReport",
    "SyntheticDataSource",
    "FileDataSource",
    "CsvDataSource",
    "TfidfEmbedder",
    "LsaEmbedder",
    "NmfEmbedder",
    "FaissVectorStore",
    "SklearnVectorStore",
    "get_store",
    "VecForgePipeline",
]
