"""VecForge 核心数据类型定义。

所有跨模块流转的结构化对象集中在此，避免循环依赖。
作者：晨星
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """单篇文档。

    doc_id 必须全局唯一；topic 为可选的主题标签（用于聚类/检索评测的真值）。
    """

    doc_id: str
    text: str
    topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    """一次检索返回的命中项。"""

    doc_id: str
    score: float
    rank: int


@dataclass
class QueryResult:
    """一次查询的完整结果。"""

    query: str
    hits: List[RetrievalHit] = field(default_factory=list)


@dataclass
class ClusterResult:
    """聚类结果。labels[i] 为第 i 篇文档的簇编号。"""

    labels: List[int]
    n_clusters: int
    method: str


@dataclass
class EvalReport:
    """评测报告：检索指标 + 聚类指标。"""

    n_docs: int
    retrieval: Dict[str, float] = field(default_factory=dict)
    clustering: Dict[str, float] = field(default_factory=dict)
    backend: str = "unknown"
    embedding: str = "unknown"
