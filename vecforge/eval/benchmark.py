"""端到端评测：检索 + 聚类。作者：晨星"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from ..core.errors import EvalError
from ..core.types import Document
from .metrics import (
    clustering_scores,
    mean_reciprocal_rank,
    precision_at_k,
    random_baseline_retrieval,
    recall_at_k,
)


class Benchmark:
    """给定文档集合、嵌入矩阵与向量库，产出可量化评测报告。"""

    def __init__(self, documents: List[Document], vectors: np.ndarray, store, k: int = 5):
        if len(documents) != vectors.shape[0]:
            raise EvalError("文档数与向量行数不一致", code="E500")
        self.documents = documents
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.store = store
        self.k = k
        self.id_to_topic = {d.doc_id: d.topic for d in documents}
        self.topic_to_ids: Dict[str, List[str]] = defaultdict(list)
        for d in documents:
            if d.topic is not None:
                self.topic_to_ids[d.topic].append(d.doc_id)

    def evaluate_retrieval(self) -> Dict[str, float]:
        """逐文档以自身为查询，同主题为相关集，统计 recall@k / precision@k / MRR。"""
        ranked_relevant: List[List[str]] = []
        recalls: List[float] = []
        precisions: List[float] = []
        for d in self.documents:
            res = self.store.query(self.vectors[self.documents.index(d)], self.k)
            retrieved = [h.doc_id for h in res.hits]
            relevant = set(self.topic_to_ids.get(d.topic, []))
            relevant.discard(d.doc_id)  # 自身不算相关
            recalls.append(recall_at_k(retrieved, relevant, self.k))
            precisions.append(precision_at_k(retrieved, relevant, self.k))
            # MRR：相关项在返回列表中的首个位置
            first = next((r for r in retrieved if r in relevant), None)
            ranked_relevant.append([first] if first else [])

        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_precision = float(np.mean(precisions)) if precisions else 0.0
        mrr = mean_reciprocal_rank(ranked_relevant)
        n_docs = len(self.documents)
        n_rel = max(1, len({t for t in self.id_to_topic.values() if t is not None}))
        # 随机基线：假设每主题平均相关数
        avg_rel = np.mean([len(v) for v in self.topic_to_ids.values()]) if self.topic_to_ids else 1.0
        random_recall = random_baseline_retrieval(n_docs, self.k, int(avg_rel))
        return {
            "recall@k": mean_recall,
            "precision@k": mean_precision,
            "mrr": mrr,
            "random_recall@k": float(random_recall),
            "lift_over_random": float(mean_recall / random_recall) if random_recall > 0 else float("inf"),
        }

    def evaluate_clustering(self, n_clusters: int | None = None) -> Dict[str, float]:
        """KMeans 聚类，与真实主题标签比对 NMI / ARI。"""
        labels = [d.topic for d in self.documents]
        if any(l is None for l in labels):
            raise EvalError("聚类评测需要文档带 topic 真值", code="E500")
        true_arr = np.array(labels)
        n_true = len(np.unique(true_arr))
        k = n_clusters or n_true
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        pred = km.fit_predict(self.vectors)
        scores = clustering_scores(true_arr, pred)
        scores["n_clusters_true"] = float(n_true)
        scores["n_clusters_pred"] = float(k)
        return scores
