"""检索与聚类评测指标。作者：晨星"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ---------------- 检索指标 ----------------

def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """retrieved 为按相关性排序的 doc_id 列表，relevant 为相关集合。"""
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for d in top if d in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if d in relevant)
    return hits / len(top)


def mean_reciprocal_rank(
    ranked_relevant: Sequence[Sequence[str]],
) -> float:
    """每个查询给一个按序的 doc_id 列表，相关项命中其首个位置的倒数平均。"""
    if not ranked_relevant:
        return 0.0
    rr = []
    for retrieved in ranked_relevant:
        mrr = 0.0
        for rank, d in enumerate(retrieved, start=1):
            rr.append(1.0 / rank)
            break
        else:
            rr.append(0.0)
    return float(np.mean(rr))


# ---------------- 聚类指标 ----------------

def clustering_scores(true_labels: Sequence, pred_labels: Sequence) -> Dict[str, float]:
    """返回 NMI 与 ARI。两者均在 [0,1]，随机聚类 ≈0。"""
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return {"nmi": float(nmi), "ari": float(ari)}


def random_baseline_retrieval(n_docs: int, k: int, n_relevant: int) -> float:
    """检索 recall@k 的随机基线：从 N 篇里随机取 k 篇命中的期望比例。"""
    if n_docs <= 0 or n_relevant <= 0:
        return 0.0
    return min(k, n_relevant) / n_docs
