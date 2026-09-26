"""VecForge 流水线：串联 数据 -> 嵌入 -> 向量库 -> 评测。作者：晨星"""
from __future__ import annotations

from typing import Dict, Optional

from ..core.config import VecForgeConfig
from ..core.types import Document, EvalReport
from ..data.synthetic import SyntheticDataSource
from ..embeddings import get_embedder
from ..eval.benchmark import Benchmark
from ..store import get_store


class VecForgePipeline:
    """端到端编排器，单一职责：组合可独立验证的模块并产出评测报告。"""

    def __init__(
        self,
        datasource=None,
        embedder_name: str = "tfidf",
        store_prefer: str = "faiss",
        config: Optional[VecForgeConfig] = None,
    ):
        self.config = config or VecForgeConfig()
        self.datasource = datasource or SyntheticDataSource(self.config)
        self.embedder = get_embedder(embedder_name, self.config)
        self.store = get_store(
            prefer=store_prefer,
            metric="cosine",
            force_sklearn=self.config.force_sklearn,
        )
        self.embedder_name = embedder_name
        self.store_prefer = store_prefer

    def run(self, k: Optional[int] = None, n_clusters: Optional[int] = None) -> EvalReport:
        k = k or self.config.k
        docs: list[Document] = self.datasource.load()
        if not docs:
            raise RuntimeError("数据源返回空文档集")
        texts = [d.text for d in docs]
        X = self.embedder.fit_transform(texts)
        self.store.add(X, [d.doc_id for d in docs])

        bench = Benchmark(docs, X, self.store, k=k)
        retrieval = bench.evaluate_retrieval()
        clustering = bench.evaluate_clustering(n_clusters=n_clusters)

        return EvalReport(
            n_docs=len(docs),
            retrieval=retrieval,
            clustering=clustering,
            backend=self.store.backend,
            embedding=self.embedder.name,
        )

    def summary(self, report: EvalReport) -> str:
        r = report.retrieval
        c = report.clustering
        return (
            f"[VecForge] backend={report.backend} embed={report.embedding} "
            f"n_docs={report.n_docs}\n"
            f"  检索 recall@{self.config.k}={r['recall@k']:.3f} "
            f"precision@{self.config.k}={r['precision@k']:.3f} "
            f"MRR={r['mrr']:.3f} (random={r['random_recall@k']:.3f}, "
            f"提升×{r['lift_over_random']:.1f})\n"
            f"  聚类 NMI={c['nmi']:.3f} ARI={c['ari']:.3f} "
            f"(true_k={int(c['n_clusters_true'])}, pred_k={int(c['n_clusters_pred'])})"
        )
