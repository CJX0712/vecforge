"""VecForge 配置层。

所有可调参数均以 ``ENV_VECFORGE_*`` 环境变量为最高优先级，便于一键复现时
不改动代码即可调参；缺失时回退到合理默认值。
作者：晨星
"""
from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class VecForgeConfig:
    """全局配置。通过环境变量覆盖。"""

    # 合成语料
    DEFAULT_N_TOPICS = 6
    DEFAULT_N_DOCS_PER_TOPIC = 40
    DEFAULT_VOCAB_NOISE = 40  # 噪声词数量
    DEFAULT_SEED = 20260926

    # 嵌入
    DEFAULT_EMBEDDING_DIM = 64  # LSA/NMF 降维维度
    DEFAULT_MAX_FEATURES = 2000  # TF-IDF 词表上限

    # 检索 / 聚类
    DEFAULT_K = 5  # 检索返回条数
    DEFAULT_N_CLUSTERS = None  # None 时取真实主题数

    def __init__(self, **overrides):
        self.n_topics: int = _env_int("ENV_VECFORGE_N_TOPICS", self.DEFAULT_N_TOPICS)
        self.n_docs_per_topic: int = _env_int(
            "ENV_VECFORGE_N_DOCS", self.DEFAULT_N_DOCS_PER_TOPIC
        )
        self.vocab_noise: int = _env_int("ENV_VECFORGE_NOISE", self.DEFAULT_VOCAB_NOISE)
        self.seed: int = _env_int("ENV_VECFORGE_SEED", self.DEFAULT_SEED)

        self.embedding_dim: int = _env_int(
            "ENV_VECFORGE_EMB_DIM", self.DEFAULT_EMBEDDING_DIM
        )
        self.max_features: int = _env_int(
            "ENV_VECFORGE_MAXFEAT", self.DEFAULT_MAX_FEATURES
        )

        self.k: int = _env_int("ENV_VECFORGE_K", self.DEFAULT_K)
        self.n_clusters: Optional[int] = overrides.get("n_clusters", self.DEFAULT_N_CLUSTERS)

        # 是否强制使用 sklearn 兜底（跳过 FAISS）
        self.force_sklearn: bool = _env_bool("ENV_VECFORGE_FORCE_SKLEARN", False)

    def as_dict(self) -> dict:
        return {
            "n_topics": self.n_topics,
            "n_docs_per_topic": self.n_docs_per_topic,
            "vocab_noise": self.vocab_noise,
            "seed": self.seed,
            "embedding_dim": self.embedding_dim,
            "max_features": self.max_features,
            "k": self.k,
            "n_clusters": self.n_clusters,
            "force_sklearn": self.force_sklearn,
        }
