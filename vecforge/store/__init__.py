"""VecForge store 包：FAISS(可选) + sklearn(强制)。作者：晨星"""
from .faiss_store import FaissVectorStore
from .sklearn_store import SklearnVectorStore


def get_store(prefer: str = "faiss", metric: str = "cosine", force_sklearn: bool = False):
    """按优先级选择向量库。FAISS 不可用时自动降级到 sklearn。"""
    if not force_sklearn and prefer == "faiss" and FaissVectorStore.available():
        try:
            return FaissVectorStore(metric=metric)
        except Exception:
            pass
    return SklearnVectorStore(metric=metric)


__all__ = ["FaissVectorStore", "SklearnVectorStore", "get_store"]
