"""VecForge embeddings 包：TF-IDF / LSA / NMF。作者：晨星"""
from .base import BaseEmbedder
from .decomposition import LsaEmbedder, NmfEmbedder
from .tfidf import TfidfEmbedder

EMBEDDERS = {
    "tfidf": TfidfEmbedder,
    "lsa": LsaEmbedder,
    "nmf": NmfEmbedder,
}


def get_embedder(name: str, config=None):
    if name not in EMBEDDERS:
        raise KeyError(f"未知嵌入器 '{name}'，可选: {list(EMBEDDERS)}")
    return EMBEDDERS[name](config)


__all__ = ["BaseEmbedder", "TfidfEmbedder", "LsaEmbedder", "NmfEmbedder", "EMBEDDERS", "get_embedder"]
