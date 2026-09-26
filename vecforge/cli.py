"""VecForge 命令行入口。作者：晨星

用法：
    python -m vecforge.cli demo            # 合成语料全流程评测
    python -m vecforge.cli demo --embed lsa --store sklearn
    python -m vecforge.cli file path.txt   # 对文本文件建库（无真值聚类）
"""
from __future__ import annotations

import argparse
import sys

from .core.config import VecForgeConfig
from .data.loader import FileDataSource
from .pipeline.vecforge_pipeline import VecForgePipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vecforge", description="VecForge 向量检索与聚类系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="合成语料端到端评测")
    d.add_argument("--embed", default="tfidf", choices=["tfidf", "lsa", "nmf"])
    d.add_argument("--store", default="faiss", choices=["faiss", "sklearn"])
    d.add_argument("--k", type=int, default=None)
    d.add_argument("--force-sklearn", action="store_true")

    f = sub.add_parser("file", help="对文本文件建库")
    f.add_argument("path")
    f.add_argument("--embed", default="tfidf", choices=["tfidf", "lsa", "nmf"])
    f.add_argument("--k", type=int, default=5)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = VecForgeConfig()
    if getattr(args, "force_sklearn", False):
        cfg.force_sklearn = True

    if args.cmd == "demo":
        pipe = VecForgePipeline(
            embedder_name=args.embed,
            store_prefer=args.store,
            config=cfg,
        )
        report = pipe.run(k=args.k)
        print(pipe.summary(report))
        return 0

    if args.cmd == "file":
        src = FileDataSource(args.path)
        pipe = VecForgePipeline(datasource=src, embedder_name=args.embed, config=cfg)
        report = pipe.run(k=args.k)
        print(pipe.summary(report))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
