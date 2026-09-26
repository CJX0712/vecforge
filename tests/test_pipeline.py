"""pipeline 层测试：端到端编排 + 报告完整性。作者：晨星"""
import pytest

from vecforge.pipeline.vecforge_pipeline import VecForgePipeline


@pytest.mark.parametrize("embed,store", [("tfidf", "faiss"), ("lsa", "sklearn"), ("nmf", "faiss")])
def test_pipeline_runs(embed, store, config):
    try:
        pipe = VecForgePipeline(embedder_name=embed, store_prefer=store, config=config)
        report = pipe.run()
    except Exception as e:  # FAISS 可能不可用
        if "FAISS" in str(e):
            pytest.skip("FAISS 不可用")
        raise
    assert report.n_docs > 0
    assert report.backend in ("faiss", "sklearn")
    assert report.embedding == embed
    assert "recall@k" in report.retrieval
    assert "nmi" in report.clustering
    # 检索显著优于随机
    assert report.retrieval["recall@k"] > report.retrieval["random_recall@k"]


def test_pipeline_summary_string(config):
    pipe = VecForgePipeline(embedder_name="tfidf", store_prefer="sklearn", config=config)
    report = pipe.run()
    s = pipe.summary(report)
    assert "VecForge" in s
    assert "recall" in s


def test_pipeline_empty_datasource(config):
    from vecforge.core.errors import DataError
    from vecforge.data.loader import FileDataSource

    with pytest.raises((DataError, RuntimeError)):
        # 指向空/不存在文件应报错
        VecForgePipeline(
            datasource=FileDataSource("/nonexistent/x.txt"), config=config
        ).run()
