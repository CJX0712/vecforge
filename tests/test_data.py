"""data 层测试：合成语料确定性 + 文件/CSV 加载。作者：晨星"""
import csv
from collections import Counter

import pytest

from vecforge.core.config import VecForgeConfig
from vecforge.core.errors import DataError
from vecforge.data.loader import CsvDataSource, FileDataSource
from vecforge.data.synthetic import SyntheticDataSource


@pytest.fixture
def cfg():
    return VecForgeConfig()


def test_synthetic_deterministic(cfg):
    a = SyntheticDataSource(cfg).load()
    b = SyntheticDataSource(cfg).load()
    assert [d.text for d in a] == [d.text for d in b]


def test_synthetic_topic_labels_present(cfg):
    docs = SyntheticDataSource(cfg).load()
    topics = {d.topic for d in docs}
    assert len(topics) == cfg.n_topics
    counts = Counter(d.topic for d in docs)
    assert all(c == cfg.n_docs_per_topic for c in counts.values())


def test_synthetic_unique_ids(cfg):
    docs = SyntheticDataSource(cfg).load()
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))


def test_file_loader(tmp_path):
    p = tmp_path / "docs.txt"
    p.write_text("alpha beta\ngamma delta\n\n", encoding="utf-8")
    docs = FileDataSource(str(p), topic="t").load()
    assert len(docs) == 2
    assert all(d.topic == "t" for d in docs)
    assert docs[0].doc_id.startswith("line-")


def test_file_loader_missing():
    with pytest.raises(DataError):
        FileDataSource("/nonexistent/path.txt").load()


def test_csv_loader(tmp_path):
    p = tmp_path / "docs.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text", "cat"])
        w.writerow(["1", "hello world", "news"])
        w.writerow(["2", "foo bar", "sport"])
    docs = CsvDataSource(str(p), text_col="text", topic_col="cat", id_col="id").load()
    assert len(docs) == 2
    assert docs[0].doc_id == "1"
    assert docs[1].topic == "sport"


def test_csv_loader_missing_col(tmp_path):
    p = tmp_path / "docs.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DataError):
        CsvDataSource(str(p), text_col="missing").load()
