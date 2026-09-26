"""文件 / CSV 数据源加载器。实现 DataSource 契约。作者：晨星"""
from __future__ import annotations

import csv
import os
from typing import List, Optional

from ..core.errors import DataError
from ..core.types import Document


class FileDataSource:
    """逐行读取纯文本文件，每行一篇文档。可选统一 topic。"""

    def __init__(self, path: str, topic: Optional[str] = None, encoding: str = "utf-8"):
        if not os.path.isfile(path):
            raise DataError(f"文件不存在: {path}", code="E200")
        self.path = path
        self.topic = topic
        self.encoding = encoding

    def load(self) -> List[Document]:
        docs: List[Document] = []
        with open(self.path, "r", encoding=self.encoding) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                docs.append(Document(doc_id=f"line-{i:05d}", text=line, topic=self.topic))
        if not docs:
            raise DataError("文件为空或无有效行", code="E200")
        return docs


class CsvDataSource:
    """读取 CSV：指定文本列，可选主题列与 id 列。"""

    def __init__(
        self,
        path: str,
        text_col: str = "text",
        topic_col: Optional[str] = None,
        id_col: Optional[str] = None,
        encoding: str = "utf-8",
    ):
        if not os.path.isfile(path):
            raise DataError(f"文件不存在: {path}", code="E200")
        self.path = path
        self.text_col = text_col
        self.topic_col = topic_col
        self.id_col = id_col
        self.encoding = encoding

    def load(self) -> List[Document]:
        docs: List[Document] = []
        with open(self.path, "r", encoding=self.encoding, newline="") as fh:
            reader = csv.DictReader(fh)
            if self.text_col not in (reader.fieldnames or []):
                raise DataError(
                    f"CSV 缺少文本列 '{self.text_col}'，可用列: {reader.fieldnames}",
                    code="E200",
                )
            for i, row in enumerate(reader):
                text = (row.get(self.text_col) or "").strip()
                if not text:
                    continue
                topic = row.get(self.topic_col) if self.topic_col else None
                doc_id = (
                    row.get(self.id_col)
                    if self.id_col
                    else f"row-{i:05d}"
                )
                docs.append(
                    Document(doc_id=str(doc_id), text=text, topic=topic)
                )
        if not docs:
            raise DataError("CSV 无有效行", code="E200")
        return docs
