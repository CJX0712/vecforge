"""VecForge 错误码体系。

约定：E1xx 配置，E2xx 数据，E3xx 嵌入，E4xx 向量库，E5xx 评测。
作者：晨星
"""
from __future__ import annotations


class VecForgeError(Exception):
    """基类错误，携带 code 与 message。"""

    code = "E000"

    def __init__(self, message: str = "", code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigError(VecForgeError):
    code = "E100"


class DataError(VecForgeError):
    code = "E200"


class EmbeddingError(VecForgeError):
    code = "E300"


class StoreError(VecForgeError):
    code = "E400"


class EvalError(VecForgeError):
    code = "E500"
