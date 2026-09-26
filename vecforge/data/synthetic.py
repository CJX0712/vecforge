"""合成主题语料生成器。

为没有真实数据时的可复现评测提供带主题标签的文档：
每个主题有一组强特征词，文档由「主题词 + 噪声词」拼接而成，因此同主题文档
在词袋空间天然聚拢，可作为检索 recall/MRR 与聚类 NMI/ARI 的真值基准。
作者：晨星
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..core.config import VecForgeConfig
from ..core.errors import DataError
from ..core.types import Document

# 6 个主题的强特征词表（互不重叠，避免主题混淆）
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "sports": ["football", "goal", "stadium", "player", "match", "coach",
               "league", "score", "tournament", "ball", "referee", "penalty"],
    "technology": ["computer", "algorithm", "software", "hardware", "network",
                   "data", "cloud", "code", "processor", "internet", "model", "gpu"],
    "health": ["doctor", "patient", "hospital", "medicine", "disease", "virus",
               "surgery", "therapy", "symptom", "vaccine", "nurse", "clinic"],
    "finance": ["bank", "stock", "market", "investment", "profit", "currency",
                "loan", "trading", "economy", "fund", "dividend", "portfolio"],
    "science": ["experiment", "theory", "research", "laboratory", "molecule",
                "physics", "biology", "hypothesis", "atom", "discovery", "quantum", "neuron"],
    "politics": ["election", "government", "policy", "parliament", "vote",
                 "senator", "law", "campaign", "president", "debate", "reform", "cabinet"],
}

_NOISE_POOL = [
    "thing", "world", "system", "point", "level", "area", "group", "case",
    "number", "part", "kind", "form", "side", "fact", "state", "value",
    "idea", "result", "process", "century", "option", "period", "member",
    "example", "reason", "method", "effect", "notice", "review", "report",
    "quality", "activity", "service", "design", "source", "income", "standard",
]


class SyntheticDataSource:
    """生成合成主题语料，实现 DataSource 契约。"""

    def __init__(self, config: Optional[VecForgeConfig] = None):
        self.config = config or VecForgeConfig()
        self.topics = list(TOPIC_KEYWORDS.keys())[: self.config.n_topics]
        if not self.topics:
            raise DataError("n_topics 必须为正整数", code="E200")

    def load(self) -> List[Document]:
        rng = random.Random(self.config.seed)
        docs: List[Document] = []
        for ti, topic in enumerate(self.topics):
            keywords = TOPIC_KEYWORDS[topic]
            for j in range(self.config.n_docs_per_topic):
                n_kw = rng.randint(4, 7)
                n_noise = rng.randint(3, 6)
                words = list(rng.sample(keywords, min(n_kw, len(keywords))))
                words += list(rng.sample(_NOISE_POOL, min(n_noise, len(_NOISE_POOL))))
                rng.shuffle(words)
                text = " ".join(words)
                docs.append(
                    Document(
                        doc_id=f"{topic}-{j:03d}",
                        text=text,
                        topic=topic,
                        metadata={"topic_index": ti},
                    )
                )
        return docs

    @property
    def topic_names(self) -> List[str]:
        return list(self.topics)
