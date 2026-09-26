# VecForge 架构文档

> 作者：晨星 · 版本：1.0.0

## 1. 设计原则

1. **优先复用业界领先开源成果**：向量检索用 Meta 的 **FAISS**，嵌入与聚类用
   **scikit-learn**，不自研底层数值算法。
2. **单一职责 + 接口解耦**：每个模块只做一件事，模块间通过 `typing.Protocol`
   契约通信，可用任意实现热插拔。
3. **离线兜底，零下载可跑通**：FAISS 为可选依赖，缺失时经 `available()` 探测自动
   降级到 sklearn `NearestNeighbors`（中小规模精确等价），保证干净环境一键复现。
4. **可独立验证**：每个模块有单测 + 最小可运行示例；模块组合成端到端 demo 链路。

## 2. 模块职责与接口

| 模块 | 职责 | 核心接口 | 依赖 |
|---|---|---|---|
| `core` | 类型/dataclass、ENV 配置、错误码 E1xx–E5xx、Protocol 契约 | `DataSource`/`Embedder`/`VectorStore`/`Metric` | 无 |
| `data` | 合成主题语料（带真值） + 文件/CSV 加载 | `load() -> list[Document]` | 无 |
| `embeddings` | TF-IDF / LSA(TruncatedSVD) / NMF | `fit` / `transform` / `fit_transform` | sklearn |
| `store` | 向量库：FAISS(可选) + sklearn(兜底) | `add` / `query` / `available` | faiss(sklearn) |
| `eval` | 检索 recall@k/precision@k/MRR + 聚类 NMI/ARI | `evaluate_retrieval` / `evaluate_clustering` | sklearn |
| `pipeline` | 编排 + benchmark + 报告 | `run()` / `summary()` | 全部 |
| `cli` / `examples` | 命令行 + 独立示例 | — | 全部 |

## 3. 数据流

```
DataSource.load()
   │  list[Document]  (text + topic 真值)
   ▼
Embedder.fit_transform(corpus)
   │  np.ndarray  (n_docs, d)   — LSA/NMF 已 L2 归一化
   ▼
VectorStore.add(vectors, ids)
   │  建立索引
   ▼
for each doc: VectorStore.query(self_vec, k)
   │  QueryResult(hits)
   ▼
Benchmark.evaluate_retrieval()  → recall@k / precision@k / MRR (+ 随机基线)
Benchmark.evaluate_clustering() → KMeans vs 真值 → NMI / ARI
   ▼
EvalReport  →  VecForgePipeline.summary()
```

## 4. 后端选择逻辑

```python
def get_store(prefer="faiss", force_sklearn=False):
    if not force_sklearn and prefer == "faiss" and FaissVectorStore.available():
        return FaissVectorStore(metric="cosine")
    return SklearnVectorStore(metric="cosine")
```

- `FaissVectorStore.available()` 在导入期探测 `import faiss` 是否成功。
- 环境变量 `ENV_VECFORGE_FORCE_SKLEARN=1` 可强制走 sklearn 路径（CI/离线场景）。

## 5. 评测指标定义

- **recall@k**：每篇文档以自身向量查询，同主题其他文档为真值相关集；
  `命中数 / 相关总数`，取全体均值。随机基线 = `min(k, n_rel) / n_docs`。
- **MRR**：相关项在返回列表首个位置的倒数均值（本系统恒为 rank-1 → 1.0）。
- **NMI / ARI**：KMeans 簇标签与真实主题标签互信息/调整兰德系数，随机≈0。

## 6. 性能基线（见 README）

TF-IDF / LSA 在合成语料上达到 recall@5≈0.102（上限 0.128）、MRR=1.0、
NMI=ARI=1.0，相对随机基线提升约 ×4.9；NMF 较弱（噪声稀释），作备选。

## 7. 可复现性

- 隔离 venv（`requirements.lock.txt` 锁定全部传递依赖）。
- `make` / `Dockerfile` 提供一键路径。
- 合成语料以固定 `seed` 保证确定性；KMeans `random_state=42`。

## 8. 已知限制与优化方向

- **嵌入维度**：LSA/NMF 默认 64 维，合成主题少时可降到 8–16 提速。
- **NMF 鲁棒性**：噪声词较多时 NMI 偏低，可加 stop-word 过滤或增大 `max_iter`。
- **规模**：sklearn `NearestNeighbors` 为 O(N·d) 暴力；百万级切换 FAISS IVF/HNSW。
- **语义层**：当前为词袋/稀疏降维，可接入句向量模型（如 Sentence-Transformers）
  提升跨词面语义召回（需作为新的 `Embedder` 实现热插拔）。
