# VecForge

> 文档向量检索与聚类系统 · 整合 **FAISS**（可选）+ **scikit-learn**（强制兜底）
> 作者：**晨星** · 版本：1.0.0

VecForge 是一套**可实际运行、一键复现**的 NLP 向量系统：把文档向量化后做
**语义检索**（recall@k / MRR）与**主题聚类**（NMI / ARI），所有模块按单一职责拆分、
接口解耦、可独立验证。上游优先复用业界领先开源成果（FAISS、scikit-learn），不自研
底层算法。

---

## 特性

- **三种嵌入**：TF-IDF、LSA（TruncatedSVD）、NMF，全部基于 scikit-learn。
- **双后端向量库**：FAISS `IndexFlat`（精确 ANN，可选）与 sklearn `NearestNeighbors`
  （强制兜底）。FAISS 缺失时 `available()` 探测并自动降级，**零下载也能跑通 demo**。
- **量化评测**：检索 recall@k / precision@k / MRR（含随机基线对比）；聚类 NMI / ARI。
- **接口契约**：`DataSource` / `Embedder` / `VectorStore` / `Metric` 用 `typing.Protocol`
  解耦，实现可热插拔。
- **一键复现**：隔离 venv + 锁定依赖（`requirements.lock.txt`）+ Dockerfile + Makefile。

---

## 快速开始

```bash
# 1) 建隔离环境并装依赖（需 Python 3.13）
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt

# 2) 跑端到端示例
.venv/Scripts/python vecforge/examples/run_demo.py

# 3) 或走 CLI
.venv/Scripts/python -m vecforge.cli demo --embed tfidf --store faiss

# 4) 跑测试
.venv/Scripts/python -m pytest -q -W ignore::UserWarning
```

> 注：Windows 用 `.venv\Scripts\python`；Linux/macOS 用 `.venv/bin/python`。

---

## 默认性能基线（合成 6 主题语料，240 篇）

| 嵌入 | 后端 | recall@5 | MRR | NMI | ARI | 随机基线 recall@5 |
|---|---|---|---|---|---|---|
| TF-IDF | FAISS | 0.102 | **1.000** | **1.000** | **1.000** | 0.021 (×4.9) |
| LSA | sklearn | 0.102 | **1.000** | **1.000** | **1.000** | 0.021 (×4.9) |
| NMF | FAISS | 0.056 | 0.867 | 0.167 | 0.087 | 0.021 |

> recall@5 上限为 5/39≈0.128（每主题 40 篇，自查询排除自身）；TF-IDF/LSA 取到
> 0.102 ≈ 上限的 80%，且 **MRR=1.0（top-1 永远同主题）**、聚类 **NMI/ARI=1.0（完美）**。
> NMF 为加性主题模型，在噪声词稀释下较弱，作为备选嵌入保留。

---

## 架构

```
DataSource ──▶ Embedder(TF-IDF/LSA/NMF) ──▶ VectorStore(FAISS/sklearn)
                                                   │
                              ┌────────────────────┼───────────────────┐
                              ▼                    ▼                   ▼
                     检索评测(recall/MRR)   聚类评测(NMI/ARI)     VecForgePipeline
                                                  └──── benchmark ────┘
```

模块划分见 [docs/architecture.md](docs/architecture.md)。

---

## 接口契约

| 契约 | 方法 | 说明 |
|---|---|---|
| `DataSource` | `load() -> list[Document]` | 产出文档 |
| `Embedder` | `fit` / `transform` / `fit_transform` | 文本→向量 |
| `VectorStore` | `add(vectors, ids)` / `query(vec, k)` / `available()` | 建库与检索 |
| `Metric` | `evaluate(...)` | 评测指标 |

---

## 项目结构

```
vecforge/
├── vecforge/
│   ├── core/         # types / config / errors / interfaces
│   ├── data/         # synthetic + file/csv loader
│   ├── embeddings/   # tfidf / lsa / nmf
│   ├── store/        # faiss_store / sklearn_store
│   ├── eval/         # metrics / benchmark
│   ├── pipeline/     # VecForgePipeline
│   ├── cli.py
│   └── examples/run_demo.py
├── tests/            # pytest 套件（51 用例）
├── requirements.txt / requirements.lock.txt
├── Dockerfile / Makefile / .gitignore
├── README.md
└── docs/architecture.md
```

---

## 一键复现（干净环境）

```bash
make venv && make install && make test && make demo
# 或容器化
docker build -t vecforge . && docker run --rm vecforge
```

---

## 许可

MIT © 晨星
