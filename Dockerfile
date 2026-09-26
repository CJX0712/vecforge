# VecForge 运行镜像 —— 干净环境一键复现
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENV_VECFORGE_FORCE_SKLEARN=0

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY vecforge/ ./vecforge/
COPY tests/ ./tests/
COPY vecforge/examples/run_demo.py ./vecforge/examples/run_demo.py

# 默认跑 demo（不依赖 GitHub / 网络）
CMD ["python", "vecforge/examples/run_demo.py"]
