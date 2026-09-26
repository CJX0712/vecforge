# VecForge Makefile —— 一键复现工作流
PY ?= python3
VENV ?= .venv

.PHONY: help venv install test demo cli lint clean lock

help:
	@echo "VecForge 命令:"
	@echo "  make venv     创建隔离虚拟环境"
	@echo "  make install  安装依赖 (requirements.txt)"
	@echo "  make test     运行 pytest"
	@echo "  make demo     跑端到端示例"
	@echo "  make cli      跑 CLI demo"
	@echo "  make lock     重新生成 requirements.lock.txt"
	@echo "  make clean    清理缓存"

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip

install: venv
	$(VENV)/bin/pip install -r requirements.txt

test:
	$(VENV)/bin/python -m pytest -q -W ignore::UserWarning

demo:
	$(VENV)/bin/python vecforge/examples/run_demo.py

cli:
	$(VENV)/bin/python -m vecforge.cli demo

lock:
	$(VENV)/bin/pip freeze > requirements.lock.txt

clean:
	rm -rf __pycache__ .pytest_cache $(VENV)
	find . -name '*.pyc' -delete
