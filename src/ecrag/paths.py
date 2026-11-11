"""Filesystem locations used by the experiment scripts.

Every location can be overridden with an environment variable; by default all paths are
resolved relative to the repository root (two levels above this file).
"""

import os

ROOT = os.environ.get("DECIDER_RAG_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
RUNS = os.environ.get("DECIDER_RAG_RUNS", os.path.join(ROOT, "runs"))
DATA = os.environ.get("DECIDER_RAG_DATA", os.path.join(ROOT, "data"))
ECRAG_SRC = os.environ.get("ECRAG_SRC", os.path.join(ROOT, "third_party", "ec_rag", "src"))
ECRAG_DATA = os.environ.get("ECRAG_DATA", os.path.join(ROOT, "third_party", "ec_rag", "data"))
HF_HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
CLOUD_MODEL = os.environ.get("CLOUD_MODEL_PATH", os.path.join(ROOT, "models", "Qwen3.8-Flash-Next-FP8"))
