"""
RAG 智能文档问答系统 — 配置中心
"""

import os

# ──────────────────────────── 模型配置 ────────────────────────────

# 默认使用的底座模型：deepseek / openai / qwen
DEFAULT_MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "deepseek")

# DeepSeek 配置（API 兼容 OpenAI 格式）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# DashScope (Qwen2) 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = "qwen2-72b-instruct"

# Embedding 模型
EMBEDDING_MODEL = "BAAI/bge-small-zh"

# 默认温度
DEFAULT_TEMPERATURE = 0.3

# ──────────────────────────── RAG 参数 ────────────────────────────

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4              # 检索返回文档数

# 多轮对话记忆窗口（轮数）
MEMORY_WINDOW = 5

# ──────────────────────────── 路径 ────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
