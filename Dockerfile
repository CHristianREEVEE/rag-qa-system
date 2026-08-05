# ──────────────────────────────────────────
# Dockerfile — RAG 智能文档问答系统
# ──────────────────────────────────────────
FROM python:3.10-slim

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 创建必要目录
RUN mkdir -p uploads faiss_index

# 预下载 BGE embedding 模型（避免启动时下载超时）
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-zh')"

EXPOSE 7860

CMD ["python", "app.py"]
