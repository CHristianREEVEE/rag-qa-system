"""
RAG 引擎核心模块 — 文档加载、切片、向量索引、检索问答链
"""

import os
import glob
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    OPENAI_API_KEY, OPENAI_MODEL,
    DASHSCOPE_API_KEY, QWEN_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEFAULT_TEMPERATURE, MEMORY_WINDOW,
    FAISS_INDEX_DIR, UPLOAD_DIR,
)


# ════════════════════════════════════════════════════════════════
# 1. Embedding 与向量库
# ════════════════════════════════════════════════════════════════

def get_embeddings() -> HuggingFaceEmbeddings:
    """加载 BAAI/bge-small-zh embedding 模型。"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_split(file_path: str):
    """根据文件类型加载文档并切片。"""
    ext = os.path.splitext(file_path)[1].lower()
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,   # Markdown 本质是文本，用 TextLoader 即可
    }
    loader_cls = loader_map.get(ext)
    if loader_cls is None:
        raise ValueError(f"不支持的文件类型: {ext}")
    docs = loader_cls(file_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_vectorstore(docs, embeddings=None, index_name: str = "default"):
    """从文档列表构建 FAISS 索引并持久化。"""
    if embeddings is None:
        embeddings = get_embeddings()
    vs = FAISS.from_documents(docs, embeddings)
    save_path = os.path.join(FAISS_INDEX_DIR, index_name)
    vs.save_local(save_path)
    return vs


def load_vectorstore(index_name: str = "default", embeddings=None):
    """加载已持久化的 FAISS 索引。"""
    if embeddings is None:
        embeddings = get_embeddings()
    save_path = os.path.join(FAISS_INDEX_DIR, index_name)
    if not os.path.exists(save_path):
        return None
    return FAISS.load_local(
        save_path, embeddings,
        allow_dangerous_deserialization=True,
    )


def rebuild_vectorstore(embeddings=None, index_name: str = "default"):
    """扫描 uploads/ 目录下所有文件，重建向量库。"""
    if embeddings is None:
        embeddings = get_embeddings()
    all_docs = []
    for fp in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        if os.path.isfile(fp) and os.path.splitext(fp)[1].lower() in (".pdf", ".txt", ".md"):
            try:
                all_docs.extend(load_and_split(fp))
            except Exception as e:
                print(f"[WARN] 跳过 {fp}: {e}")
    if not all_docs:
        return None
    return build_vectorstore(all_docs, embeddings, index_name)


# ════════════════════════════════════════════════════════════════
# 2. LLM 工厂
# ════════════════════════════════════════════════════════════════

def get_llm(provider: str = "deepseek", temperature: float = DEFAULT_TEMPERATURE):
    """动态切换底座模型。"""
    if provider == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置")
        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=temperature,
        )
    else:
        # OpenAI 兼容路径
        if not OPENAI_API_KEY:
            raise ValueError("环境变量 OPENAI_API_KEY 未设置")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )


# ════════════════════════════════════════════════════════════════
# 3. RAG 问答链（新版 create_retrieval_chain 架构 + 多轮记忆）
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个专业的文档问答助手。请根据以下检索到的上下文片段回答用户问题。

要求：
1. 回答必须基于上下文内容，不要编造信息。
2. 如果上下文不足以回答问题，请明确说明"根据现有文档无法回答该问题"。
3. 回答末尾请标注引用来源（文档名称+页码/片段编号）。

检索到的上下文：
<context>
{context}
</context>
"""

# 会话历史存储（内存）
_session_store: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def build_rag_chain(
    vectorstore,
    provider: str = "openai",
    temperature: float = DEFAULT_TEMPERATURE,
):
    """构建带多轮对话记忆的 RAG 检索问答链。

    使用 LangChain 新版 LCEL 架构：
      create_stuff_documents_chain  →  把检索文档塞进 prompt
      create_retrieval_chain        →  检索 + 生成串联
    """
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

    llm = get_llm(provider, temperature)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # --- prompt：注入 context + 历史 ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # --- stuff chain：把文档拼进 context ---
    question_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_chain)

    # --- 包装 RunnableWithMessageHistory 实现多轮记忆 ---
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history, retriever


def clear_session(session_id: str = "default"):
    """清空指定会话的记忆。"""
    if session_id in _session_store:
        _session_store[session_id].clear()


def ask(
    chain,
    question: str,
    session_id: str = "default",
):
    """调用 RAG 链获取回答 + 引用来源。

    Returns:
        dict: {"answer": str, "sources": list[dict]}
    """
    result = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    answer = result.get("answer", "")
    # 提取引用文档来源
    sources = []
    for doc in result.get("context", []):
        sources.append({
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source", "未知"),
            "page": doc.metadata.get("page", "—"),
        })
    return {"answer": answer, "sources": sources}
