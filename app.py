"""
RAG 智能文档问答系统 — Gradio Web 界面
双栏布局：左侧控制面板 | 右侧对话区 + 引用来源
"""

import os
import glob

# HuggingFace 镜像（仅在国内环境使用，Render 在美国直连即可）
if os.getenv('USE_MIRROR', '0') == '1':
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import gradio as gr

from config import (
    DEFAULT_MODEL_PROVIDER, DEFAULT_TEMPERATURE,
    UPLOAD_DIR, FAISS_INDEX_DIR,
)
from rag_engine import (
    get_embeddings, rebuild_vectorstore, load_vectorstore,
    build_rag_chain, clear_session, ask,
)


# ──────────────────────────── 全局状态 ────────────────────────────

class AppState:
    """全局应用状态。"""
    vectorstore = None
    chain = None
    embeddings = None


def init_app():
    """启动时尝试加载已有索引。"""
    AppState.embeddings = get_embeddings()
    AppState.vectorstore = load_vectorstore(embeddings=AppState.embeddings)
    if AppState.vectorstore is not None:
        AppState.chain, _ = build_rag_chain(
            AppState.vectorstore,
            provider=DEFAULT_MODEL_PROVIDER,
            temperature=DEFAULT_TEMPERATURE,
        )


# ──────────────────────────── 回调函数 ────────────────────────────

def cb_upload(files):
    """处理文件上传：保存到 uploads/ 目录。"""
    if not files:
        return "⚠️ 未选择文件"
    saved = []
    for f in files:
        name = os.path.basename(f.name) if hasattr(f, "name") else f
        dst = os.path.join(UPLOAD_DIR, os.path.basename(name))
        # Gradio 返回的是临时文件路径，直接复制
        import shutil
        shutil.copy2(f.name if hasattr(f, "name") else f, dst)
        saved.append(os.path.basename(name))
    return f"✅ 已上传 {len(saved)} 个文件：{', '.join(saved)}\n点击「重建向量库」以索引这些文件。"


def cb_rebuild(provider, temperature):
    """重建向量库并重新构建 RAG 链。"""
    try:
        vs = rebuild_vectorstore(embeddings=AppState.embeddings)
        if vs is None:
            return "⚠️ uploads/ 目录中没有可索引的 PDF/TXT/MD 文件"
        AppState.vectorstore = vs
        AppState.chain, _ = build_rag_chain(vs, provider=provider, temperature=temperature)
        clear_session()
        return f"✅ 向量库重建完成，当前底座模型：{provider}，temperature={temperature}"
    except Exception as e:
        return f"❌ 重建失败：{e}"


def cb_chat(message, history, provider, temperature):
    """Gradio ChatInterface 回调。"""
    if AppState.chain is None:
        # 尝试加载已有索引
        if AppState.vectorstore is None:
            AppState.vectorstore = load_vectorstore(embeddings=AppState.embeddings)
        if AppState.vectorstore is not None:
            AppState.chain, _ = build_rag_chain(
                AppState.vectorstore, provider=provider, temperature=temperature
            )
        else:
            yield "⚠️ 请先上传文档并重建向量库（左侧面板）。"
            return

    # 如果切换了模型/温度，重建链
    try:
        from langchain_classic.chains.retrieval import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from rag_engine import get_llm, SYSTEM_PROMPT
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_community.chat_message_histories import ChatMessageHistory
        from config import TOP_K

        llm = get_llm(provider, temperature)
        retriever = AppState.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        question_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_chain)
        from rag_engine import _get_session_history
        AppState.chain = RunnableWithMessageHistory(
            rag_chain, _get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    except Exception:
        pass  # 重建失败则沿用旧链

    try:
        result = ask(AppState.chain, message)
        answer = result["answer"]
        sources = result["sources"]

        # 组装引用来源文本
        src_text = ""
        if sources:
            src_lines = []
            for i, s in enumerate(sources, 1):
                src_lines.append(
                    f"  [{i}] 📄 {s['source']} (p.{s['page']})\n"
                    f"      {s['content'][:120]}..."
                )
            src_text = "\n\n---\n**📎 引用来源：**\n" + "\n".join(src_lines)

        yield answer + src_text
    except Exception as e:
        yield f"❌ 回答出错：{e}"


def cb_clear_history():
    """清空对话历史。"""
    clear_session()
    return "✅ 对话历史已清空"


# ──────────────────────────── 界面 ────────────────────────────

def create_ui():
    with gr.Blocks(title="RAG 智能文档问答系统") as demo:
        gr.Markdown("# 🔍 RAG 智能文档问答系统")
        gr.Markdown("基于 LangChain + FAISS + BGE Embeddings 的检索增强生成系统，支持 PDF/TXT/MD 文档问答与多轮对话。")

        with gr.Row():
            # ═══ 左栏：控制面板 ═══
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("### ⚙️ 控制面板")

                with gr.Group(elem_classes=["control-panel"]):
                    file_upload = gr.File(
                        label="📁 上传文档",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md"],
                    )
                    upload_btn = gr.Button("📤 上传文件", variant="secondary")
                    upload_status = gr.Textbox(
                        label="上传状态", interactive=False,
                        elem_classes=["status-box"],
                    )

                    gr.Markdown("---")

                    model_select = gr.Dropdown(
                        label="🤖 底座模型",
                        choices=[
                            ("DeepSeek (deepseek-chat)", "deepseek"),
                            ("OpenAI GPT-4o-mini", "openai"),
                        ],
                        value=DEFAULT_MODEL_PROVIDER,
                    )

                    temp_slider = gr.Slider(
                        label="🌡️ Temperature",
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=DEFAULT_TEMPERATURE,
                    )

                    rebuild_btn = gr.Button("🔄 重建向量库", variant="primary")
                    rebuild_status = gr.Textbox(
                        label="向量库状态", interactive=False,
                        elem_classes=["status-box"],
                    )

                    clear_btn = gr.Button("🧹 清空对话历史", variant="stop")
                    clear_status = gr.Textbox(
                        label="操作状态", interactive=False,
                        elem_classes=["status-box"],
                    )

            # ═══ 右栏：对话区 ═══
            with gr.Column(scale=2):
                gr.Markdown("### 💬 对话区")

                chatbot = gr.Chatbot(
                    label="RAG 问答",
                    height=500,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入问题",
                        placeholder="基于上传的文档提问...",
                        scale=4,
                        lines=2,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Accordion("📎 引用文档来源（Source Documents）", open=False):
                    sources_display = gr.JSON(label="最近一次引用", value=[])

        # ── 事件绑定 ──
        upload_btn.click(cb_upload, inputs=[file_upload], outputs=[upload_status])
        rebuild_btn.click(
            cb_rebuild,
            inputs=[model_select, temp_slider],
            outputs=[rebuild_status],
        )
        clear_btn.click(cb_clear_history, outputs=[clear_status])

        # Chatbot 发送
        def chat_wrapper(message, history, provider, temperature):
            yield from cb_chat(message, history, provider, temperature)

        send_btn.click(
            chat_wrapper,
            inputs=[msg_input, chatbot, model_select, temp_slider],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[msg_input])

        msg_input.submit(
            chat_wrapper,
            inputs=[msg_input, chatbot, model_select, temp_slider],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[msg_input])

    return demo


# ──────────────────────────── 启动 ────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG 智能文档问答系统启动中...")
    print(f"  上传目录: {UPLOAD_DIR}")
    print(f"  索引目录: {FAISS_INDEX_DIR}")
    print("=" * 60)

    init_app()

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )
