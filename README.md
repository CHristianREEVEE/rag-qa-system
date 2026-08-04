---
title: RAG 智能文档问答系统
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# RAG 智能文档问答系统

基于 LangChain + FAISS + BGE Embeddings + DeepSeek 的检索增强生成系统。

## 功能

- 📄 支持 PDF / TXT / Markdown 文档上传
- 🔍 BGE 向量检索 + FAISS 索引
- 🤖 DeepSeek / OpenAI 动态切换
- 💬 多轮对话（保留最近 5 轮）
- 📎 引用来源展示

## 使用方法

1. 在 Settings → Secrets 中配置 `DEEPSEEK_API_KEY`
2. 上传文档 → 点击「重建向量库」
3. 在对话框中提问
