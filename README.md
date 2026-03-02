# Multilingual Customer Support Chatbot – Mistral 7B + RAG

**For Kenyan & East African businesses** — Handles English, Kiswahili, Sheng & code-mixed queries using company FAQs/docs.

## Why this project?
Standard chatbots fail on informal Kenyan language patterns ("Nipe discount tafadhali bro", "Bei ya hii item iko aje?").  
This RAG agent retrieves accurate context from business documents then generates polite, brand-aligned responses.

## Features
- Multilingual query understanding (English + Kiswahili + Sheng)
- Retrieval-Augmented Generation (RAG) with FAISS
- LoRA-fine-tuned Mistral-7B-Instruct for support-style tone
- Local-first: Runs efficiently on Mac Mini (Apple Silicon MPS / MLX)
- Modular: Easy to swap retriever, LLM, or knowledge base

## Tech Stack
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.3` (LoRA fine-tuned)
- **Embeddings & Retrieval**: `intfloat/multilingual-e5-large` + FAISS
- **Fine-tuning**: PEFT (LoRA)
- **Orchestration**: LangChain / LlamaIndex style chains
- **Hardware accel**: PyTorch MPS backend (Apple Metal)

## Quick Start (Local Inference)

1. Clone & install
```bash
git clone https://github.com/jumamiller/customer-support-chatbot.git
cd customer-support-chatbot
pip install -r requirements.txt
