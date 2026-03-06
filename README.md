# AI Evolution Lab

A hands-on AI learning repo tracking progression from direct API calls to agentic, tool-using systems. Each sub-project is self-contained with its own `.venv` and `.env`.

## Projects

| Project | What it is | Stack |
|---------|-----------|-------|
| [ai-chatbot/](ai-chatbot/) | Multi-provider Streamlit chatbot | LangChain, OpenAI, Gemini, HuggingFace, Ollama |
| [hf-examples/](hf-examples/) | Local LLM inference with HuggingFace Transformers | Transformers, Gemma 3 |
| [ollama-examples/](ollama-examples/) | Prompt a local model via Ollama | Ollama, Llama 2 |
| [rag_chatbot/](rag_chatbot/) | RAG over a PDF using FAISS + LangChain | LangChain LCEL, FAISS, OpenAI |
| [agentic-ai-bootcamp/assignment1/](agentic-ai-bootcamp/assignment1/) | Multi-provider RAG with model comparison | LangChain, FAISS, OpenAI/Gemini/Ollama |
| [yt-chatbot/](yt-chatbot/) | RAG over a YouTube transcript | LangChain, ChromaDB, OpenAI |
| [personal-ai-assistant/](personal-ai-assistant/) | Agentic assistant with 5 tools, multi-provider LLM | litellm, FAISS, Streamlit |
| [notebooks/](notebooks/) | Jupyter exploration — embeddings, FAISS, ChromaDB | LangChain, OpenAI |

## Learning Tiers

**Tier 1 — Direct API integration**
- `hf-examples/` — local model inference via HuggingFace Transformers
- `ollama-examples/` — local model via Ollama
- `ai-chatbot/` — Streamlit chatbot switching across OpenAI, Gemini, HuggingFace, Ollama

**Tier 2 — Orchestration & RAG**
- `rag_chatbot/` — RAG pipeline with LangChain LCEL, FAISS, PDF source
- `agentic-ai-bootcamp/assignment1/` — configurable multi-provider RAG with side-by-side model comparison
- `yt-chatbot/` — RAG over YouTube transcripts with ChromaDB

**Tier 3 — Agentic & tool-calling**
- `personal-ai-assistant/` — agent loop with web search, math, weather, stocks, and PDF Q&A; model-agnostic via litellm

## Setup (per project)

```bash
cd <project>
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# edit .env with your API keys
```

## Conventions

- Each sub-project has its own `.venv` — never mix dependencies across projects
- `.env` files hold API keys and are never committed
- LangChain LCEL pipe syntax (`|`) is preferred for chains
- Python 3.12
