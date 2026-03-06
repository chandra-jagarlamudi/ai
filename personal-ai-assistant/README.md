# Personal AI Assistant

A Streamlit-based personal assistant powered by [litellm](https://github.com/BerriAI/litellm) — works with any LLM provider. Set a model name and API key in `.env`, done.

## Tools

| Tool | What it does | Key required? |
|------|-------------|---------------|
| `web_search` | Live DuckDuckGo search for current news and facts | No |
| `solve_math` | Evaluates math expressions — arithmetic, trig, logs, factorials, and more | No |
| `get_weather` | Real-time weather via wttr.in (temperature, wind, UV index, …) | No |
| `get_stock_info` | Live stock price, P/E, market cap, 52-week range via Yahoo Finance | No |
| `search_pdf` | Semantic search over an uploaded PDF using FAISS + OpenAI embeddings | `OPENAI_API_KEY`* |

\* PDF indexing always uses `text-embedding-3-small` regardless of the active chat model.

## Setup

```bash
cd personal-ai-assistant

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Edit `.env`:

```env
# Any litellm-compatible model string
MODEL=gpt-4o-mini

# API key for that model's provider
API_KEY=sk-...

# Only needed if you use the PDF tool (OpenAI embeddings)
OPENAI_API_KEY=sk-...
```

## Switching providers

Just change two lines in `.env` — no code changes needed:

| Provider | `MODEL` example | Where to get `API_KEY` |
|----------|----------------|------------------------|
| OpenAI | `gpt-4o-mini`, `gpt-4.1`, `o4-mini` | platform.openai.com |
| Google | `gemini/gemini-2.0-flash` | aistudio.google.com |
| Groq (Meta/Llama) | `groq/llama-3.3-70b-versatile` | console.groq.com |
| Anthropic | `anthropic/claude-sonnet-4-6` | console.anthropic.com |
| Mistral | `mistral/mistral-large-latest` | console.mistral.ai |

The sidebar shows a text input for the model name — change it live without restarting.

## Run

```bash
source .venv/bin/activate
streamlit run assistant.py
```

Opens at `http://localhost:8501`.

## Using the PDF Tool

1. Upload a PDF using the **📄 PDF Search** file uploader in the sidebar
2. Wait for "Ready — N chunks indexed" confirmation
3. Ask any question about the document in the chat
4. The PDF store is cleared when you click **Clear conversation**

PDF indexing requires `OPENAI_API_KEY` regardless of the chat model. The PDF is chunked (800 chars, 100-char overlap), embedded with `text-embedding-3-small`, and indexed in a FAISS flat-IP index.

## Architecture

```
assistant.py   # Streamlit UI + litellm agent loop + PDF processing
tools.py       # Tool implementations + OpenAI schema definitions
  ├── web_search()     → ddgs (DuckDuckGo)
  ├── solve_math()     → ast-safe Python eval + math module
  ├── get_weather()    → wttr.in JSON API (no key)
  ├── get_stock_info() → yfinance / Yahoo Finance → JSON → stock card UI
  └── search_pdf()     → FAISS IndexFlatIP + text-embedding-3-small
requirements.txt
.env           # MODEL + API_KEY (never commit)
```

**Agent loop:** user message → `litellm.completion()` → tool calls → results → final answer. Full history kept in `st.session_state`.

**Multi-provider:** litellm routes calls to the right provider based on the model prefix (`gemini/`, `groq/`, `anthropic/`, etc.). OpenAI (no prefix) is the default.
