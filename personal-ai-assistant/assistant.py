"""
assistant.py — Personal AI Assistant (Streamlit UI)

Run:
    streamlit run assistant.py
"""

import io
import json
import os

import faiss
import litellm
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

import tools
from tools import dispatch_tool, get_tool_definitions, set_pdf_store

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model & API key ───────────────────────────────────────────────────────────
# litellm uses provider-specific env vars; infer the right one from the model prefix.
_PROVIDER_KEY_MAP = {
    "gemini/":    "GEMINI_API_KEY",
    "groq/":      "GROQ_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "cohere/":    "COHERE_API_KEY",
    "mistral/":   "MISTRAL_API_KEY",
}


def _provider_name(model: str) -> str:
    for prefix, _ in _PROVIDER_KEY_MAP.items():
        if model.startswith(prefix):
            return prefix.rstrip("/").capitalize()
    return "OpenAI"


def _apply_api_key(model: str, api_key: str) -> None:
    """Set the correct provider env var for litellm based on the model prefix."""
    for prefix, env_var in _PROVIDER_KEY_MAP.items():
        if model.startswith(prefix):
            os.environ[env_var] = api_key
            return
    os.environ["OPENAI_API_KEY"] = api_key


_LLM_API_KEY   = os.getenv("API_KEY", "")
_DEFAULT_MODEL = os.getenv("MODEL", "gpt-4o-mini")

if "model" not in st.session_state:
    st.session_state.model = _DEFAULT_MODEL

# Apply key for the current model on every rerun
if _LLM_API_KEY:
    _apply_api_key(st.session_state.model, _LLM_API_KEY)

# OpenAI client — kept solely for PDF embeddings (text-embedding-3-small)
_openai_key   = os.getenv("OPENAI_API_KEY", "")
_embed_client = OpenAI(api_key=_openai_key) if _openai_key else None

# ── Tool metadata ─────────────────────────────────────────────────────────────
TOOLS_INFO = [
    {
        "icon": "🔍",
        "name": "Web Search",
        "tool": "web_search",
        "color": "#4A90E2",
        "description": "Live DuckDuckGo search for current news and facts.",
        "examples": [
            "What are the latest AI news today?",
            "Search for Python 3.13 new features",
            "Latest breakthroughs in quantum computing",
        ],
    },
    {
        "icon": "🧮",
        "name": "Math Solver",
        "tool": "solve_math",
        "color": "#9B59B6",
        "description": "Evaluate expressions: trig, logs, roots, factorials, and more.",
        "examples": [
            "What is sqrt(144) + factorial(5)?",
            "Calculate sin(pi/4) and cos(pi/3)",
            "What is 2**32 mod 1000000007?",
        ],
    },
    {
        "icon": "🌤️",
        "name": "Weather",
        "tool": "get_weather",
        "color": "#27AE60",
        "description": "Real-time weather for any city — no API key needed.",
        "examples": [
            "What's the weather in Tokyo right now?",
            "Weather in New York",
            "Is it raining in London today?",
        ],
    },
    {
        "icon": "📈",
        "name": "Stock Info",
        "tool": "get_stock_info",
        "color": "#E74C3C",
        "description": "Live price, P/E, market cap, and more via Yahoo Finance.",
        "examples": [
            "What's Apple's current stock price?",
            "Show me TSLA stock data",
            "NVIDIA stock price and P/E ratio",
        ],
    },
    {
        "icon": "📄",
        "name": "PDF Search",
        "tool": "search_pdf",
        "color": "#F39C12",
        "description": "Upload a PDF via the sidebar, then ask any question about it.",
        "examples": [
            "Summarise this PDF",
            "What are the key findings?",
            "Explain the methodology used",
        ],
    },
]

TOOL_COLORS = {t["tool"]: t["color"] for t in TOOLS_INFO}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a capable and concise personal AI assistant.
You have access to the following tools:

1. web_search      — Search DuckDuckGo for up-to-date information.
2. solve_math      — Evaluate mathematical expressions precisely.
3. get_weather     — Get current weather for any location.
4. get_stock_info  — Get real-time stock price and financial data.
5. search_pdf      — Semantically search the uploaded PDF document.

Guidelines:
- Always use a tool when the question calls for live data, calculations, or file content.
- After a tool returns results, synthesize the output into a clear, helpful answer.
- For PDF questions, ALWAYS call search_pdf with the user's specific question. Never ask the user
  for a file path — the PDF is uploaded via the sidebar and already indexed. Call search_pdf
  for every PDF-related question, including follow-ups (each call fetches the most relevant chunks).
- Be concise but thorough. Format numerical data neatly using markdown tables or lists where helpful.
- NEVER use LaTeX notation (\\( ... \\), \\[ ... \\], \\frac, \\sin, etc.). Always write math results
  in plain text, e.g. "sin(π/4) ≈ 0.7071" or use a code block for multi-line results.
- For weather results, always reproduce the tool output EXACTLY and verbatim inside a markdown
  code block (``` ... ```) so the aligned formatting is preserved. Do not paraphrase or reformat it.
- For stock results, the UI renders a visual card automatically. Just give a brief one-sentence
  summary (e.g. "NVDA is trading at 183.34, up 0.09% from yesterday's close."). Do not list all
  the fields again — the card already shows them.
"""

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }

/* Sidebar tool cards */
.tool-card {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    border-left: 3px solid var(--card-color);
    transition: background 0.15s ease;
}
.tool-card:hover { background: rgba(255,255,255,0.08); }
.tool-card .card-title {
    margin: 0 0 3px 0;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}
.tool-card .card-desc {
    margin: 0;
    font-size: 0.76rem;
    opacity: 0.65;
    line-height: 1.4;
}

/* Tool usage badges */
.tool-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.71rem;
    font-weight: 600;
    margin: 0 3px 6px 0;
    letter-spacing: 0.02em;
}

/* Stock card */
.stock-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0 10px;
    font-family: monospace;
}
.stock-header {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 10px;
    opacity: 0.9;
}
.stock-header .sticker {
    opacity: 0.6;
    margin-left: 4px;
    font-weight: 400;
}
.stock-header .scurrency {
    opacity: 0.45;
    font-size: 0.8rem;
    margin-left: 8px;
}
.stock-price-row {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 14px;
}
.stock-price-row .sprice {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.stock-price-row .schange {
    font-size: 0.85rem;
    font-weight: 600;
}
.up   { color: #2ECC71; }
.down { color: #E74C3C; }
.flat { color: #95A5A6; }
.srow {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.83rem;
}
.srow:last-child { border-bottom: none; }
.slabel { opacity: 0.55; }
.svalue { font-weight: 500; }

/* Welcome screen */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px 40px;
    opacity: 0.55;
    text-align: center;
}
.welcome-wrap .big-icon { font-size: 3.5rem; margin-bottom: 10px; }
.welcome-wrap h2 { font-size: 1.8rem; margin: 0 0 6px; font-weight: 700; }
.welcome-wrap p  { font-size: 0.95rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "display_messages" not in st.session_state:
    # Each entry: {role, content, tool_calls: [{name, args, result}]}
    st.session_state.display_messages = []

if "api_messages" not in st.session_state:
    # Full OpenAI-format history including tool messages
    st.session_state.api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = None  # dict with FAISS index + chunks, or None

if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None

# Sync the module-level PDF store in tools.py on every Streamlit rerun
set_pdf_store(st.session_state.pdf_store)


# ── PDF processing ────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 100


def process_pdf(file_bytes: bytes, filename: str) -> dict | None:
    """
    Extract text from a PDF, chunk it, embed with OpenAI, and build a FAISS
    flat-IP index (cosine similarity via normalised vectors).
    Returns a store dict or None if no readable text found.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            full_text += text + "\n\n"

    if not full_text.strip():
        return None

    # Character-level chunking with overlap
    chunks, start = [], 0
    while start < len(full_text):
        chunks.append(full_text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    # Embed all chunks (OpenAI batch request — requires OPENAI_API_KEY)
    if _embed_client is None:
        return None  # caller shows an error via the UI
    resp = _embed_client.embeddings.create(input=chunks, model=EMBED_MODEL)
    embeddings = np.array([e.embedding for e in resp.data], dtype=np.float32)

    # Normalise → cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-10)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    return {
        "name": filename,
        "chunks": chunks,
        "index": index,
        "num_chunks": len(chunks),
    }


# ── Agent logic ───────────────────────────────────────────────────────────────
def run_agent(api_messages: list) -> tuple[list, str]:
    """
    Run one full agent turn.
    Mutates api_messages in-place with intermediate tool messages.
    Returns (tool_calls_log, final_reply_text).
    """
    tools = get_tool_definitions()
    tool_calls_log = []

    while True:
        response = litellm.completion(
            model=st.session_state.model,
            messages=api_messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message

        # No tool calls → final answer
        if not message.tool_calls:
            return tool_calls_log, message.content or ""

        # Convert to plain dict for cross-provider compatibility
        api_messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            name = tc.function.name
            args_str = tc.function.arguments
            result = dispatch_tool(name, args_str)

            try:
                args_display = json.loads(args_str)
            except Exception:
                args_display = args_str

            tool_calls_log.append({"name": name, "args": args_display, "result": result})

            api_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


# ── Stock card renderer ───────────────────────────────────────────────────────
def render_stock_card(result_json: str) -> None:
    """Parse get_stock_info JSON and render a colored inline card."""
    try:
        d = json.loads(result_json)
    except Exception:
        st.text(result_json)
        return

    if "error" in d:
        st.error(d["error"])
        return

    price = d.get("price")
    prev  = d.get("prev_close")

    # Determine direction colour
    if price and prev:
        change     = price - prev
        change_pct = (change / prev) * 100
        if change > 0:
            cls, arrow = "up",   "▲"
        elif change < 0:
            cls, arrow = "down", "▼"
        else:
            cls, arrow = "flat", "—"
        change_str = f"{arrow} {abs(change):.2f} ({abs(change_pct):.2f}%)"
    else:
        cls, change_str = "flat", ""

    def fmt(v, decimals=2, prefix="", suffix=""):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{prefix}{v:,.{decimals}f}{suffix}"
        if isinstance(v, int):
            return f"{prefix}{v:,}{suffix}"
        return str(v)

    volume_str = fmt(d.get("volume"), decimals=0)

    rows = [
        ("Previous Close", fmt(prev)),
        ("Day Range",      f"{fmt(d.get('day_low'))} – {fmt(d.get('day_high'))}"),
        ("52-Week Range",  f"{fmt(d.get('week_52_low'))} – {fmt(d.get('week_52_high'))}"),
        ("Volume",         volume_str),
        ("Market Cap",     d.get("market_cap") or "N/A"),
        ("Sector",         d.get("sector") or "N/A"),
        ("P/E Ratio",      fmt(d.get("pe"))),
        ("EPS",            fmt(d.get("eps"))),
        ("Dividend Yield", d.get("dividend_yield") or "N/A"),
    ]

    rows_html = "".join(
        f'<div class="srow"><span class="slabel">{label}</span>'
        f'<span class="svalue">{value}</span></div>'
        for label, value in rows
    )

    card = f"""
    <div class="stock-card">
      <div class="stock-header">
        {d.get("name", d.get("ticker", ""))}
        <span class="sticker">({d.get("ticker", "")})</span>
        <span class="scurrency">{d.get("currency", "USD")}</span>
      </div>
      <div class="stock-price-row">
        <span class="sprice {cls}">{fmt(price)}</span>
        <span class="schange {cls}">{change_str}</span>
      </div>
      {rows_html}
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)


# ── Helper: render tool call details ─────────────────────────────────────────
def render_tool_calls(tool_calls: list) -> None:
    """Render tool-usage badges and expandable detail blocks."""
    if not tool_calls:
        return

    # Badges row
    badges = ""
    for tc in tool_calls:
        color = TOOL_COLORS.get(tc["name"], "#888")
        label = tc["name"].replace("_", " ").title()
        badges += (
            f'<span class="tool-badge" '
            f'style="background:{color}1a;color:{color};border:1px solid {color}55;">'
            f'{label}</span>'
        )
    st.markdown(badges, unsafe_allow_html=True)

    # Per-tool rendering
    for tc in tool_calls:
        color = TOOL_COLORS.get(tc["name"], "#888")
        label = tc["name"].replace("_", " ").title()

        # Stock gets an inline card, no expander needed
        if tc["name"] == "get_stock_info":
            render_stock_card(tc["result"])
            continue

        with st.expander(f"🔧 {label}", expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Arguments**")
                st.json(tc["args"], expanded=True)
            with col2:
                st.markdown("**Result**")
                preview = tc["result"]
                if len(preview) > 1500:
                    preview = preview[:1500] + "\n\n…(truncated)"
                st.text(preview)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Personal AI")
    _new_model = st.text_input("Model", value=st.session_state.model).strip()
    if _new_model and _new_model != st.session_state.model:
        st.session_state.model = _new_model
        if _LLM_API_KEY:
            _apply_api_key(_new_model, _LLM_API_KEY)
    st.caption(f"Provider: {_provider_name(st.session_state.model)}")
    if not _LLM_API_KEY:
        st.warning("⚠️ `API_KEY` not set in `.env`")
    st.divider()

    st.markdown("#### What I can do")

    for tool in TOOLS_INFO:
        color = tool["color"]
        st.markdown(
            f'<div class="tool-card" style="--card-color:{color};">'
            f'<p class="card-title">{tool["icon"]} {tool["name"]}</p>'
            f'<p class="card-desc">{tool["description"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if tool["tool"] == "search_pdf":
            # File uploader lives inside the PDF Search card block
            uploaded = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                key="pdf_uploader",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                if st.session_state.pdf_filename != uploaded.name:
                    with st.spinner(f"Indexing {uploaded.name}…"):
                        store = process_pdf(uploaded.read(), uploaded.name)
                    if store:
                        st.session_state.pdf_store = store
                        st.session_state.pdf_filename = uploaded.name
                        set_pdf_store(store)
                        st.success(f"Ready — {store['num_chunks']} chunks indexed")
                    elif store is None and _embed_client is None:
                        st.error("PDF indexing requires `OPENAI_API_KEY` (used for embeddings).")
                    else:
                        st.error("No readable text found (scanned/image PDF?).")
                else:
                    st.success(f"Loaded: {uploaded.name}")
            elif st.session_state.pdf_filename:
                st.session_state.pdf_store = None
                st.session_state.pdf_filename = None
                set_pdf_store(None)

        with st.expander("Try these →", expanded=False):
            for ex in tool["examples"]:
                if st.button(ex, key=f"ex_{tool['tool']}_{ex}", use_container_width=True):
                    st.session_state.pending_prompt = ex

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True, type="secondary"):
        st.session_state.display_messages = []
        st.session_state.api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.pending_prompt = None
        st.session_state.pdf_store = None
        st.session_state.pdf_filename = None
        set_pdf_store(None)
        st.rerun()

    st.caption("Providers: OpenAI · Google Gemini · Meta via Groq · wttr.in · Yahoo Finance")


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("## Personal AI Assistant")
st.markdown(
    "Ask me anything — I'll search the web, solve math, check weather, "
    "look up stocks, or read your PDFs."
)
st.divider()

# Welcome screen
if not st.session_state.display_messages:
    st.markdown(
        '<div class="welcome-wrap">'
        '<div class="big-icon">👋</div>'
        "<h2>Hello! How can I help?</h2>"
        "<p>Pick an example from the sidebar or type your question below.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# Render conversation history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_tool_calls(msg.get("tool_calls", []))
        st.markdown(msg["content"])

# ── Handle input ──────────────────────────────────────────────────────────────
# Consume any pending example-button prompt, then let chat_input override it
prompt = st.session_state.pending_prompt
st.session_state.pending_prompt = None

typed = st.chat_input("Ask me anything…")
if typed:
    prompt = typed

if prompt:
    # Add and display user message
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    st.session_state.api_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent and stream response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                tool_calls_log, reply = run_agent(st.session_state.api_messages)
            except Exception as e:
                tool_calls_log = []
                reply = f"Sorry, I ran into an error: {e}"

        render_tool_calls(tool_calls_log)
        st.markdown(reply)

    # Persist to session state
    st.session_state.display_messages.append({
        "role": "assistant",
        "content": reply,
        "tool_calls": tool_calls_log,
    })
    # Append final assistant text turn to API history
    st.session_state.api_messages.append({"role": "assistant", "content": reply})
