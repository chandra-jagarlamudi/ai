"""
assistant.py - Personal AI Assistant (Streamlit UI)

This is the main application file. It handles:
  - The Streamlit web interface (sidebar, chat area)
  - Selecting the LLM model and provider (via litellm)
  - Processing PDF uploads into a searchable vector index
  - Running the "agent loop" — sending messages to the LLM and handling tool calls

Run with:
    streamlit run assistant.py
"""

import io       # For reading uploaded file bytes in memory
import json     # For parsing tool call arguments
import os       # For reading environment variables

import faiss    # Fast vector search (for PDF semantic search)
import litellm  # Unified LLM client — works with OpenAI, Gemini, Groq, Anthropic, etc.
import numpy as np  # Numerical arrays for vector math
import streamlit as st  # Web UI framework
from dotenv import load_dotenv  # Reads key=value pairs from a .env file into os.environ
from openai import OpenAI       # Used only for generating text embeddings (PDF feature)
from pypdf import PdfReader     # Extracts text from PDF files

import tools
from tools import dispatch_tool, get_tool_definitions, set_pdf_store

# Load environment variables from the .env file in this directory.
# After this line, os.getenv("API_KEY") etc. will return the values in .env.
load_dotenv()


# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
# This must be the first Streamlit call in the script.

st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Model / Provider Configuration
# ---------------------------------------------------------------------------
#
# This app uses litellm, which lets you swap LLM providers by just changing
# the model name prefix — no code changes needed.
#
# Examples:
#   "gpt-4o-mini"                    → OpenAI (default, no prefix)
#   "gemini/gemini-2.0-flash"        → Google Gemini
#   "groq/llama-3.3-70b-versatile"   → Meta Llama via Groq
#   "anthropic/claude-sonnet-4-6"    → Anthropic Claude
#
# Each provider expects a different environment variable for the API key.
# The map below tells us which env var to set for each model prefix.

_PROVIDER_KEY_MAP = {
    "gemini/":    "GEMINI_API_KEY",
    "groq/":      "GROQ_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "cohere/":    "COHERE_API_KEY",
    "mistral/":   "MISTRAL_API_KEY",
}


def _get_provider_name(model: str) -> str:
    """Return a human-readable provider name for display in the sidebar."""
    for prefix in _PROVIDER_KEY_MAP:
        if model.startswith(prefix):
            return prefix.rstrip("/").capitalize()
    return "OpenAI"  # Default when no prefix matches


def _apply_api_key(model: str, api_key: str) -> None:
    """
    Set the correct environment variable so litellm can authenticate.

    litellm reads the standard env var for each provider automatically.
    We just need to make sure the right one is set before calling litellm.completion().
    """
    for prefix, env_var in _PROVIDER_KEY_MAP.items():
        if model.startswith(prefix):
            os.environ[env_var] = api_key
            return
    # No prefix matched — assume OpenAI
    os.environ["OPENAI_API_KEY"] = api_key


# Read the API key and model from .env (or fall back to defaults)
_llm_api_key = os.getenv("API_KEY", "")
_default_model = os.getenv("MODEL", "gpt-4o-mini")

# Store the current model in Streamlit's session state so it persists across reruns.
# Session state is like a dictionary that survives Streamlit's top-to-bottom reruns.
if "model" not in st.session_state:
    st.session_state.model = _default_model

# Apply the API key every rerun (Streamlit reruns the whole script on every interaction)
if _llm_api_key:
    _apply_api_key(st.session_state.model, _llm_api_key)

# The OpenAI client is kept separately just for generating PDF embeddings.
# Even when using a non-OpenAI model for chat, we still use OpenAI's
# text-embedding-3-small to vectorize PDF chunks.
_openai_api_key = os.getenv("OPENAI_API_KEY", "")
_embed_client = OpenAI(api_key=_openai_api_key) if _openai_api_key else None


# ---------------------------------------------------------------------------
# Tool Metadata for the Sidebar UI
# ---------------------------------------------------------------------------
# This list drives the sidebar cards and example buttons.
# The "tool" key matches the function names in tools.py.

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

# Build a quick lookup: tool function name → its display color
# Example: {"web_search": "#4A90E2", "solve_math": "#9B59B6", ...}
TOOL_COLORS = {tool["tool"]: tool["color"] for tool in TOOLS_INFO}


# ---------------------------------------------------------------------------
# System Prompt — Instructions Given to the LLM at the Start of Every Chat
# ---------------------------------------------------------------------------
# This is the first message in every conversation (role: "system").
# It tells the LLM what it can do and how to behave.

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
  in plain text, e.g. "sin(pi/4) = 0.7071" or use a code block for multi-line results.
- For weather results, always reproduce the tool output EXACTLY and verbatim inside a markdown
  code block (``` ... ```) so the aligned formatting is preserved. Do not paraphrase or reformat it.
- For stock results, the UI renders a visual card automatically. Just give a brief one-sentence
  summary (e.g. "NVDA is trading at 183.34, up 0.09% from yesterday's close."). Do not list all
  the fields again — the card already shows them.
"""


# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------
# We inject CSS to style tool cards in the sidebar, tool usage badges,
# the stock info card, and the welcome screen.

st.markdown("""
<style>
/* Hide default Streamlit chrome (hamburger menu and footer) */
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

/* Tool usage badges (shown above the AI's reply when a tool was used) */
.tool-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.71rem;
    font-weight: 600;
    margin: 0 3px 6px 0;
    letter-spacing: 0.02em;
}

/* Stock info card */
.stock-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0 10px;
    font-family: monospace;
}
.stock-header { font-size: 0.95rem; font-weight: 700; margin-bottom: 10px; opacity: 0.9; }
.stock-header .sticker { opacity: 0.6; margin-left: 4px; font-weight: 400; }
.stock-header .scurrency { opacity: 0.45; font-size: 0.8rem; margin-left: 8px; }
.stock-price-row { display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px; }
.stock-price-row .sprice { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; }
.stock-price-row .schange { font-size: 0.85rem; font-weight: 600; }
.up   { color: #2ECC71; }  /* green for price up */
.down { color: #E74C3C; }  /* red for price down */
.flat { color: #95A5A6; }  /* grey for no change */
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

/* Welcome screen (shown before any messages) */
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


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
#
# Streamlit reruns the entire script from top to bottom on every user action.
# st.session_state is a persistent dictionary that keeps values between reruns.
#
# We maintain TWO separate message histories:
#
#   display_messages  — What we show in the chat UI.
#                       Includes tool call details for the expanders.
#                       Format: [{role, content, tool_calls: [{name, args, result}]}]
#
#   api_messages      — The full conversation history we send to the LLM.
#                       Includes "tool" role messages with raw results.
#                       Format follows the OpenAI chat format.
#
# Why two? The API needs the raw tool messages so the LLM has full context.
# The display messages are cleaned up for the user-facing chat bubbles.

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "api_messages" not in st.session_state:
    # Every conversation starts with the system prompt as the first message
    st.session_state.api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if "pending_prompt" not in st.session_state:
    # Stores a prompt queued by clicking an example button, consumed in the next rerun
    st.session_state.pending_prompt = None

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = None  # dict with FAISS index + chunks, or None

if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None  # tracks which file is currently indexed

# Sync the PDF store from session state into the tools module.
# Because Streamlit reruns the whole script, tools.py loses the in-memory store —
# we re-inject it here on every rerun.
set_pdf_store(st.session_state.pdf_store)


# ---------------------------------------------------------------------------
# PDF Processing: Upload → Chunks → Embeddings → FAISS Index
# ---------------------------------------------------------------------------
#
# When a user uploads a PDF, we:
#   1. Extract all text from every page using PdfReader
#   2. Split the text into overlapping chunks (to avoid losing context at boundaries)
#   3. Send all chunks to OpenAI's embedding model to get 1536-dim float vectors
#   4. Normalize vectors and load them into a FAISS index for fast similarity search
#
# Overlap: if chunk size is 800 and overlap is 100, chunks start at:
#   0, 700, 1400, 2100, ...
# This means the end of chunk N overlaps with the start of chunk N+1,
# so no sentence or idea is cut off at a chunk boundary.

EMBED_MODEL = "text-embedding-3-small"  # OpenAI embedding model
EMBED_DIM = 1536        # Output vector dimensions for this model
CHUNK_SIZE = 800        # Characters per chunk
CHUNK_OVERLAP = 100     # Overlapping characters between adjacent chunks


def process_pdf(file_bytes: bytes, filename: str) -> dict | None:
    """
    Process a PDF file into a FAISS searchable vector store.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file
        filename:   Original filename (stored for display purposes)

    Returns:
        A store dict with keys: name, chunks, index, num_chunks
        Returns None if the PDF has no readable text or OPENAI_API_KEY is missing.
    """
    # Step 1: Extract text from every page of the PDF
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            full_text += page_text + "\n\n"

    if not full_text.strip():
        return None  # No readable text (might be a scanned image PDF)

    # Step 2: Split text into overlapping chunks
    chunks = []
    start = 0
    while start < len(full_text):
        chunks.append(full_text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP  # advance by (chunk - overlap) each step

    # Step 3: Embed all chunks with OpenAI (requires OPENAI_API_KEY)
    if _embed_client is None:
        return None  # Caller shows an error message in the UI

    embedding_response = _embed_client.embeddings.create(input=chunks, model=EMBED_MODEL)
    embeddings = np.array([e.embedding for e in embedding_response.data], dtype=np.float32)

    # Step 4: Normalize vectors so that dot product = cosine similarity
    # Each row is divided by its length (norm), making all vectors unit length.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-10)  # avoid division by zero with small epsilon

    # Step 5: Build a FAISS index and add all chunk vectors
    # IndexFlatIP = "Flat Inner Product" — exact search, no approximation
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    return {
        "name":       filename,
        "chunks":     chunks,
        "index":      index,
        "num_chunks": len(chunks),
    }


# ---------------------------------------------------------------------------
# Agent Loop — The Core AI Engine
# ---------------------------------------------------------------------------
#
# This is an "agentic loop" — the LLM can call tools repeatedly until it
# has enough information to give the user a final answer.
#
# Flow:
#   1. Send all messages to the LLM
#   2. If the LLM returns tool_calls → execute each tool → add results to messages → goto 1
#   3. If the LLM returns a text reply (no tool_calls) → that's the final answer, exit loop
#
# Example for "What's the weather in Paris and also solve sqrt(25)?":
#   Round 1: LLM returns 2 tool calls: get_weather("Paris"), solve_math("sqrt(25)")
#            → We run both, add results to messages
#   Round 2: LLM returns a final text answer combining both results
#            → Loop exits, we display the reply

def run_agent(api_messages: list) -> tuple[list, str]:
    """
    Run the agentic loop until the LLM produces a final text reply.

    Mutates api_messages in-place by appending tool call and result messages.
    This keeps the full conversation history for the next user turn.

    Args:
        api_messages: The current full conversation history (system + user + assistant turns)

    Returns:
        (tool_calls_log, final_reply)
        tool_calls_log: list of dicts with name, args, result for each tool used
        final_reply:    the LLM's final text answer
    """
    tool_definitions = get_tool_definitions()
    tool_calls_log = []  # Collect all tool uses for display in the UI

    while True:
        # Ask the LLM what to do next (either call tools or give the final answer)
        response = litellm.completion(
            model=st.session_state.model,
            messages=api_messages,
            tools=tool_definitions,
            tool_choice="auto",   # let the LLM decide whether to call a tool
        )
        message = response.choices[0].message

        # No tool calls in the response → this is the final answer, exit the loop
        if not message.tool_calls:
            return tool_calls_log, message.content or ""

        # The LLM wants to call one or more tools.
        # Add the assistant's intent to the message history before executing.
        # We convert to a plain dict because litellm's object isn't JSON-serializable.
        api_messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        # Execute each requested tool and add the results to the message history
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments_json = tool_call.function.arguments

            # Call the actual Python function via dispatch_tool()
            result = dispatch_tool(tool_name, arguments_json)

            # Parse args for display in the UI (fall back to raw string if invalid JSON)
            try:
                args_for_display = json.loads(arguments_json)
            except Exception:
                args_for_display = arguments_json

            # Log this tool call for the sidebar badges + expandable detail blocks
            tool_calls_log.append({
                "name":   tool_name,
                "args":   args_for_display,
                "result": result,
            })

            # Add the tool result to message history so the LLM can see what it got back
            api_messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      result,
            })
        # Loop back to send messages + results to the LLM for the next round


# ---------------------------------------------------------------------------
# Stock Card Renderer
# ---------------------------------------------------------------------------
# get_stock_info() returns JSON. Instead of showing raw JSON to the user,
# we parse it and render a styled HTML card with price, change, and key metrics.

def render_stock_card(result_json: str) -> None:
    """
    Parse the JSON returned by get_stock_info and render a styled stock card.
    Falls back to plain text if the JSON is invalid or contains an error.
    """
    try:
        data = json.loads(result_json)
    except Exception:
        st.text(result_json)  # can't parse, just show raw text
        return

    if "error" in data:
        st.error(data["error"])
        return

    price = data.get("price")
    prev_close = data.get("prev_close")

    # Determine if price went up, down, or stayed flat compared to yesterday
    if price and prev_close:
        change = price - prev_close
        change_pct = (change / prev_close) * 100
        if change > 0:
            css_class, arrow = "up",   "▲"
        elif change < 0:
            css_class, arrow = "down", "▼"
        else:
            css_class, arrow = "flat", "—"
        change_display = f"{arrow} {abs(change):.2f} ({abs(change_pct):.2f}%)"
    else:
        css_class, change_display = "flat", ""

    def fmt(value, decimals=2, prefix="", suffix=""):
        """Format a number nicely, or return 'N/A' if the value is None."""
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{prefix}{value:,.{decimals}f}{suffix}"
        if isinstance(value, int):
            return f"{prefix}{value:,}{suffix}"
        return str(value)

    # Build the rows for the lower section of the card
    metric_rows = [
        ("Previous Close", fmt(prev_close)),
        ("Day Range",      f"{fmt(data.get('day_low'))} – {fmt(data.get('day_high'))}"),
        ("52-Week Range",  f"{fmt(data.get('week_52_low'))} – {fmt(data.get('week_52_high'))}"),
        ("Volume",         fmt(data.get("volume"), decimals=0)),
        ("Market Cap",     data.get("market_cap") or "N/A"),
        ("Sector",         data.get("sector") or "N/A"),
        ("P/E Ratio",      fmt(data.get("pe"))),
        ("EPS",            fmt(data.get("eps"))),
        ("Dividend Yield", data.get("dividend_yield") or "N/A"),
    ]

    # Build each row as an HTML div
    rows_html = "".join(
        f'<div class="srow">'
        f'<span class="slabel">{label}</span>'
        f'<span class="svalue">{value}</span>'
        f'</div>'
        for label, value in metric_rows
    )

    # Assemble the full card HTML
    card_html = f"""
    <div class="stock-card">
      <div class="stock-header">
        {data.get("name", data.get("ticker", ""))}
        <span class="sticker">({data.get("ticker", "")})</span>
        <span class="scurrency">{data.get("currency", "USD")}</span>
      </div>
      <div class="stock-price-row">
        <span class="sprice {css_class}">{fmt(price)}</span>
        <span class="schange {css_class}">{change_display}</span>
      </div>
      {rows_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tool Call Display — Badges + Expandable Detail Blocks
# ---------------------------------------------------------------------------

def render_tool_calls(tool_calls: list) -> None:
    """
    Render the tool usage indicators above the AI's reply:
      - Colored badge for each tool that was called
      - Expandable section showing the arguments and raw result
      - Stock tool gets an inline styled card instead of an expander

    Args:
        tool_calls: list of dicts with keys: name, args, result
    """
    if not tool_calls:
        return

    # Build a row of colored pill badges (one per tool call)
    badges_html = ""
    for tc in tool_calls:
        color = TOOL_COLORS.get(tc["name"], "#888")
        label = tc["name"].replace("_", " ").title()
        badges_html += (
            f'<span class="tool-badge" '
            f'style="background:{color}1a;color:{color};border:1px solid {color}55;">'
            f'{label}</span>'
        )
    st.markdown(badges_html, unsafe_allow_html=True)

    # Render each tool's result
    for tc in tool_calls:
        color = TOOL_COLORS.get(tc["name"], "#888")
        label = tc["name"].replace("_", " ").title()

        if tc["name"] == "get_stock_info":
            # Stock gets a rich visual card — no expander needed
            render_stock_card(tc["result"])
            continue

        # All other tools: show args + result in a collapsible expander
        with st.expander(f"🔧 {label}", expanded=False):
            col_args, col_result = st.columns([1, 2])
            with col_args:
                st.markdown("**Arguments**")
                st.json(tc["args"], expanded=True)
            with col_result:
                st.markdown("**Result**")
                # Truncate very long results to avoid overwhelming the UI
                preview = tc["result"]
                if len(preview) > 1500:
                    preview = preview[:1500] + "\n\n...(truncated)"
                st.text(preview)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ Personal AI")

    # Model selector — user can type any litellm-compatible model string
    new_model = st.text_input("Model", value=st.session_state.model).strip()
    if new_model and new_model != st.session_state.model:
        # Model changed — update session state and re-apply the API key
        st.session_state.model = new_model
        if _llm_api_key:
            _apply_api_key(new_model, _llm_api_key)

    st.caption(f"Provider: {_get_provider_name(st.session_state.model)}")

    if not _llm_api_key:
        st.warning("⚠️ `API_KEY` not set in `.env`")

    st.divider()
    st.markdown("#### What I can do")

    # Render a card for each tool in the sidebar
    for tool in TOOLS_INFO:
        color = tool["color"]
        st.markdown(
            f'<div class="tool-card" style="--card-color:{color};">'
            f'<p class="card-title">{tool["icon"]} {tool["name"]}</p>'
            f'<p class="card-desc">{tool["description"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # The PDF Search card has an embedded file uploader
        if tool["tool"] == "search_pdf":
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                key="pdf_uploader",
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                if st.session_state.pdf_filename != uploaded_file.name:
                    # New file uploaded — process it into a FAISS index
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        store = process_pdf(uploaded_file.read(), uploaded_file.name)

                    if store:
                        st.session_state.pdf_store = store
                        st.session_state.pdf_filename = uploaded_file.name
                        set_pdf_store(store)  # share with tools.py
                        st.success(f"Ready — {store['num_chunks']} chunks indexed")
                    elif _embed_client is None:
                        st.error("PDF indexing requires `OPENAI_API_KEY` (used for embeddings).")
                    else:
                        st.error("No readable text found (scanned/image PDF?).")
                else:
                    # Same file as before — already indexed, nothing to do
                    st.success(f"Loaded: {uploaded_file.name}")

            elif st.session_state.pdf_filename:
                # File was removed from the uploader — clear the store
                st.session_state.pdf_store = None
                st.session_state.pdf_filename = None
                set_pdf_store(None)

        # Example prompt buttons for each tool
        with st.expander("Try these →", expanded=False):
            for example in tool["examples"]:
                if st.button(example, key=f"ex_{tool['tool']}_{example}", use_container_width=True):
                    # Queue this example as the next prompt to process
                    st.session_state.pending_prompt = example

    st.divider()

    # Clear button — resets the full conversation and PDF state
    if st.button("🗑️ Clear conversation", use_container_width=True, type="secondary"):
        st.session_state.display_messages = []
        st.session_state.api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.pending_prompt = None
        st.session_state.pdf_store = None
        st.session_state.pdf_filename = None
        set_pdf_store(None)
        st.rerun()

    st.caption("Providers: OpenAI · Google Gemini · Meta via Groq · wttr.in · Yahoo Finance")


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.markdown("## Personal AI Assistant")
st.markdown(
    "Ask me anything — I'll search the web, solve math, check weather, "
    "look up stocks, or read your PDFs."
)
st.divider()

# Welcome screen — only shown when there are no messages yet
if not st.session_state.display_messages:
    st.markdown(
        '<div class="welcome-wrap">'
        '<div class="big-icon">👋</div>'
        "<h2>Hello! How can I help?</h2>"
        "<p>Pick an example from the sidebar or type your question below.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# Render all past messages in order
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Show tool usage badges and expandable result blocks above the reply
            render_tool_calls(msg.get("tool_calls", []))
        st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Handle User Input
# ---------------------------------------------------------------------------
#
# Two ways the user can send a prompt:
#   1. Type in the chat_input box at the bottom
#   2. Click an example button in the sidebar (sets pending_prompt)
#
# We check pending_prompt first, then let a typed message override it.

# Consume any pending example prompt from the sidebar buttons
prompt = st.session_state.pending_prompt
st.session_state.pending_prompt = None  # clear it immediately so it doesn't repeat

# If the user also typed something, that takes priority
typed_input = st.chat_input("Ask me anything...")
if typed_input:
    prompt = typed_input

if prompt:
    # Add the user's message to both histories
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    st.session_state.api_messages.append({"role": "user", "content": prompt})

    # Display the user's message in the chat UI immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent and display the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                tool_calls_log, reply = run_agent(st.session_state.api_messages)
            except Exception as e:
                # Something unexpected went wrong — show a friendly error instead of crashing
                tool_calls_log = []
                reply = f"Sorry, I ran into an error: {e}"

        # Show tool usage badges and any expandable tool results
        render_tool_calls(tool_calls_log)
        # Show the AI's final text reply
        st.markdown(reply)

    # Save the assistant's response to the display history (for future reruns)
    st.session_state.display_messages.append({
        "role":       "assistant",
        "content":    reply,
        "tool_calls": tool_calls_log,
    })

    # Also append the final text reply to the API history so the LLM has full context
    st.session_state.api_messages.append({"role": "assistant", "content": reply})
