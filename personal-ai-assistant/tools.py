"""
tools.py — All tool implementations for the personal AI assistant.

Each function maps to an OpenAI function-calling tool definition returned
by get_tool_definitions() at the bottom of this file.
"""

import ast
import json
import math
import operator
import os
import tempfile
from typing import Any

import numpy as np
import faiss
import requests
import yfinance as yf
from ddgs import DDGS
from openai import OpenAI as _OpenAI
from pypdf import PdfReader

# Public API — everything assistant.py (or any other consumer) should import from here.
__all__ = [
    # Individual tool functions (call directly or via dispatch_tool)
    "web_search",
    "solve_math",
    "get_weather",
    "get_stock_info",
    "search_pdf",
    # PDF store management (called by assistant.py after upload / on clear)
    "set_pdf_store",
    # OpenAI integration helpers
    "get_tool_definitions",   # returns the list[dict] of OpenAI tool schemas
    "dispatch_tool",          # routes a tool-call name + JSON args to the right function
    "TOOL_REGISTRY",          # dict[str, callable] — useful for introspection / testing
]

# ---------------------------------------------------------------------------
# PDF vector store — set by assistant.py after the user uploads a PDF.
# Holds: {"name", "chunks", "index" (faiss), "num_chunks"}
# ---------------------------------------------------------------------------
_pdf_store: dict | None = None
_embed_client: _OpenAI | None = None

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536


def set_pdf_store(store: dict | None) -> None:
    """Called by assistant.py to inject (or clear) the active PDF store."""
    global _pdf_store
    _pdf_store = store


def _get_embed_client() -> _OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _embed_client


# ---------------------------------------------------------------------------
# 1. DuckDuckGo Web Search
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return a formatted summary."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['body']}")
            lines.append(f"   Source: {r['href']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


# ---------------------------------------------------------------------------
# 2. Python Math Solver
# ---------------------------------------------------------------------------

# Safe names available inside math expressions
_MATH_SAFE_NAMES: dict[str, Any] = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}
_MATH_SAFE_NAMES.update({"abs": abs, "round": round, "int": int, "float": float})

_SAFE_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.UAdd, ast.USub, ast.Load,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _safe_eval(expression: str) -> Any:
    """Parse and evaluate a math expression in a restricted, sandboxed environment.

    Supported operations
    --------------------
    Arithmetic:
        +  -  *  /  //  %  **          (add, sub, mul, div, floor-div, mod, power)
        -x  +x                          (unary negation / unary plus)

    Comparisons (return True/False):
        ==  !=  <  <=  >  >=

    Constants (from math module):
        pi (~3.14159)   e (~2.71828)   tau (~6.28318)   inf   nan

    Rounding / conversion:
        abs(x)   round(x)   int(x)   float(x)

    Powers & roots:
        sqrt(x)   pow(x, y)   exp(x)   log(x)   log2(x)   log10(x)

    Trigonometry (angles in radians):
        sin(x)  cos(x)  tan(x)
        asin(x) acos(x) atan(x)  atan2(y, x)
        degrees(x)   radians(x)

    Hyperbolic:
        sinh(x)  cosh(x)  tanh(x)
        asinh(x) acosh(x) atanh(x)

    Rounding variants:
        ceil(x)   floor(x)   trunc(x)

    Combinatorics:
        factorial(n)      n!
        comb(n, k)        n choose k
        perm(n, k)        permutations of k from n

    Number theory / geometry / misc:
        gcd(a, b)         greatest common divisor
        lcm(a, b)         least common multiple
        hypot(*coords)    Euclidean distance, e.g. hypot(3, 4) → 5.0
        dist(p, q)        Euclidean distance between two point sequences
        fabs(x)           floating-point absolute value
        fmod(x, y)        floating-point remainder
        copysign(x, y)    magnitude of x with sign of y
        isfinite(x)  isinf(x)  isnan(x)

    NOT supported (raises ValueError):
        Variable assignment, imports, function definitions (def / lambda),
        subscripts/slices, list/dict/set literals, attribute access, and
        any built-in not in {abs, round, int, float}.
    """
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")
    return eval(  # noqa: S307  (safe — restricted AST checked above)
        compile(tree, "<expr>", "eval"),
        {"__builtins__": {}},
        _MATH_SAFE_NAMES,
    )


def solve_math(*expressions: str) -> str:
    """
    Evaluate one or more mathematical expressions.
    Accepts standard Python math syntax plus all functions from the math module
    (sin, cos, sqrt, log, factorial, …).
    """
    results = []
    for expr in expressions:
        expr = expr.strip()
        try:
            result = _safe_eval(expr)
            results.append(f"{expr} = {result}")
        except Exception as e:
            results.append(f"{expr} → Error: {e}")
    return "\n".join(results)


# ---------------------------------------------------------------------------
# 3. Weather (wttr.in — no API key required)
# ---------------------------------------------------------------------------

def get_weather(location: str) -> str:
    """
    Fetch current weather for a city/location using wttr.in (no API key needed).
    Returns a human-readable summary.
    """
    try:
        url = f"https://wttr.in/{requests.utils.quote(location)}?format=j1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data["current_condition"][0]
        area = data["nearest_area"][0]

        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        feels_c = current["FeelsLikeC"]
        feels_f = current["FeelsLikeF"]
        humidity = current["humidity"]
        wind_kmph = current["windspeedKmph"]
        wind_dir = current["winddir16Point"]
        visibility = current["visibility"]
        uv_index = current["uvIndex"]

        return (
            f"Weather in {city}, {country}:\n"
            f"  Condition   : {desc}\n"
            f"  Temperature : {temp_c}°C / {temp_f}°F  (feels like {feels_c}°C / {feels_f}°F)\n"
            f"  Humidity    : {humidity}%\n"
            f"  Wind        : {wind_kmph} km/h {wind_dir}\n"
            f"  Visibility  : {visibility} km\n"
            f"  UV Index    : {uv_index}"
        )
    except Exception as e:
        return f"Could not fetch weather for '{location}': {e}"


# ---------------------------------------------------------------------------
# 4. Stock Quote (yfinance — no API key required)
# ---------------------------------------------------------------------------

def get_stock_info(ticker: str) -> str:
    """
    Fetch current price, key financials, and company info for a stock ticker
    using yfinance (Yahoo Finance — no API key required).
    Returns a JSON string so the UI can render a styled card and the model
    can still reference the values in its reply.
    """
    try:
        t = ticker.strip().upper()
        stock = yf.Ticker(t)
        info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        market_cap_raw = info.get("marketCap")

        if market_cap_raw:
            if market_cap_raw >= 1e12:
                market_cap = f"${market_cap_raw / 1e12:.2f}T"
            else:
                market_cap = f"${market_cap_raw / 1e9:.2f}B"
        else:
            market_cap = None

        div_yield = info.get("dividendYield")

        data = {
            "ticker": t,
            "name": info.get("longName", t),
            "currency": info.get("currency", "USD"),
            "price": price,
            "prev_close": prev_close,
            "day_low": info.get("dayLow"),
            "day_high": info.get("dayHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "volume": info.get("volume"),
            "market_cap": market_cap,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "pe": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": f"{div_yield * 100:.2f}%" if div_yield else None,
        }
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": f"Could not fetch data for ticker '{ticker}': {e}"})


# ---------------------------------------------------------------------------
# 5. PDF Semantic Search (FAISS vector store, set via set_pdf_store())
# ---------------------------------------------------------------------------

def search_pdf(question: str, top_k: int = 5) -> str:
    """
    Retrieve the most relevant chunks from the currently loaded PDF and return
    them so the model can answer the user's question.

    The PDF must be uploaded via the sidebar first; set_pdf_store() is called
    by assistant.py once the FAISS index is built.
    """
    if _pdf_store is None:
        return (
            "No PDF is currently loaded. "
            "Please upload a PDF using the file uploader in the sidebar."
        )

    client = _get_embed_client()

    # Embed the question with the same model used at index time
    resp = client.embeddings.create(input=[question], model=EMBED_MODEL)
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    q_emb /= np.linalg.norm(q_emb)          # normalise for cosine similarity
    q_emb = q_emb.reshape(1, -1)

    scores, indices = _pdf_store["index"].search(q_emb, top_k)

    chunks = _pdf_store["chunks"]
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx < 0:
            continue
        results.append(f"[Chunk {rank} · relevance {score:.3f}]\n{chunks[idx]}")

    if not results:
        return "No relevant content found in the PDF for that question."

    header = f"PDF: {_pdf_store['name']} ({_pdf_store['num_chunks']} chunks total)\n\n"
    return header + "\n\n---\n\n".join(results)


# ---------------------------------------------------------------------------
# OpenAI Tool Definitions
# ---------------------------------------------------------------------------

def get_tool_definitions() -> list[dict]:
    """Return the OpenAI-format tool schemas for all tools above."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web using DuckDuckGo for current, factual information. "
                    "Use this for news, recent events, or any question that benefits from "
                    "live web results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of results to return (default 5, max 10).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "solve_math",
                "description": (
                    "Evaluate one or more mathematical expressions using Python math. "
                    "Supports arithmetic, trigonometry (sin, cos, tan), logarithms (log, log10), "
                    "square root (sqrt), factorial, constants (pi, e), and more. "
                    "Pass multiple expressions to solve several problems at once."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expressions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of math expressions to evaluate, e.g. "
                                '[\"2**10\", \"sqrt(144)\", \"sin(pi/2)\"]'
                            ),
                        },
                    },
                    "required": ["expressions"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": (
                    "Get current weather conditions for any city or location. "
                    "Returns temperature, humidity, wind speed, UV index, and more. "
                    "No API key required."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or location, e.g. 'London' or 'New York'.",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_info",
                "description": (
                    "Fetch real-time stock price, market cap, P/E ratio, EPS, dividend yield, "
                    "52-week range, and company info for any publicly traded stock. "
                    "Uses Yahoo Finance — no API key required."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'MSFT'.",
                        },
                    },
                    "required": ["ticker"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_pdf",
                "description": (
                    "Semantically search the currently loaded PDF document and retrieve the most "
                    "relevant passages to answer the user's question. The PDF is uploaded by the "
                    "user via the sidebar — do NOT ask them for a file path. "
                    "Call this whenever the user asks anything about their PDF."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The specific question to find relevant content for.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to retrieve (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["question"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tool Dispatcher — called by the assistant to route tool calls
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "web_search":    lambda args: web_search(**args),
    "solve_math":    lambda args: solve_math(*args["expressions"]),
    "get_weather":   lambda args: get_weather(**args),
    "get_stock_info": lambda args: get_stock_info(**args),
    "search_pdf":    lambda args: search_pdf(**args),
}


def dispatch_tool(name: str, arguments: str) -> str:
    """Parse JSON arguments and call the matching tool function."""
    if name not in TOOL_REGISTRY:
        return f"Unknown tool: {name}"
    try:
        args = json.loads(arguments)
        return TOOL_REGISTRY[name](args)
    except Exception as e:
        return f"Tool '{name}' raised an error: {e}"
