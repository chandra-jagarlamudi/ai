"""
tools.py - Tool implementations for the Personal AI Assistant

Each function here is a "tool" the AI can call to get real information.
When the user asks "What's the weather in Paris?", the LLM doesn't guess —
it calls get_weather("Paris") here and returns real data.

Tools available:
  1. web_search     - Search DuckDuckGo for live web results
  2. solve_math     - Safely evaluate math expressions
  3. get_weather    - Get current weather for any city
  4. get_stock_info - Get live stock price and company data
  5. search_pdf     - Semantically search an uploaded PDF

At the bottom of this file:
  - get_tool_definitions() - Tells the LLM what tools exist and how to call them
  - dispatch_tool()        - Routes a tool call by name to the right function
"""

import ast        # Used to safely parse math expressions (prevents code injection)
import json       # For encoding/decoding tool results as JSON strings
import math       # Python's built-in math functions (sin, cos, sqrt, etc.)
import os         # For reading environment variables like OPENAI_API_KEY
from typing import Any

import faiss      # Facebook's fast vector search library (used for PDF search)
import numpy as np  # Numerical arrays, needed for FAISS vector operations
import requests   # HTTP client for weather API calls
import yfinance as yf  # Yahoo Finance wrapper for stock data
from ddgs import DDGS   # DuckDuckGo Search client
from openai import OpenAI as _OpenAI  # Used only for generating text embeddings (PDF feature)

# This tells Python what names are "public" when someone does `from tools import *`
__all__ = [
    "web_search",
    "solve_math",
    "get_weather",
    "get_stock_info",
    "search_pdf",
    "set_pdf_store",
    "get_tool_definitions",
    "dispatch_tool",
    "TOOL_REGISTRY",
]


# ---------------------------------------------------------------------------
# PDF Store — Shared State Between assistant.py and tools.py
# ---------------------------------------------------------------------------
#
# When a user uploads a PDF in the sidebar, assistant.py processes it into a
# "store" dictionary containing:
#   - "name"       : original filename
#   - "chunks"     : list of text pieces (each ~800 characters)
#   - "index"      : FAISS vector index for similarity search
#   - "num_chunks" : total number of chunks
#
# assistant.py calls set_pdf_store(store) to make it available here.
# If no PDF is loaded, _pdf_store stays None and search_pdf returns an error.

_pdf_store: dict | None = None

# OpenAI client for generating embeddings — created lazily (only when needed)
_embed_client: _OpenAI | None = None

# Embedding model and its output dimension.
# "text-embedding-3-small" produces 1536-dimensional float vectors.
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def set_pdf_store(store: dict | None) -> None:
    """
    Called by assistant.py to share the active PDF store with this module.
    Pass None to clear it (e.g., when the user removes the PDF or clears chat).
    """
    global _pdf_store
    _pdf_store = store


def _get_embed_client() -> _OpenAI:
    """
    Returns a cached OpenAI client for generating embeddings.
    We only create it once and reuse it (lazy initialization).
    """
    global _embed_client
    if _embed_client is None:
        _embed_client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _embed_client


# ---------------------------------------------------------------------------
# Tool 1: Web Search via DuckDuckGo
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.

    DuckDuckGo is used because it doesn't require an API key.
    Results include a title, a short snippet, and the source URL.

    Args:
        query:       The search query string, e.g. "latest Python news"
        max_results: How many results to return (default 5)

    Returns:
        A formatted string with numbered results, or an error message.
    """
    try:
        with DDGS() as ddgs:
            # Fetch results — DDGS returns a list of dicts with keys:
            # 'title', 'body' (snippet), and 'href' (URL)
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found."

        # Format each result as:
        #   1. **Title**
        #      Snippet text
        #      Source: https://...
        lines = []
        for i, result in enumerate(results, start=1):
            lines.append(f"{i}. **{result['title']}**")
            lines.append(f"   {result['body']}")
            lines.append(f"   Source: {result['href']}")

        return "\n".join(lines)

    except Exception as e:
        return f"Search failed: {e}"


# ---------------------------------------------------------------------------
# Tool 2: Safe Math Evaluator
# ---------------------------------------------------------------------------
#
# Problem: Python's eval() is dangerous — it can run arbitrary code.
# Example: eval("__import__('os').system('rm -rf /')") would delete files!
#
# Solution: We use Python's AST (Abstract Syntax Tree) module to:
#   1. Parse the expression into a tree of nodes
#   2. Walk every node and check it is in our safe whitelist
#   3. Only then evaluate it
#
# This means things like imports, function definitions, and attribute access
# are all blocked — only pure math operations are allowed.

# These are the only math functions accessible inside expressions.
# We pull everything from Python's math module (sin, cos, sqrt, log, etc.)
# and add a few basic built-ins that are safe.
_MATH_SAFE_NAMES: dict[str, Any] = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")   # skip private/dunder names like __doc__
}
_MATH_SAFE_NAMES.update({
    "abs":   abs,
    "round": round,
    "int":   int,
    "float": float,
})

# These are the only AST node types we allow.
# Any node type not in this tuple will raise a ValueError.
_SAFE_AST_NODES = (
    ast.Expression,                                 # top-level expression wrapper
    ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant, ast.Name,  # operations and values
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,  # arithmetic
    ast.UAdd, ast.USub,                             # unary +x and -x
    ast.Load,                                       # variable lookup (read-only)
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,  # comparisons
)


def _safe_eval(expression: str) -> Any:
    """
    Parse and evaluate a math expression safely.

    Steps:
      1. Parse expression into an AST (no execution yet)
      2. Walk every node — reject anything outside the safe whitelist
      3. Compile and evaluate with a restricted namespace (no builtins)

    Raises:
        ValueError: if the expression contains a disallowed AST node
        Any math error (ZeroDivisionError, OverflowError, etc.) bubbles up naturally
    """
    # Step 1: Parse into a tree without running anything
    tree = ast.parse(expression, mode="eval")

    # Step 2: Inspect every single node in the tree
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_AST_NODES):
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    # Step 3: Execute with "__builtins__": {} so built-ins like open(), exec() are hidden
    return eval(  # noqa: S307  (intentionally safe — AST already validated above)
        compile(tree, "<expr>", "eval"),
        {"__builtins__": {}},   # completely empty builtins namespace
        _MATH_SAFE_NAMES,       # only our math functions are available
    )


def solve_math(*expressions: str) -> str:
    """
    Evaluate one or more math expressions and return the results.

    Supports standard Python math syntax:
        Arithmetic:   2 + 3, 10 / 4, 2**8, 15 % 7
        Functions:    sqrt(144), sin(pi/2), log(100), factorial(5)
        Constants:    pi, e, tau, inf

    Args:
        *expressions: One or more expression strings, e.g. "sqrt(144)", "2**10"

    Returns:
        A string with each expression and its result, one per line.
        Errors are reported inline so other expressions still evaluate.
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
# Tool 3: Current Weather via wttr.in
# ---------------------------------------------------------------------------

def get_weather(location: str) -> str:
    """
    Fetch the current weather for a city or location.

    Uses the free wttr.in API — no API key required.
    Returns a human-readable formatted block with all key conditions.

    Args:
        location: City name or location string, e.g. "London" or "New York"

    Returns:
        A formatted weather summary string, or an error message.
    """
    try:
        # wttr.in returns structured JSON when you add ?format=j1
        # We URL-encode the location so spaces and special chars are handled correctly
        url = f"https://wttr.in/{requests.utils.quote(location)}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raises an error if HTTP status is 4xx or 5xx
        data = response.json()

        # The JSON structure from wttr.in:
        # data["current_condition"][0]  → current weather details
        # data["nearest_area"][0]       → location metadata
        current = data["current_condition"][0]
        area = data["nearest_area"][0]

        # Extract human-readable location info
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]

        # Extract weather condition details
        description = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        feels_like_c = current["FeelsLikeC"]
        feels_like_f = current["FeelsLikeF"]
        humidity = current["humidity"]
        wind_speed_kmph = current["windspeedKmph"]
        wind_direction = current["winddir16Point"]
        visibility = current["visibility"]
        uv_index = current["uvIndex"]

        # Return a neatly aligned text block
        return (
            f"Weather in {city}, {country}:\n"
            f"  Condition   : {description}\n"
            f"  Temperature : {temp_c}°C / {temp_f}°F  (feels like {feels_like_c}°C / {feels_like_f}°F)\n"
            f"  Humidity    : {humidity}%\n"
            f"  Wind        : {wind_speed_kmph} km/h {wind_direction}\n"
            f"  Visibility  : {visibility} km\n"
            f"  UV Index    : {uv_index}"
        )

    except Exception as e:
        return f"Could not fetch weather for '{location}': {e}"


# ---------------------------------------------------------------------------
# Tool 4: Stock Quote via Yahoo Finance
# ---------------------------------------------------------------------------

def get_stock_info(ticker: str) -> str:
    """
    Fetch real-time stock data for a given ticker symbol.

    Uses yfinance (Yahoo Finance) — no API key required.
    Returns a JSON string so the UI can render a styled stock card.

    Args:
        ticker: Stock symbol, e.g. "AAPL", "TSLA", "NVDA"

    Returns:
        JSON string with price, market cap, P/E ratio, and more.
        On error, returns JSON with an "error" key.
    """
    try:
        ticker_symbol = ticker.strip().upper()

        # yf.Ticker gives us a wrapper object; .info is a dict of company data
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        # currentPrice is preferred; regularMarketPrice is a fallback
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")

        # Convert raw market cap (e.g. 3_000_000_000_000) to a readable string
        market_cap_raw = info.get("marketCap")
        if market_cap_raw:
            if market_cap_raw >= 1e12:
                market_cap = f"${market_cap_raw / 1e12:.2f}T"   # Trillions
            else:
                market_cap = f"${market_cap_raw / 1e9:.2f}B"    # Billions
        else:
            market_cap = None

        # dividendYield comes as a decimal (e.g. 0.0052), we convert to percent
        div_yield = info.get("dividendYield")

        result = {
            "ticker":         ticker_symbol,
            "name":           info.get("longName", ticker_symbol),
            "currency":       info.get("currency", "USD"),
            "price":          price,
            "prev_close":     prev_close,
            "day_low":        info.get("dayLow"),
            "day_high":       info.get("dayHigh"),
            "week_52_low":    info.get("fiftyTwoWeekLow"),
            "week_52_high":   info.get("fiftyTwoWeekHigh"),
            "volume":         info.get("volume"),
            "market_cap":     market_cap,
            "sector":         info.get("sector"),
            "industry":       info.get("industry"),
            "pe":             info.get("trailingPE"),
            "eps":            info.get("trailingEps"),
            "dividend_yield": f"{div_yield * 100:.2f}%" if div_yield else None,
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Could not fetch data for ticker '{ticker}': {e}"})


# ---------------------------------------------------------------------------
# Tool 5: Semantic PDF Search (FAISS + OpenAI Embeddings)
# ---------------------------------------------------------------------------
#
# How semantic search works (high-level):
#   1. When a PDF is uploaded, every text chunk is converted to a vector
#      (an array of 1536 numbers) by OpenAI's embedding model.
#   2. These vectors are stored in a FAISS index.
#   3. When the user asks a question, the question is also converted to a vector.
#   4. FAISS finds the stored vectors closest to the question vector.
#   5. The matching text chunks are returned as context for the LLM to answer from.
#
# "Cosine similarity" means we measure the angle between vectors — chunks
# that talk about similar topics have vectors pointing in the same direction.

def search_pdf(question: str, top_k: int = 5) -> str:
    """
    Find the most relevant text chunks in the uploaded PDF for a given question.

    The PDF must be uploaded via the sidebar first. assistant.py handles
    the upload, chunking, embedding, and indexing — then calls set_pdf_store()
    to make the FAISS index available here.

    Args:
        question: The user's question, e.g. "What is the methodology?"
        top_k:    How many of the most relevant chunks to return (default 5)

    Returns:
        A string containing the top matching chunks with relevance scores.
        Returns an instruction to upload a PDF if none is loaded.
    """
    if _pdf_store is None:
        return (
            "No PDF is currently loaded. "
            "Please upload a PDF using the file uploader in the sidebar."
        )

    client = _get_embed_client()

    # Step 1: Convert the user's question into a vector using the same embedding model
    embedding_response = client.embeddings.create(input=[question], model=EMBED_MODEL)
    question_vector = np.array(embedding_response.data[0].embedding, dtype=np.float32)

    # Step 2: Normalize the vector so that FAISS inner product == cosine similarity
    # (Without normalization, longer vectors would dominate the search unfairly)
    question_vector /= np.linalg.norm(question_vector)
    question_vector = question_vector.reshape(1, -1)  # FAISS expects shape (1, 1536)

    # Step 3: Search the FAISS index for the top_k closest chunk vectors
    # scores:  similarity score for each match (higher = more relevant)
    # indices: which chunk numbers were matched
    scores, indices = _pdf_store["index"].search(question_vector, top_k)

    # Step 4: Build the result text from matching chunks
    chunks = _pdf_store["chunks"]
    result_parts = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            # FAISS returns -1 when there aren't enough chunks to fill top_k
            continue
        result_parts.append(f"[Chunk {rank} · relevance {score:.3f}]\n{chunks[idx]}")

    if not result_parts:
        return "No relevant content found in the PDF for that question."

    # Add a header line so the LLM knows which PDF was searched
    header = f"PDF: {_pdf_store['name']} ({_pdf_store['num_chunks']} chunks total)\n\n"
    return header + "\n\n---\n\n".join(result_parts)


# ---------------------------------------------------------------------------
# Tool Schemas — Tell the LLM What Tools Exist and How to Call Them
# ---------------------------------------------------------------------------
#
# The LLM doesn't call Python functions directly. Instead, the OpenAI API
# uses a "function calling" protocol:
#   - We send a list of tool definitions (JSON schemas) with each request
#   - The LLM decides if a tool is needed and replies with a structured call
#   - We execute the matching Python function and send the result back
#
# Each definition describes:
#   - "name":        matches the Python function name
#   - "description": helps the LLM decide when to use this tool
#   - "parameters":  JSON Schema defining the expected arguments

def get_tool_definitions() -> list[dict]:
    """
    Return the OpenAI-format tool schema list for all tools.
    This is sent to the LLM on every request so it knows what it can call.
    """
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
                                'List of math expressions to evaluate, e.g. '
                                '["2**10", "sqrt(144)", "sin(pi/2)"]'
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
# Tool Dispatcher — Routes LLM Tool Calls to the Right Python Function
# ---------------------------------------------------------------------------
#
# When the LLM wants to call a tool, it returns:
#   - name:      e.g. "web_search"
#   - arguments: a JSON string, e.g. '{"query": "Python news"}'
#
# dispatch_tool() parses the JSON and calls the matching function.
# TOOL_REGISTRY maps each tool name to a lambda that unpacks the args dict.

TOOL_REGISTRY = {
    "web_search":     lambda args: web_search(**args),
    "solve_math":     lambda args: solve_math(*args["expressions"]),  # unpack list as *args
    "get_weather":    lambda args: get_weather(**args),
    "get_stock_info": lambda args: get_stock_info(**args),
    "search_pdf":     lambda args: search_pdf(**args),
}


def dispatch_tool(tool_name: str, arguments_json: str) -> str:
    """
    Parse a JSON argument string and call the matching tool function.

    Args:
        tool_name:       Name of the tool to call, e.g. "web_search"
        arguments_json:  JSON string of arguments, e.g. '{"query": "AI news"}'

    Returns:
        The tool's return value as a string, or an error message.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"Unknown tool: {tool_name}"

    try:
        args = json.loads(arguments_json)
        return TOOL_REGISTRY[tool_name](args)
    except Exception as e:
        return f"Tool '{tool_name}' raised an error: {e}"
