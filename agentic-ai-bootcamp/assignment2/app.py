"""
Personal Financial AI Agent — Streamlit Application
-----------------------------------------------------
Layout:
    Left sidebar  : Tool descriptions and example queries
    Main area     : Chat interface with in-session message history

The agent is cached with st.cache_resource so it is initialised once
per browser session (not on every Streamlit re-run).

The AgentExecutor accepts {"input": str, "chat_history": list[BaseMessage]},
so chat history is rebuilt from st.session_state on each turn.
"""

import logging
import time
from typing import Any

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from agent import create_financial_agent

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Financial AI Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cache agent so it is not re-created on every Streamlit re-run ──────────────
@st.cache_resource(show_spinner="Loading financial agent...")
def get_agent():
    """Initialise and cache the LangChain AgentExecutor."""
    logger.info("Creating financial agent (cached)")
    return create_financial_agent()


# ── Session state initialisation ───────────────────────────────────────────────
if "messages" not in st.session_state:
    # Each message: {"role": "user" | "assistant", "content": str}
    st.session_state.messages = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Financial AI Agent")
    st.caption("Powered by LangChain · LiteLLM · Yahoo Finance")
    st.header("🔧 Available Tools")
     # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        logger.info("Chat history cleared by user")
        st.rerun()

    with st.expander("📊 EMI Calculator"):
        st.markdown(
            """
            Calculates your monthly loan installment.

            **Formula:** `EMI = P × r × (1+r)^n / ((1+r)^n - 1)`

            **Try:**
            - *"EMI for a ₹10,00,000 home loan at 8.5% for 20 years"*
            - *"Monthly payment on a $30,000 car loan at 6% for 5 years"*
            """
        )

    with st.expander("📈 SIP Calculator"):
        st.markdown(
            """
            Projects the future value of monthly SIP investments.

            **Formula:** `FV = P × ((1+r)^n − 1) / r`

            **Try:**
            - *"Invest ₹5,000/month at 12% for 10 years — how much will I get?"*
            - *"SIP returns for $200/month at 10% annual return over 25 years"*
            """
        )

    with st.expander("💼 Budget Planner"):
        st.markdown(
            """
            Allocates your income using the **50-30-20 rule**:
            - 50% Needs · 30% Wants · 20% Savings

            **Try:**
            - *"Plan my budget for a monthly salary of ₹80,000"*
            - *"How should I split a $6,000 monthly income?"*
            """
        )

    with st.expander("📉 Stock Price"):
        st.markdown(
            """
            Fetches live prices from Yahoo Finance (no API key needed).

            **Ticker formats:**
            - US: `AAPL`, `TSLA`, `MSFT`
            - India NSE: `RELIANCE.NS`, `TCS.NS`
            - Crypto: `BTC-USD`, `ETH-USD`

            **Try:**
            - *"What is Apple's current stock price?"*
            - *"Show me stock info for RELIANCE.NS"*
            """
        )

    with st.expander("🧮 Calculator"):
        st.markdown(
            """
            Evaluates math expressions using NumPy.

            **Supported:** `sqrt`, `log`, `log10`, `exp`, `sin`, `cos`,
            `tan`, `abs`, `ceil`, `floor`, `pi`, `e`

            **Try:**
            - *"What is sqrt(144) + 15% of 50,000?"*
            - *"Calculate log10(1000) divided by pi"*
            - *"2 to the power of 10 plus sqrt(625)"*
            """
        )

    st.caption(
        "Switch models by editing `LLM_MODEL` in your `.env` file."
    )


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("💬 Personal Financial AI Assistant")
st.caption("Ask me about loans, investments, budgets, stocks, or any maths problem.")

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input — returns None when the user hasn't typed anything
if user_input := st.chat_input("Ask a financial question…"):

    # 1. Show and save the user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    logger.info("User message: %s", user_input)

    # 2. Build chat_history from all previous turns (excluding the current one).
    #    Passed to the AgentExecutor so the LLM has full conversational context.
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # 3. Run the agent. ToolCallTracker hooks into LangChain callbacks to
    #    track which tools are called and how long each takes.
    #    A plain st.caption() shows the summary — static, not clickable.
    with st.chat_message("assistant"):

        class ToolCallTracker(BaseCallbackHandler):
            """
            Lightweight LangChain callback that tracks tool lifecycle events.
            Updates a spinner label while running; results are surfaced as
            a static caption after completion — no expandable widget.
            """

            def __init__(self, spinner_slot: Any) -> None:
                self.spinner_slot = spinner_slot  # st.empty used inside st.spinner
                self.calls: list[dict] = []       # {tool, elapsed}
                self._start: float = 0.0

            def on_tool_start(
                self, serialized: dict, input_str: str, **kwargs: Any
            ) -> None:
                """Called when the agent decides to invoke a tool."""
                tool_name = serialized.get("name", "unknown tool")
                self._start = time.perf_counter()
                self.spinner_slot.markdown(f"🔧 Calling **{tool_name}**…")
                logger.info("Tool started: %s | input: %s", tool_name, input_str)

            def on_tool_end(self, output: str, **kwargs: Any) -> None:
                """Called when the tool returns its result."""
                elapsed = time.perf_counter() - self._start
                tool_name = kwargs.get("name", "tool")
                self.calls.append({"tool": tool_name, "elapsed": elapsed})
                logger.info("Tool finished: %s in %.2fs", tool_name, elapsed)

            def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
                """Called when the tool raises an exception."""
                logger.error("Tool error: %s", error)

        summary_slot = st.empty()   # will hold the static caption after the run
        answer = ""

        with st.spinner("Thinking…"):
            spinner_slot = st.empty()
            tracker = ToolCallTracker(spinner_slot)

            try:
                start_total = time.perf_counter()
                response = get_agent().invoke(
                    {
                        "input": user_input,
                        "chat_history": chat_history,
                    },
                    config={"callbacks": [tracker]},
                )
                answer = response["output"]
                total_time = time.perf_counter() - start_total
                tools_used = ", ".join(c["tool"] for c in tracker.calls) or "none"
                logger.info("Agent done in %.2fs | tools: %s", total_time, tools_used)

            except Exception as exc:
                answer = (
                    f"⚠️ I encountered an error: `{exc}`\n\n"
                    "Please rephrase your question or check your `.env` configuration."
                )
                logger.error("Agent error: %s", exc)
                total_time = time.perf_counter() - start_total
                tools_used = "none"

        # Spinner is gone — replace with a plain, non-clickable caption
        spinner_slot.empty()
        summary_slot.caption(
            f"✅ Done in {total_time:.2f}s · tools used: {tools_used}"
        )

        st.markdown(answer)

    # 4. Persist the assistant reply to session history
    st.session_state.messages.append({"role": "assistant", "content": answer})
