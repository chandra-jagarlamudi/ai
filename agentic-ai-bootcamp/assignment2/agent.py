"""
Agent Configuration
--------------------
Sets up a LangChain OpenAI-style function-calling agent backed by LiteLLM.

Uses langchain_classic which provides the classic AgentExecutor pattern:
    create_openai_tools_agent  →  builds the runnable agent chain
    AgentExecutor              →  runs the think → tool → observe loop

LiteLLM acts as a universal proxy — by changing LLM_MODEL in .env
you can route calls to OpenAI, Anthropic, Ollama, or any other provider
without touching application code.

Flow:
    User message + chat history
        → ChatLiteLLM  (model selected via LLM_MODEL in .env)
            → OpenAI-style function calling
                → Tool execution (EMI / SIP / Budget / Stock / Calculator)
                    → Final response back to Streamlit
"""

import logging
import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import ALL_TOOLS

load_dotenv()

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable and friendly personal financial AI assistant.

You help users with:
- 📊 Loan EMI calculations — monthly payments, total interest on any loan
- 📈 SIP investment projections — future value of systematic monthly investments
- 💼 Budget planning — 50-30-20 rule allocation for any income level
- 📉 Live stock prices — real-time quotes from Yahoo Finance
- 🧮 Mathematical calculations — arithmetic and scientific expressions via NumPy

Guidelines:
- Always call the appropriate tool for numerical calculations; never guess figures.
- Present results clearly, with currency formatting and explanatory context.
- If the user provides incomplete information (e.g., missing interest rate), ask for it.
- Keep responses concise but informative.
- When asked about investments, always add a brief disclaimer that past returns
  do not guarantee future performance.
"""


def create_financial_agent() -> AgentExecutor:
    """
    Build and return a configured LangChain AgentExecutor.

    The model is read from the LLM_MODEL environment variable so that
    switching providers requires only an .env change.

    Returns:
        AgentExecutor that accepts {"input": str, "chat_history": list}.
    """
    model = os.getenv("LLM_MODEL", "openai/gpt-4o")
    logger.info("Initialising financial agent with model: %s", model)

    # ChatLiteLLM wraps LiteLLM inside the LangChain chat model interface.
    # LiteLLM normalises the API for OpenAI, Anthropic, Ollama, etc.
    llm = ChatLiteLLM(model=model)

    # Prompt template — the 'agent_scratchpad' placeholder is required by
    # create_openai_tools_agent to record intermediate tool-call steps.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # create_openai_tools_agent uses OpenAI's parallel function-calling format.
    # Works with any LiteLLM-supported model that supports tool/function calling.
    agent = create_openai_tools_agent(llm, ALL_TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,               # logs tool calls to stdout for debugging
        max_iterations=5,           # prevent infinite loops on ambiguous queries
        handle_parsing_errors=True, # gracefully handle malformed LLM output
    )
