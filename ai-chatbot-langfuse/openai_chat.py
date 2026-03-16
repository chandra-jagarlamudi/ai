"""
OpenAI chat provider (KISS).

Steps:
  1. Read OPENAI_API_KEY and OPENAI_MODEL from environment (.env).
  2. Build one ChatOpenAI client on first use.
  3. query(prompt) calls the API; if trace_id is given, Langfuse callback attaches to that trace.

Env: OPENAI_API_KEY (required), OPENAI_MODEL (default: gpt-4.1).
"""
import os
import logging

logger = logging.getLogger(__name__)

_instance = None


def get_provider():
    """
    Return the single OpenAI provider instance.

    Important: Fails with ValueError if OPENAI_API_KEY is not set.
    """
    global _instance
    if _instance is None:
        _instance = OpenAIProvider()
    return _instance


class OpenAIProvider:
    """
    Thin wrapper around LangChain ChatOpenAI.

    - __init__: Read env, create ChatOpenAI (fails if key missing).
    - query: Invoke model; optional trace_id wires LangChain callback to Langfuse.
    """

    def __init__(self):
        from langchain_openai import ChatOpenAI

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        self.name = "OpenAI"
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
        self.llm = ChatOpenAI(model=self.model_name, api_key=key)

    def query(self, prompt: str, *, trace_id: str | None = None) -> str:
        """
        Send prompt to OpenAI, return reply text.

        If trace_id is set, LangChain CallbackHandler sends spans to that Langfuse trace.
        """
        try:
            if trace_id:
                from langfuse.langchain import CallbackHandler

                config = {"callbacks": [CallbackHandler()], "metadata": {"langfuse_trace_id": trace_id}}
                out = self.llm.invoke(prompt, config=config)
            else:
                out = self.llm.invoke(prompt)
            return out.content if hasattr(out, "content") else str(out)
        except Exception as e:
            logger.exception("OpenAI query failed")
            return f"Error calling OpenAI: {e}"
