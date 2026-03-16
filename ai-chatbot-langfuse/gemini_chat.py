"""
Google Gemini chat provider (KISS).

Steps:
  1. Read GOOGLE_API_KEY and GEMINI_MODEL from environment (.env).
  2. Configure genai and create one GenerativeModel on first use.
  3. query(prompt) calls generate_content and returns the text.

Env: GOOGLE_API_KEY (required), GEMINI_MODEL (default: gemini-2.0-flash).
"""
import os
import logging

logger = logging.getLogger(__name__)

_instance = None


def get_provider():
    """
    Return the single Gemini provider instance.

    Important: Fails with ValueError if GOOGLE_API_KEY is not set.
    """
    global _instance
    if _instance is None:
        _instance = GeminiProvider()
    return _instance


class GeminiProvider:
    """
    Thin wrapper around google.generativeai.

    - __init__: Read env, configure API, create GenerativeModel.
    - query: generate_content(prompt) and return .text.
    """

    def __init__(self):
        import google.generativeai as genai

        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set in environment")

        self.name = "Google Gemini"
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(self.model_name)

    def query(self, prompt: str, *, trace_id: str | None = None) -> str:
        """Send prompt to Gemini, return reply text. trace_id unused (tracing done in config)."""
        try:
            response = self.model.generate_content(prompt)
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            logger.exception("Gemini query failed")
            return f"Error calling Gemini: {e}"
