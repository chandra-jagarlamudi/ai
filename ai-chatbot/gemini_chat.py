import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

NAME = "Gemini"
_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")


def is_configured() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def chat(prompt: str, history: list[dict] | None = None) -> str:
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(_MODEL)

        # Gemini uses "model" for the assistant role (not "assistant"),
        # and wraps each message content in a "parts" list.
        gemini_history = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
            for m in (history or [])
        ]
        response = model.start_chat(history=gemini_history).send_message(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return f"Error: {e}"
