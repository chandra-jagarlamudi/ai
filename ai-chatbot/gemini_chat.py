import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

NAME = "Gemini"
_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")


def is_configured() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def chat(prompt: str) -> str:
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return f"Error: {e}"
