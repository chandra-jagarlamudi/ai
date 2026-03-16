import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

NAME = "OpenAI"
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def is_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def chat(prompt: str) -> str:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return f"Error: {e}"
