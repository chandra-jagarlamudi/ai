import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

NAME = "OpenAI"

# OPENAI_MODEL may be a single model or a comma-separated list.
# e.g. "gpt-4o-mini,gpt-4.1-nano-2025-04-14,gpt-5-nano-2025-08-07"
MODELS: list[str] = [
    m.strip()
    for m in os.getenv("OPENAI_MODEL", "gpt-4o-mini").split(",")
    if m.strip()
]
DEFAULT_MODEL = MODELS[0]


def is_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def chat(prompt: str, history: list[dict] | None = None, model: str = DEFAULT_MODEL) -> str:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = list(history or []) + [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return f"Error: {e}"
