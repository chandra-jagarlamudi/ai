import os
import logging
import ollama

logger = logging.getLogger(__name__)

NAME = "Ollama"
_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def is_configured() -> bool:
    # Ollama runs locally — always attempt it
    return True


def chat(prompt: str) -> str:
    try:
        client = ollama.Client(host=_HOST)
        response = client.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return f"Error: {e}"
