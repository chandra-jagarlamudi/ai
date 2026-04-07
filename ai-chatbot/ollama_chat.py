import os
import logging
from typing import Generator
import ollama

logger = logging.getLogger(__name__)

NAME = "Ollama"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b-q4_K_M")
_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _client() -> ollama.Client:
    return ollama.Client(host=_HOST)


def is_configured() -> bool:
    # Ollama runs locally — always show the tab
    return True


def list_models() -> list[str]:
    """Return names of all locally installed Ollama models."""
    try:
        result = _client().list()
        return [m["model"] for m in result.get("models", [])]
    except Exception as e:
        logger.error("Could not list Ollama models: %s", e)
        return []


def stream_chat(
    prompt: str,
    history: list[dict],
    model: str,
) -> Generator[str, None, None]:
    """
    Yield response tokens one at a time for streaming display.

    history: list of {"role": "user"|"assistant", "content": str}
             representing the conversation so far (NOT including the current prompt).
    """
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": prompt})

    try:
        stream = _client().chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except ollama.ResponseError as e:
        msg = str(e).lower()
        if "not found" in msg or "pull" in msg:
            yield f"\n\nModel `{model}` is not installed locally.\nRun this in your terminal:\n\n```\nollama pull {model}\n```"
        else:
            logger.error("Ollama ResponseError: %s", e)
            yield f"Error from Ollama: {e}"
    except Exception as e:
        logger.error("Ollama connection error: %s", e)
        yield (
            f"Could not connect to Ollama at `{_HOST}`.\n\n"
            "Make sure Ollama is running:\n```\nollama serve\n```"
        )


def chat(prompt: str) -> str:
    """Non-streaming fallback (used by the generic render_tab if ever called)."""
    return "".join(stream_chat(prompt, [], DEFAULT_MODEL))
