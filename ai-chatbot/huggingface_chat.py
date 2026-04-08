import os
import logging
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

NAME = "HuggingFace"
_MODEL = os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_PROVIDER = os.getenv("HF_PROVIDER", "featherless-ai")


def is_configured() -> bool:
    return bool(os.getenv("HF_API_TOKEN"))


def chat(prompt: str, history: list[dict] | None = None) -> str:
    try:
        client = InferenceClient(
            provider=_PROVIDER,
            api_key=os.getenv("HF_API_TOKEN"),
        )
        messages = list(history or []) + [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(model=_MODEL, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"HuggingFace error: {e}")
        return f"Error: {e}"
