"""
Shared config: load .env, init Langfuse, and central tracing.
All model names and API keys are read from .env.
"""
import os
import logging
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Load .env from this project directory (same when run from repo root)
_PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(_PROJECT_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mandatory Langfuse (fail at startup if keys missing) ---
if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
    raise ValueError(
        "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required. Set them in .env. "
        "Get keys from https://cloud.langfuse.com (or your Langfuse project settings)."
    )
try:
    from langfuse import get_client
    _langfuse = get_client()
    try:
        _langfuse.auth_check()
    except Exception as e:
        logger.error(f"Langfuse auth check failed: {e}")
        raise ValueError(
            "Langfuse authentication failed. Verify LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL in .env."
        ) from e
except Exception as e:
    logger.error(f"Langfuse client init failed: {e}")
    raise


def get_langfuse():
    """Return the Langfuse client."""
    return _langfuse


# Provider registry: populated by get_all_providers()
_providers_cache: dict[str, object] = {}
_provider_errors: dict[str, str] = {}


def get_all_providers() -> tuple[dict[str, object], dict[str, str]]:
    """
    Load all four providers at startup. Returns (providers, errors).
    providers[key] is the provider instance or None; errors[key] is the error message if init failed.
    """
    global _providers_cache, _provider_errors
    if _providers_cache or _provider_errors:
        return _providers_cache.copy(), _provider_errors.copy()

    for key in ("openai", "gemini", "huggingface", "ollama"):
        try:
            if key == "openai":
                from openai_chat import get_provider
            elif key == "gemini":
                from gemini_chat import get_provider
            elif key == "huggingface":
                from huggingface_chat import get_provider
            else:  # ollama
                from ollama_chat import get_provider
            provider = get_provider()
            _providers_cache[key] = provider
            logger.info(f"Initialized provider: {provider.name}")
        except Exception as e:
            logger.warning(f"Provider {key} not available: {e}")
            _providers_cache[key] = None
            _provider_errors[key] = str(e)

    return _providers_cache.copy(), _provider_errors.copy()


def query_with_trace(provider_key: str, prompt: str) -> str:
    """
    Create one Langfuse trace per message and call the provider's query.
    OpenAI uses LangChain callback with trace_id; others use start_as_current_observation (generation).
    """
    providers, _ = get_all_providers()
    provider = providers.get(provider_key)
    if provider is None:
        return f"Error: Provider '{provider_key}' is not configured. Check .env and try again."

    trace_id = _langfuse.create_trace_id()

    try:
        with _langfuse.start_as_current_observation(
            name="chatbot-query",
            trace_context={"trace_id": trace_id},
        ) as span:
            try:
                if provider.name == "OpenAI":
                    response = provider.query(prompt, trace_id=trace_id)
                else:
                    with _langfuse.start_as_current_observation(
                        name="llm",
                        as_type="generation",
                        input=prompt,
                        model=provider.model_name if getattr(provider, "model_name", None) else provider.name,
                    ) as gen:
                        response = provider.query(prompt)
                        gen.update(output=response)
                return response
            except Exception as e:
                try:
                    span.update(status_message=str(e), level="ERROR")
                except Exception:
                    pass
                logger.error(f"Provider error: {e}\n{traceback.format_exc()}")
                return f"Error: {e}"
    finally:
        try:
            _langfuse.flush()
        except Exception as e:
            logger.warning(f"Langfuse flush failed: {e}")
