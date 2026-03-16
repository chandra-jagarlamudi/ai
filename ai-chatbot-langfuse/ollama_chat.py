"""
Ollama local chat provider (KISS).

Steps:
  1. Read OLLAMA_BASE_URL and OLLAMA_MODEL from environment (.env).
  2. On first use: check Ollama is reachable (GET /api/tags), then keep client.
  3. query(prompt) calls ollama.generate(model, prompt) and returns response text.

Env: OLLAMA_BASE_URL (default: http://localhost:11434), OLLAMA_MODEL (default: llama2).
Requires: Ollama server running and model pulled (e.g. ollama pull llama2).
"""
import os
import logging

logger = logging.getLogger(__name__)

_instance = None


def get_provider():
    """
    Return the single Ollama provider instance.

    Important: Fails if Ollama server is not reachable or model missing.
    """
    global _instance
    if _instance is None:
        _instance = OllamaProvider()
    return _instance


class OllamaProvider:
    """
    Thin wrapper around the ollama Python client.

    - __init__: Read env, test connection to Ollama, store client and model name.
    - query: ollama.generate(model, prompt); return response body text.
    """

    def __init__(self):
        import ollama
        import requests

        self.name = "Ollama (Local)"
        self.model_name = os.getenv("OLLAMA_MODEL", "llama2")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama = ollama

        # Fail fast if Ollama not running
        r = requests.get(f"{self.base_url}/api/tags", timeout=5)
        if r.status_code != 200:
            raise ConnectionError(f"Ollama not responding at {self.base_url} (status {r.status_code})")
        logger.info("Ollama connected at %s", self.base_url)

    def query(self, prompt: str, *, trace_id: str | None = None) -> str:
        """Send prompt to Ollama, return reply text. trace_id unused (tracing in config)."""
        try:
            out = self.ollama.generate(model=self.model_name, prompt=prompt, stream=False)
            if isinstance(out, dict):
                return out.get("response", "")
            return getattr(out, "response", str(out))
        except Exception as e:
            logger.exception("Ollama query failed")
            return f"Error calling Ollama: {e}"
