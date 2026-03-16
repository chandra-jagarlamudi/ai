"""
Hugging Face chat provider (KISS): API or local.

Steps:
  1. Read HF_* and USE_LOCAL_MODEL from environment (.env).
  2. If USE_LOCAL_MODEL is true: load transformers pipeline once (can be slow).
  3. If false: use InferenceClient; no model load.
  4. query(prompt) returns generated text (local: strip prompt from output; API: parse choices/message).

Env: HF_CHAT_MODEL (must be a chat/instruction model, e.g. *-it or *-Instruct, not *-pt),
     HF_API_TOKEN (for API), HF_DEVICE, HF_MAX_TOKENS, HF_USE_8BIT;
     USE_LOCAL_MODEL (true/false) toggles local vs API.
"""
import os
import logging

logger = logging.getLogger(__name__)

_instance = None


def get_provider():
    """
    Return the single HuggingFace provider instance.

    Important: Local mode loads the model in __init__ (slow first time). API mode only needs HF_API_TOKEN.
    """
    global _instance
    if _instance is None:
        _instance = HuggingFaceProvider()
    return _instance


class HuggingFaceProvider:
    """
    Two modes: local (transformers pipeline) or API (InferenceClient).

    - __init__: Read env; if local, build text-generation pipeline; else just store model name and token.
    - query: _query_local or _query_api; both return a single string.
    """

    def __init__(self):
        self.use_local = os.getenv("USE_LOCAL_MODEL", "false").lower() in ("1", "true", "yes")
        self.model_name = os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.api_token = os.getenv("HF_API_TOKEN")
        self.device = os.getenv("HF_DEVICE", "auto")
        self.max_tokens = int(os.getenv("HF_MAX_TOKENS", "512"))
        self.use_8bit = os.getenv("HF_USE_8BIT", "false").lower() in ("1", "true", "yes")
        self.pipeline = None

        if self.use_local:
            self.name = "Hugging Face (Local)"
            self._init_pipeline()
        else:
            self.name = "Hugging Face (API)"

    def _init_pipeline(self):
        """Load transformers text-generation pipeline once. Uses HF_CHAT_MODEL, HF_DEVICE, HF_USE_8BIT."""
        from transformers import pipeline

        kwargs = {"model": self.model_name, "trust_remote_code": True, "device_map": self.device}
        if self.use_8bit:
            kwargs["load_in_8bit"] = True
        token = os.getenv("HF_TOKEN") or self.api_token
        if token:
            kwargs["token"] = token
        self.pipeline = pipeline("text-generation", **kwargs)

    def query(self, prompt: str, *, trace_id: str | None = None) -> str:
        """Dispatch to local or API; return reply text. trace_id unused (tracing in config)."""
        try:
            if self.use_local:
                return self._query_local(prompt)
            return self._query_api(prompt)
        except Exception as e:
            logger.exception("HuggingFace query failed")
            return f"Error calling Hugging Face: {e}"

    def _query_local(self, prompt: str) -> str:
        """Run pipeline, get generated_text, strip the prompt prefix so we only return the new part."""
        if self.pipeline is None:
            self._init_pipeline()
        resp = self.pipeline(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        # Pipeline returns list of dicts with generated_text or similar
        if isinstance(resp, list) and resp and isinstance(resp[0], dict):
            text = resp[0].get("generated_text") or resp[0].get("text") or ""
        elif isinstance(resp, dict):
            text = resp.get("generated_text") or resp.get("text") or ""
        else:
            text = str(resp)
        if text.startswith(prompt):
            text = text[len(prompt) :].strip()
        return text or "(no output)"

    def _query_api(self, prompt: str) -> str:
        """Call HF Inference API via configured provider (HF_PROVIDER in .env).
        HF now routes through third-party providers — set HF_PROVIDER to one
        enabled on your account (huggingface.co/settings/inference-providers).
        """
        from huggingface_hub import InferenceClient

        provider = os.getenv("HF_PROVIDER", "featherless-ai")
        client = InferenceClient(provider=provider, api_key=self.api_token)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
