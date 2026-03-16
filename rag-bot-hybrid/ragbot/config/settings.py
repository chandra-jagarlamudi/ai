import os
import yaml
from dotenv import load_dotenv


load_dotenv()


def load_config(config_path: str = "config.yml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_runtime_profile(config: dict, selected_profile: str | None = None) -> tuple[str, dict]:
    profile_name = selected_profile or os.getenv("ENV_MODE", "poc")
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown profile '{profile_name}'. Available profiles: {available}")
    return profile_name, profiles[profile_name]


def apply_provider_overrides(profile: dict, embedding_provider: str | None, generator_provider: str | None) -> dict:
    runtime = {
        "stores": dict(profile["stores"]),
        "embedding": dict(profile["embedding"]),
        "reranker": dict(profile["reranker"]),
        "generator": dict(profile["generator"]),
    }

    if embedding_provider:
        runtime["embedding"]["provider"] = embedding_provider
        if embedding_provider == "huggingface_local":
            runtime["embedding"]["model"] = "BAAI/bge-small-en-v1.5"
            runtime["embedding"]["dims"] = 384
        elif embedding_provider == "openai":
            runtime["embedding"].setdefault("model", "text-embedding-3-small")
            runtime["embedding"].setdefault("dims", 1536)

    if generator_provider:
        runtime["generator"]["provider"] = generator_provider
        if generator_provider == "gemini":
            runtime["generator"]["model"] = "gemini-1.5-flash"
        elif generator_provider == "openai":
            runtime["generator"]["model"] = "gpt-4o-mini"
        elif generator_provider == "ollama":
            runtime["generator"]["model"] = "llama3.1:8b-instruct-q4_K_M"

    return runtime
