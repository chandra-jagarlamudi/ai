#!/usr/bin/env python3
"""
Run the same RAG test questions against all configured comparison models.
Records response time and response text. Use the output to fill COMPARISON_TABLE.md.

Usage:
  - Ensure .env has the API keys you need (OPENAI_API_KEY, GOOGLE_API_KEY for Gemini).
  - For Ollama: have Ollama running and `ollama pull gemma` (or your chosen model).
  - Build the vectorstore once from the Streamlit app (same embedding for all runs).
  - From assignment1/: python run_comparison.py

Output:
  - comparison_results.json  (full responses and timings)
  - Printed summary table (Question, Model, Time (s), Response preview)
  - Fill COMPARISON_TABLE.md with Accuracy and Hallucination from manual review.
"""
from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path

# macOS FAISS/OpenMP workaround
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# Test questions: (short label, full question)
TEST_QUESTIONS = [
    ("Leave policies", "What are the leave policies? How many days of annual leave do employees get?"),
    ("Remote work rules", "What are the remote work rules? Is remote work allowed and under what conditions?"),
    ("Startup pricing model", "What is the startup's pricing model? How does the company price its product?"),
    ("Sick leave", "What is the policy on sick leave and how should employees report it?"),
    ("Company values", "What are the company's core values or mission mentioned in the documents?"),
]

# ── Comparison model list ──────────────────────────────────────────────────────
# To add a model to the comparison, add one tuple here:
#   (<env_var_for_model>, <provider>, <default_model>)
# The provider must exist in PROVIDER_REGISTRY in rag_pipeline.py.
_COMPARISON_PROVIDERS: list[tuple[str, str, str]] = [
    ("OPENAI_CHAT_MODEL", "openai", "gpt-4o-mini"),
    ("OLLAMA_CHAT_MODEL", "ollama", "gemma3:1b"),
    ("GEMINI_CHAT_MODEL", "gemini", "gemini-2.5-flash"),
]


def _models_from_env() -> list[tuple[str, str, str]]:
    """Returns (label, llm_provider, llm_model) for each entry in _COMPARISON_PROVIDERS."""
    models = []
    for env_var, provider, default in _COMPARISON_PROVIDERS:
        model = os.getenv(env_var, default)
        label = f"{provider.title()} ({model})"
        models.append((label, provider, model))
    return models


MODELS = _models_from_env()


def run_comparison(vectorstore, cfg, *, base_path: Path | None = None) -> tuple[list[dict], Path]:
    """
    Run test questions against all models in MODELS. Writes comparison_results.json.

    Returns (results, out_path). Can be called from Streamlit app or from CLI.
    """
    from rag_pipeline import config_for_llm, get_llm, rag_answer

    base = base_path or Path(__file__).resolve().parent
    results: list[dict] = []
    logger.info("Comparison started models=%d questions=%d", len(MODELS), len(TEST_QUESTIONS))

    for q_label, question in TEST_QUESTIONS:
        for model_label, llm_provider, llm_model in MODELS:
            model_cfg = config_for_llm(cfg, llm_provider, llm_model)
            llm = get_llm(model_cfg)
            try:
                logger.info("Running: question=%s model=%s", q_label, model_label)
                answer, meta = rag_answer(
                    question=question,
                    vectorstore=vectorstore,
                    llm=llm,
                    cfg=model_cfg,
                )
                results.append({
                    "question_label": q_label,
                    "question": question,
                    "model_label": model_label,
                    "model_provider": llm_provider,
                    "model_name": llm_model,
                    "response": answer,
                    "time_seconds": meta.get("time_seconds"),
                })
                logger.info("Success: question=%s model=%s time_s=%s", q_label, model_label, meta.get("time_seconds"))
            except Exception as e:
                results.append({
                    "question_label": q_label,
                    "question": question,
                    "model_label": model_label,
                    "model_provider": llm_provider,
                    "model_name": llm_model,
                    "response": "",
                    "error": str(e),
                    "time_seconds": None,
                })
                logger.exception("Failure: question=%s model=%s", q_label, model_label)

    out_path = base / "comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Comparison finished out=%s", str(out_path))
    return results, out_path


def summary_table_markdown(results: list[dict]) -> str:
    """Build Response Speed summary as a markdown table."""
    lines = [
        "| Question | " + " | ".join(m[0] for m in MODELS) + " |",
        "| " + " | ".join(["----------"] * (len(MODELS) + 1)) + " |",
    ]
    for q_label, _ in TEST_QUESTIONS:
        row: list[str] = [q_label]
        for model_label, _, _ in MODELS:
            r = next((x for x in results if x["question_label"] == q_label and x["model_label"] == model_label), None)
            if r and r.get("time_seconds") is not None:
                row.append(f"{r['time_seconds']} s")
            else:
                row.append(r.get("error", "—")[:20] if r else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    base = Path(__file__).resolve().parent
    sys.path.insert(0, str(base))

    from rag_pipeline import comparison_config, load_vectorstore

    cfg = comparison_config()
    vectorstore = load_vectorstore(cfg)
    if vectorstore is None:
        print("Error: Comparison vectorstore not found.")
        print("Build it from the Streamlit app: sidebar → 'Build comparison vectorstore'.")
        print("Uses OpenAI embeddings and saves to vectorstore_comparison/ (or COMPARISON_VECTORSTORE_DIR).")
        sys.exit(1)

    results, out_path = run_comparison(vectorstore, cfg, base_path=base)
    print(f"\nWrote {out_path}")
    print("\n--- Summary (Response Speed) ---\n")
    print(summary_table_markdown(results))
    print(f"\nFill COMPARISON_TABLE.md with: Accuracy and Hallucination (manual); Speed from above; Cost and Ease of Setup (see README).")


if __name__ == "__main__":
    main()
