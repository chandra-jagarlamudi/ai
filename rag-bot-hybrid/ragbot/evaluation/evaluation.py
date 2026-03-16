import json
import os
import re
from datetime import datetime

import pandas as pd


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", _normalize(text))
    return [t for t in cleaned.split() if t]


def exact_match(prediction: str, expected: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(expected) else 0.0


def token_f1(prediction: str, expected: str) -> float:
    pred_tokens = _tokens(prediction)
    exp_tokens = _tokens(expected)
    if not pred_tokens or not exp_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    exp_set = set(exp_tokens)
    overlap = len(pred_set.intersection(exp_set))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_set)
    recall = overlap / len(exp_set)
    return 2 * precision * recall / (precision + recall)


def grounding_overlap(answer: str, context: str) -> float:
    answer_tokens = set(_tokens(answer))
    context_tokens = set(_tokens(context))
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens.intersection(context_tokens))
    return overlap / len(answer_tokens)


def retrieval_hit_at_k(hits: list[dict], expected_sources: list[dict]) -> float:
    if not expected_sources:
        return 0.0

    expected_pairs = {
        (str(item.get("source", "")).strip(), int(item.get("page", -1))) for item in expected_sources
    }

    for hit in hits:
        metadata = hit.get("metadata", {})
        candidate = (str(metadata.get("source", "")).strip(), int(metadata.get("page", -1)))
        if candidate in expected_pairs:
            return 1.0
    return 0.0


def make_interaction_record(
    query: str,
    answer: str,
    context: str,
    retrieval_mode: str,
    embedding_provider: str,
    generator_provider: str,
    hits: list[dict],
    scores: list[float],
    latency_ms: float,
) -> dict:
    top_hit = hits[0] if hits else {}
    top_meta = top_hit.get("metadata", {})

    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "query": query,
        "answer": answer,
        "retrieval_mode": retrieval_mode,
        "embedding_provider": embedding_provider,
        "generator_provider": generator_provider,
        "latency_ms": round(latency_ms, 2),
        "sources_used": len(hits),
        "top_source": top_meta.get("source", ""),
        "top_page": top_meta.get("page", ""),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "grounding_overlap": round(grounding_overlap(answer, context), 4),
    }


def append_interaction_log(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_logs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def summarize_logs(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_grounding_overlap": 0.0,
            "avg_sources_used": 0.0,
        }

    return {
        "total_queries": int(len(df)),
        "avg_latency_ms": float(df["latency_ms"].mean()),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        "avg_grounding_overlap": float(df["grounding_overlap"].mean()),
        "avg_sources_used": float(df["sources_used"].mean()),
    }
