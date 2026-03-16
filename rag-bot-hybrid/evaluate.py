import argparse
import json
import os
import time

from dotenv import load_dotenv

from ragbot.evaluation.evaluation import exact_match, grounding_overlap, retrieval_hit_at_k, token_f1
from ragbot.llm.models import generate_answer, get_embedding
from ragbot.retrieval.retriever import HybridRetriever
from ragbot.retrieval.reranker import LocalReranker
from ragbot.config.settings import apply_provider_overrides, load_config, resolve_runtime_profile


load_dotenv()


def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for the hybrid RAG app")
    parser.add_argument("--dataset", default="eval_data/eval_dataset.jsonl", help="Path to eval dataset jsonl")
    parser.add_argument("--profile", default=os.getenv("ENV_MODE", "poc"), help="Runtime profile")
    parser.add_argument("--retrieval-mode", default="hybrid", choices=["hybrid", "keyword", "vector"])
    parser.add_argument("--embedding-provider", default=None, choices=["openai", "huggingface_local", None])
    parser.add_argument("--generator-provider", default=None, choices=["openai", "gemini", "ollama", None])
    parser.add_argument("--output", default="eval_logs/batch_eval_results.jsonl", help="Output jsonl file")
    args = parser.parse_args()

    config = load_config()
    _, base_profile = resolve_runtime_profile(config, args.profile)
    runtime = apply_provider_overrides(base_profile, args.embedding_provider, args.generator_provider)

    retriever = HybridRetriever(
        index_name=config["indexes"]["elastic_index_name"],
        collection_name=config["indexes"]["qdrant_collection_name"],
        dims=runtime["embedding"]["dims"],
        top_k=config["retrieval"]["top_k"],
        rrf_k=config["retrieval"]["rrf_k"],
    )
    retriever.ensure_stores()

    reranker = None
    if config["reranking"]["enabled"] and args.retrieval_mode == "hybrid":
        reranker = LocalReranker(config["reranking"]["model_name"])

    dataset = load_dataset(args.dataset)
    if not dataset:
        raise ValueError(f"Dataset is empty: {args.dataset}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []

    for row in dataset:
        query = row["query"]
        expected_answer = row.get("expected_answer", "")
        expected_sources = row.get("expected_sources", [])

        t0 = time.perf_counter()
        query_vector = get_embedding(query, runtime["embedding"])
        hits = retriever.search(query=query, query_vector=query_vector, mode=args.retrieval_mode)

        if reranker and hits:
            selected_hits, scores = reranker.rerank(
                query=query,
                hits=hits,
                top_n=config["reranking"]["top_n"],
            )
        else:
            selected_hits = hits[: config["reranking"]["top_n"]]
            scores = [float(item.get("score", 0.0)) for item in selected_hits]

        context = "\n\n".join([hit["text"] for hit in selected_hits])
        answer = generate_answer(query=query, context=context, generator_cfg=runtime["generator"], history=[])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "id": row.get("id", ""),
            "query": query,
            "retrieval_mode": args.retrieval_mode,
            "latency_ms": round(latency_ms, 2),
            "sources_used": len(selected_hits),
            "grounding_overlap": round(grounding_overlap(answer, context), 4),
            "retrieval_hit_at_k": retrieval_hit_at_k(selected_hits, expected_sources),
            "answer": answer,
            "expected_answer": expected_answer,
            "exact_match": exact_match(answer, expected_answer) if expected_answer else None,
            "token_f1": round(token_f1(answer, expected_answer), 4) if expected_answer else None,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }
        results.append(result)

    with open(args.output, "w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=True) + "\n")

    total = len(results)
    avg_latency = sum(item["latency_ms"] for item in results) / total
    avg_grounding = sum(item["grounding_overlap"] for item in results) / total
    avg_retrieval_hit = sum(item["retrieval_hit_at_k"] for item in results) / total

    f1_values = [item["token_f1"] for item in results if item["token_f1"] is not None]
    avg_f1 = (sum(f1_values) / len(f1_values)) if f1_values else 0.0

    print("Evaluation completed")
    print(f"Rows: {total}")
    print(f"Avg latency (ms): {avg_latency:.2f}")
    print(f"Avg grounding overlap: {avg_grounding:.3f}")
    print(f"Retrieval hit@k: {avg_retrieval_hit:.3f}")
    print(f"Avg token F1: {avg_f1:.3f}")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
