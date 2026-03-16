import logging
import os
import subprocess
import sys
import time
from importlib.util import find_spec

import streamlit as st
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

from ragbot.ingestion.ingest import ingest_pdf
from ragbot.llm.models import generate_answer, get_embedding
from ragbot.retrieval.reranker import LocalReranker
from ragbot.retrieval.retriever import HybridRetriever
from ragbot.config.settings import apply_provider_overrides, load_config, resolve_runtime_profile
from ragbot.evaluation.evaluation import append_interaction_log, load_logs, make_interaction_record, summarize_logs


load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("APP_LOG_PATH", "logs/app.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
else:
    root_logger.setLevel(LOG_LEVEL)
    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", "").endswith(os.path.normpath(LOG_FILE))
        for handler in root_logger.handlers
    )
    if not has_file_handler:
        root_logger.addHandler(logging.FileHandler(LOG_FILE, encoding="utf-8"))

logger = logging.getLogger(__name__)


@st.cache_resource
def get_reranker(model_name: str):
    return LocalReranker(model_name=model_name)


def render_sources(hits: list[dict], scores: list[float], retrieval_mode: str, preview_chars: int):
    for i, (hit, score) in enumerate(zip(hits, scores), start=1):
        meta = hit.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        chunk = meta.get("chunk", "?")
        retriever = hit.get("retriever", retrieval_mode)
        text = hit.get("text", "")
        preview = text[:preview_chars].rstrip() + ("…" if len(text) > preview_chars else "")

        score_label = f"{score:.4f}" if score is not None else "n/a"
        score_type = "Rerank" if retrieval_mode == "hybrid" else "Score"

        with st.container(border=True):
            col_rank, col_meta, col_score = st.columns([1, 5, 2])
            col_rank.markdown(f"**#{i}**")
            col_meta.markdown(
                f"📄 **{source}** &nbsp;·&nbsp; page {page} &nbsp;·&nbsp; chunk {chunk}  "
                f"<br><span style='color:#888;font-size:0.8rem'>retriever: {retriever}</span>",
                unsafe_allow_html=True,
            )
            col_score.markdown(
                f"<div style='text-align:right'><span style='font-size:0.75rem;color:#888'>{score_type}</span><br>"
                f"<span style='font-size:1rem;font-weight:600'>{score_label}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(preview)


def get_runtime_state(config: dict, selected_profile: str, embed_provider: str, gen_provider: str):
    profile_name, base_profile = resolve_runtime_profile(config, selected_profile)
    runtime = apply_provider_overrides(base_profile, embed_provider, gen_provider)
    dims = runtime["embedding"]["dims"]

    retriever = HybridRetriever(
        index_name=config["indexes"]["elastic_index_name"],
        collection_name=config["indexes"]["qdrant_collection_name"],
        dims=dims,
        top_k=config["retrieval"]["top_k"],
        rrf_k=config["retrieval"]["rrf_k"],
    )
    return profile_name, runtime, retriever


def dependency_report(embedding_provider: str, generator_provider: str, reranking_enabled: bool) -> dict:
    required_packages = ["pypdf", "elasticsearch", "qdrant_client"]
    required_env_vars = []

    if embedding_provider == "openai":
        required_packages.append("openai")
        required_env_vars.append("OPENAI_API_KEY")
    elif embedding_provider == "huggingface_local":
        required_packages.append("sentence_transformers")

    if generator_provider == "openai":
        required_packages.append("openai")
        required_env_vars.append("OPENAI_API_KEY")
    elif generator_provider == "gemini":
        required_packages.append("google.generativeai")
        required_env_vars.append("GOOGLE_API_KEY")
    elif generator_provider == "ollama":
        required_packages.append("ollama")

    if reranking_enabled:
        required_packages.append("sentence_transformers")

    missing_packages = sorted({pkg for pkg in required_packages if find_spec(pkg) is None})
    missing_env_vars = sorted({key for key in required_env_vars if not os.getenv(key)})

    return {
        "missing_packages": missing_packages,
        "missing_env_vars": missing_env_vars,
    }


def connectivity_report() -> dict:
    results = {
        "elasticsearch": {"ok": False, "detail": "not checked"},
        "qdrant": {"ok": False, "detail": "not checked"},
    }

    try:
        es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        es = Elasticsearch(es_url)
        health = es.cluster.health()
        status = health.get("status", "unknown")
        results["elasticsearch"] = {
            "ok": status in {"green", "yellow"},
            "detail": f"status={status}",
        }
    except Exception as exc:
        results["elasticsearch"] = {"ok": False, "detail": str(exc)}

    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)
        client.get_collections()
        results["qdrant"] = {"ok": True, "detail": "reachable"}
    except Exception as exc:
        results["qdrant"] = {"ok": False, "detail": str(exc)}

    return results


def run_compose_command(args: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["docker", "compose", *args],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "docker command not found. Install Docker Desktop and ensure 'docker' is on PATH."
    except Exception as exc:
        return False, str(exc)

    output = (completed.stdout or "").strip()
    errors = (completed.stderr or "").strip()

    if completed.returncode == 0:
        return True, output or "Command finished successfully."

    message = errors or output or f"Command failed with exit code {completed.returncode}."
    return False, message


def wait_for_infra_ready(timeout_seconds: int = 25, interval_seconds: int = 2) -> tuple[bool, str]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        infra = connectivity_report()
        if infra["elasticsearch"]["ok"] and infra["qdrant"]["ok"]:
            return True, "Elasticsearch and Qdrant are reachable."
        time.sleep(interval_seconds)
    return False, "Infrastructure is starting but not ready yet. Retry in a few seconds."


def run_batch_eval(
    dataset_path: str,
    output_path: str,
    profile: str,
    retrieval_mode: str,
    embedding_provider: str,
    generator_provider: str,
) -> tuple[bool, str]:
    command = [
        sys.executable,
        "evaluate.py",
        "--dataset",
        dataset_path,
        "--output",
        output_path,
        "--profile",
        profile,
        "--retrieval-mode",
        retrieval_mode,
        "--embedding-provider",
        embedding_provider,
        "--generator-provider",
        generator_provider,
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    output = "\n".join(part for part in [stdout, stderr] if part).strip()

    if completed.returncode == 0:
        return True, output or "Batch evaluation completed."

    return False, output or f"Batch evaluation failed with code {completed.returncode}."


def main():
    config = load_config()

    st.set_page_config(page_title="Hybrid RAG", layout="wide")
    st.title(config["app"]["title"])

    with st.sidebar:
        st.header("Runtime")
        profiles = list(config["profiles"].keys())
        selected_profile = st.selectbox("Profile", profiles, index=profiles.index("poc") if "poc" in profiles else 0)

        embedding_provider = st.selectbox("Embedding provider", ["openai", "huggingface_local"])
        generator_provider = st.selectbox("Generator provider", ["openai", "gemini", "ollama"])

        retrieval_mode = st.selectbox(
            "Retrieval mode",
            ["hybrid", "keyword", "vector"],
            key="chat_retrieval_mode",
        )
        show_sources = st.toggle("Show sources", value=config["sources"]["show"])

        st.divider()
        st.header("Batch Evaluation")
        dataset_path = st.text_input("Dataset path", value=config["evaluation"]["dataset_path"])
        output_path = st.text_input("Output path", value=config["evaluation"]["batch_output_path"])
        batch_mode = st.selectbox(
            "Batch retrieval mode",
            ["hybrid", "keyword", "vector"],
            index=["hybrid", "keyword", "vector"].index(retrieval_mode),
            key="batch_retrieval_mode",
        )
        run_batch_eval_clicked = st.button("Run Batch Eval", use_container_width=True)

        if run_batch_eval_clicked:
            if not os.path.exists(dataset_path):
                st.error(f"Dataset file not found: {dataset_path}")
                logger.error("Batch eval failed. Dataset not found: %s", dataset_path)
            else:
                logger.info(
                    "Starting batch eval | dataset=%s output=%s profile=%s mode=%s embedding=%s generator=%s",
                    dataset_path,
                    output_path,
                    selected_profile,
                    batch_mode,
                    embedding_provider,
                    generator_provider,
                )
                with st.spinner("Running batch evaluation..."):
                    ok, msg = run_batch_eval(
                        dataset_path=dataset_path,
                        output_path=output_path,
                        profile=selected_profile,
                        retrieval_mode=batch_mode,
                        embedding_provider=embedding_provider,
                        generator_provider=generator_provider,
                    )
                if ok:
                    st.success("Batch evaluation completed")
                    st.code(msg)
                    logger.info("Batch eval completed successfully")
                else:
                    st.error("Batch evaluation failed")
                    st.code(msg)
                    logger.error("Batch eval failed: %s", msg)

        st.divider()
        st.header("Infrastructure")

        # --- Service status indicators ---
        infra = connectivity_report()
        es_ok = infra["elasticsearch"]["ok"]
        q_ok = infra["qdrant"]["ok"]
        col_es, col_q = st.columns(2)
        col_es.markdown(
            f"**Elasticsearch**  \n"
            f"<span style='color:{'#21c354' if es_ok else '#ff4b4b'};font-size:1.1rem'>{'● Online' if es_ok else '● Offline'}</span>",
            unsafe_allow_html=True,
        )
        col_q.markdown(
            f"**Qdrant**  \n"
            f"<span style='color:{'#21c354' if q_ok else '#ff4b4b'};font-size:1.1rem'>{'● Online' if q_ok else '● Offline'}</span>",
            unsafe_allow_html=True,
        )

        # --- Controls ---
        c1, c2, c3 = st.columns(3)
        with c1:
            start_clicked = st.button("Start", use_container_width=True)
        with c2:
            stop_clicked = st.button("Stop", use_container_width=True)
        with c3:
            status_clicked = st.button("Status", use_container_width=True)

        if start_clicked:
            with st.spinner("Starting..."):
                ok, msg = run_compose_command(["up", "-d"])
            if ok:
                with st.spinner("Waiting for services..."):
                    ready, ready_msg = wait_for_infra_ready()
                if ready:
                    st.success(ready_msg)
                else:
                    st.warning(ready_msg)
            else:
                st.error(msg)

        if stop_clicked:
            with st.spinner("Stopping..."):
                ok, msg = run_compose_command(["down"])
            if ok:
                st.success("Infrastructure stopped.")
            else:
                st.error(msg)

        if status_clicked:
            ok, msg = run_compose_command(["ps"])
            if ok:
                st.code(msg or "No running services.")
            else:
                st.error(msg)

        # --- Dependency issues (only shown when there are problems) ---
        report = dependency_report(
            embedding_provider=embedding_provider,
            generator_provider=generator_provider,
            reranking_enabled=config["reranking"]["enabled"],
        )
        if report["missing_packages"] or report["missing_env_vars"]:
            with st.expander("⚠️ Issues detected", expanded=True):
                for pkg in report["missing_packages"]:
                    st.warning(f"Missing package: `{pkg}`")
                for key in report["missing_env_vars"]:
                    st.warning(f"Missing env var: `{key}`")

        st.divider()
        st.header("Ingest PDFs")
        uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if uploads and st.button("Ingest uploaded PDFs"):
            ingest_now = True
        else:
            ingest_now = False

    profile_name, runtime, retriever = get_runtime_state(
        config,
        selected_profile=selected_profile,
        embed_provider=embedding_provider,
        gen_provider=generator_provider,
    )

    st.caption(
        f"Profile: {profile_name} | Embedding: {runtime['embedding']['provider']}/{runtime['embedding']['model']} | "
        f"Generator: {runtime['generator']['provider']}/{runtime['generator']['model']}"
    )

    tab_chat, tab_eval = st.tabs(["Chat", "Evaluation Metrics"])

    eval_log_path = config["evaluation"]["log_path"]

    if ingest_now:
        os.makedirs("data", exist_ok=True)
        total_chunks = 0

        try:
            retriever.ensure_stores()
        except Exception as exc:
            st.error(f"Infra is not ready yet. Start infra and retry in a few seconds. Details: {exc}")
            return

        for uploaded in uploads:
            destination = os.path.join("data", uploaded.name)
            with open(destination, "wb") as file:
                file.write(uploaded.getbuffer())

            with st.spinner(f"Ingesting {uploaded.name}..."):
                count = ingest_pdf(
                    file_path=destination,
                    embedding_cfg=runtime["embedding"],
                    ingestion_cfg=config["ingestion"],
                    elastic_index=config["indexes"]["elastic_index_name"],
                    qdrant_collection=config["indexes"]["qdrant_collection_name"],
                )
                total_chunks += count

        st.success(f"Ingestion complete. Indexed {total_chunks} chunks.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    with tab_chat:
        st.caption(f"Active retrieval mode: {retrieval_mode}")

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and message.get("sources") and show_sources:
                    with st.expander(f"Sources ({len(message['sources'])})"):
                        render_sources(
                            hits=message["sources"],
                            scores=message["scores"],
                            retrieval_mode=message.get("retrieval_mode", retrieval_mode),
                            preview_chars=config["sources"]["preview_chars"],
                        )

        query = st.chat_input("Ask a question about your uploaded PDFs")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.spinner("Running retrieval, reranking, and generation..."):
                request_start = time.perf_counter()
                try:
                    retriever.ensure_stores()
                except Exception as exc:
                    st.error(f"Infra is not ready yet. Start infra and retry in a few seconds. Details: {exc}")
                    return

                query_vector = get_embedding(query, runtime["embedding"])
                hits = retriever.search(query=query, query_vector=query_vector, mode=retrieval_mode)

                if not hits:
                    st.warning("No results found. Ingest at least one PDF first.")
                    return

                # Reranking only makes sense for hybrid mode — keyword and vector
                # already return a ranked list from a single retriever.
                if config["reranking"]["enabled"] and retrieval_mode == "hybrid":
                    reranker = get_reranker(config["reranking"]["model_name"])
                    reranked_hits, rerank_scores = reranker.rerank(
                        query=query,
                        hits=hits,
                        top_n=config["reranking"]["top_n"],
                    )
                else:
                    reranked_hits = hits[: config["reranking"]["top_n"]]
                    rerank_scores = [float(hit.get("score", 0.0)) for hit in reranked_hits]

                context = "\n\n".join([item["text"] for item in reranked_hits])
                history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                    if msg["role"] in {"user", "assistant"}
                ]

                answer = generate_answer(
                    query=query,
                    context=context,
                    generator_cfg=runtime["generator"],
                    history=history,
                )
                latency_ms = (time.perf_counter() - request_start) * 1000.0
                logger.info(
                    "Query completed | mode=%s embedding=%s generator=%s latency_ms=%.2f sources=%d",
                    retrieval_mode,
                    runtime["embedding"]["provider"],
                    runtime["generator"]["provider"],
                    latency_ms,
                    len(reranked_hits),
                )

            if config["evaluation"]["enabled"]:
                interaction = make_interaction_record(
                    query=query,
                    answer=answer,
                    context=context,
                    retrieval_mode=retrieval_mode,
                    embedding_provider=runtime["embedding"]["provider"],
                    generator_provider=runtime["generator"]["provider"],
                    hits=reranked_hits,
                    scores=rerank_scores,
                    latency_ms=latency_ms,
                )
                append_interaction_log(eval_log_path, interaction)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": reranked_hits,
                    "scores": rerank_scores,
                    "retrieval_mode": retrieval_mode,
                }
            )
            st.rerun()

    with tab_eval:
        if not config["evaluation"]["enabled"]:
            st.info("Evaluation is disabled in config. Enable it to see metrics.")
        else:
            st.subheader("Evaluation Snapshot")

            logs_df = load_logs(eval_log_path)
            if logs_df.empty:
                st.info("No evaluation records yet. Ask a question in Chat to create the first record.")
            else:
                summary = summarize_logs(logs_df)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Queries", summary["total_queries"])
                m2.metric("Avg Latency (ms)", f"{summary['avg_latency_ms']:.1f}")
                m3.metric("P95 Latency (ms)", f"{summary['p95_latency_ms']:.1f}")
                m4.metric("Avg Grounding", f"{summary['avg_grounding_overlap']:.2f}")

                with st.expander("Latency Trend", expanded=False):
                    trend_df = logs_df[["timestamp", "latency_ms", "grounding_overlap"]].copy()
                    trend_df = trend_df.tail(config["evaluation"]["recent_rows"])
                    trend_df = trend_df.set_index("timestamp")
                    st.line_chart(trend_df)

                with st.expander("Recent Evaluation Records", expanded=False):
                    cols = [
                        "timestamp",
                        "retrieval_mode",
                        "embedding_provider",
                        "generator_provider",
                        "latency_ms",
                        "grounding_overlap",
                        "sources_used",
                        "top_source",
                        "query",
                    ]
                    display_df = logs_df[cols].tail(config["evaluation"]["recent_rows"])
                    st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
