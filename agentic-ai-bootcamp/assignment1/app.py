from __future__ import annotations

import os
import logging

# Workaround for macOS OpenMP duplicate runtime crash that can happen when FAISS loads.
# Safer fix is to ensure a single OpenMP runtime, but this unblocks local development.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

import streamlit as st

from rag_pipeline import (
    RAGConfig,
    build_and_save_vectorstore,
    comparison_config,
    default_config,
    get_llm,
    load_vectorstore,
    rag_answer,
)
from run_comparison import MODELS, TEST_QUESTIONS, run_comparison


cfg = default_config()
logger.info("Streamlit app start: provider=%s model=%s vectorstore=%s", cfg.llm_provider, cfg.llm_model, str(cfg.vectorstore_dir))

st.set_page_config(page_title=f"Assignment 1 - RAG Chatbot ({cfg.llm_provider})", layout="centered")
st.title(f"Assignment 1 - RAG Chatbot ({cfg.llm_provider})")
st.caption(f"LLM: `{cfg.llm_model}` | Embeddings: `{cfg.embedding_provider}:{cfg.embedding_model}` | Vectorstore: `{cfg.vectorstore_dir}`")

@st.cache_resource
def get_vectorstore(_cfg: RAGConfig):
    return load_vectorstore(_cfg)

@st.cache_resource
def get_cached_llm(_cfg: RAGConfig):
    return get_llm(_cfg)


# Chat uses provider-specific vectorstore (vectorstore_openai, vectorstore_ollama, vectorstore_gemini)
vectorstore = get_vectorstore(cfg)

# Sidebar: model comparison (uses single shared vectorstore_comparison with OpenAI embeddings)
with st.sidebar:
    st.header("Model comparison")
    comp_cfg = comparison_config()
    if st.button("Build comparison vectorstore"):
        logger.info("Comparison: build requested dir=%s", str(comp_cfg.vectorstore_dir))
        with st.spinner("Building comparison index (OpenAI embeddings)…"):
            try:
                build_and_save_vectorstore(comp_cfg)
                st.success(f"Saved to {comp_cfg.vectorstore_dir}")
                st.cache_resource.clear()
                logger.info("Comparison: build succeeded dir=%s", str(comp_cfg.vectorstore_dir))
            except Exception as e:
                st.error(str(e))
                logger.exception("Comparison: build failed dir=%s", str(comp_cfg.vectorstore_dir))
        st.rerun()

    comp_vectorstore = load_vectorstore(comp_cfg)
    if st.button("Run comparison (all 3 models)"):
        if comp_vectorstore is None:
            st.session_state.last_comparison_error = "Build the comparison vectorstore first (button above)."
            logger.warning("Comparison: run requested but comparison vectorstore missing")
        else:
            with st.spinner("Running test questions against OpenAI, Ollama, Gemini…"):
                try:
                    logger.info("Comparison: run started")
                    results, out_path = run_comparison(comp_vectorstore, comp_cfg)
                    st.session_state.last_comparison_results = results
                    st.session_state.last_comparison_path = str(out_path)
                    st.session_state.pop("last_comparison_error", None)
                    logger.info("Comparison: run succeeded results=%d out=%s", len(results), str(out_path))
                except Exception as e:
                    st.session_state.last_comparison_error = str(e)
                    logger.exception("Comparison: run failed")
        st.rerun()

    if "last_comparison_error" in st.session_state:
        st.error(st.session_state.last_comparison_error)

    if "last_comparison_results" in st.session_state:
        results = st.session_state.last_comparison_results
        st.success(f"Results saved to `{st.session_state.get('last_comparison_path', 'comparison_results.json')}`")
        # Single table: Question | Model | Time (s) | Response
        rows = [
            {
                "Question": r["question_label"],
                "Model": r["model_label"],
                "Time (s)": r.get("time_seconds") if r.get("time_seconds") is not None else "—",
                "Response": r.get("error") if r.get("error") else r.get("response", ""),
            }
            for r in results
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Fill COMPARISON_TABLE.md with Accuracy and Hallucination from manual review.")

if vectorstore is None:
    st.info("Vectorstore not found yet. Build it from PDFs in `assignment1/data/`.")
    if st.button("Build vectorstore"):
        logger.info("Chat: build requested dir=%s", str(cfg.vectorstore_dir))
        with st.spinner("Loading PDFs, chunking, embedding, and building FAISS index..."):
            build_and_save_vectorstore(cfg)
            st.cache_resource.clear()
        logger.info("Chat: build succeeded dir=%s", str(cfg.vectorstore_dir))
        st.rerun()
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

col_left, col_right = st.columns([1, 1])
with col_left:
    if st.button("Clear chat"):
        logger.info("Chat: clear requested")
        st.session_state.messages = []
        st.rerun()

st.divider()

# Display chat history (top)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input (renders under history and moves down as chat grows)
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input(
        "Question",
        placeholder="Ask a question about the PDFs in assignment1/data/",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Send")

if submitted:
    question = (question or "").strip()
    if question:
        logger.info("Chat: question submitted length=%d", len(question))
        st.session_state.messages.append({"role": "user", "content": question})

        llm = get_cached_llm(cfg)
        with st.spinner("Generating answer..."):
            answer, _meta = rag_answer(question=question, vectorstore=vectorstore, llm=llm, cfg=cfg)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        logger.info("Chat: answer generated length=%d", len(answer))
        st.rerun()

