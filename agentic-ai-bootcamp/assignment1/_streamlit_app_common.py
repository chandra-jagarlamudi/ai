from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st

from rag_pipeline import RAGConfig, build_and_save_vectorstore, default_config, load_vectorstore


def render_app(*, title: str, make_llm: Callable[[], object]) -> None:
    st.set_page_config(page_title=title, layout="centered")
    st.title(title)

    cfg = default_config()

    st.sidebar.header("Setup")
    st.sidebar.write(f"**Data dir**: `{cfg.data_dir}`")
    st.sidebar.write(f"**Vectorstore dir**: `{cfg.vectorstore_dir}`")
    st.sidebar.write(f"**Chunking**: size={cfg.chunk_size}, overlap={cfg.chunk_overlap}")
    st.sidebar.write(f"**Retriever**: top_k={cfg.top_k}")

    if not Path(cfg.data_dir).exists():
        st.sidebar.error("Data directory not found.")
        st.stop()

    @st.cache_resource
    def _get_vectorstore(_cfg: RAGConfig):
        return load_vectorstore(_cfg)

    vectorstore = _get_vectorstore(cfg)

    if vectorstore is None:
        st.sidebar.warning("Vectorstore not found. Build it from PDFs.")
        if st.sidebar.button("Build / Rebuild Vectorstore"):
            with st.spinner("Building vectorstore from PDFs..."):
                build_and_save_vectorstore(cfg)
                st.cache_resource.clear()
            st.rerun()
        st.info("Chat is disabled until the vectorstore is built.")
        st.stop()
    else:
        if st.sidebar.button("Rebuild Vectorstore"):
            with st.spinner("Rebuilding vectorstore from PDFs..."):
                build_and_save_vectorstore(cfg)
                st.cache_resource.clear()
            st.rerun()
        st.sidebar.success("Vectorstore ready.")

    st.header("Chat")
    question = st.text_input("Ask a question about the documents in assignment1/data/")
    if not question:
        return

    from rag_pipeline import rag_answer

    llm = make_llm()
    with st.spinner("Retrieving context and generating answer..."):
        answer, meta = rag_answer(question=question, vectorstore=vectorstore, llm=llm, cfg=cfg)
    st.write(answer)

    with st.expander("Retrieved context metadata"):
        st.json(meta)

