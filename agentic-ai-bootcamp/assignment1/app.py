from __future__ import annotations

import os

# Workaround for macOS OpenMP duplicate runtime crash that can happen when FAISS loads.
# Safer fix is to ensure a single OpenMP runtime, but this unblocks local development.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from rag_pipeline import (
    RAGConfig,
    build_and_save_vectorstore,
    default_config,
    load_vectorstore,
    make_openai_llm,
    rag_answer,
)


st.set_page_config(page_title="Assignment 1 - RAG Chatbot (OpenAI)", layout="centered")
st.title("Assignment 1 - RAG Chatbot (OpenAI)")

cfg = default_config()

@st.cache_resource
def get_vectorstore(_cfg: RAGConfig):
    return load_vectorstore(_cfg)

@st.cache_resource
def get_llm():
    return make_openai_llm()


vectorstore = get_vectorstore(cfg)

if vectorstore is None:
    st.info("Vectorstore not found yet. Build it from PDFs in `assignment1/data/`.")
    if st.button("Build vectorstore"):
        with st.spinner("Loading PDFs, chunking, embedding, and building FAISS index..."):
            build_and_save_vectorstore(cfg)
            st.cache_resource.clear()
        st.rerun()
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

col_left, col_right = st.columns([1, 1])
with col_left:
    if st.button("Clear chat"):
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
        st.session_state.messages.append({"role": "user", "content": question})

        llm = get_llm()
        with st.spinner("Generating answer..."):
            answer, _meta = rag_answer(question=question, vectorstore=vectorstore, llm=llm, cfg=cfg)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

