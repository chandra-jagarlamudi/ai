import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

import openai_chat
import gemini_chat
import huggingface_chat
import ollama_chat
import db

ALL_PROVIDERS = [openai_chat, gemini_chat, huggingface_chat, ollama_chat]

# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------

# Ollama — list[dict] messages, DB session id, selected model
_MSGS = "ollama_messages"
_SID  = "ollama_session_id"
_MDL  = "ollama_model"

# OpenAI — same shape as Ollama so sidebar can load/clear both uniformly
_OAI_MSGS = "openai_messages"
_OAI_SID  = "openai_session_id"
_OAI_MDL  = "openai_model"

# ---------------------------------------------------------------------------
# Sidebar helpers — provider-agnostic load/clear/check
# ---------------------------------------------------------------------------

# Providers that persist to SQLite.  Others (Gemini, HF) are in-memory only.
_PERSISTENT = {
    "OpenAI": (_OAI_SID, _OAI_MSGS),
    "Ollama": (_SID,     _MSGS),
}


def _clear_session(provider_name: str):
    sid_key, msgs_key = _PERSISTENT[provider_name]
    st.session_state[sid_key]  = None
    st.session_state[msgs_key] = []


def _load_session(provider_name: str, session_id: str):
    sid_key, msgs_key = _PERSISTENT[provider_name]
    st.session_state[sid_key]  = session_id
    st.session_state[msgs_key] = db.load_messages(session_id)


def _is_active(provider_name: str, session_id: str) -> bool:
    sid_key, _ = _PERSISTENT[provider_name]
    return st.session_state.get(sid_key) == session_id


# ---------------------------------------------------------------------------
# Unified sidebar — all persisted providers in collapsible sections
# ---------------------------------------------------------------------------

def render_sidebar(available: list):
    # Only show sections for providers that are both configured and persistent
    persistent_available = [p for p in available if p.NAME in _PERSISTENT]
    if not persistent_available:
        return

    with st.sidebar:
        st.header("Chat History")

        for provider in persistent_available:
            pname    = provider.NAME
            sessions = db.list_sessions(pname)

            with st.expander(pname, expanded=True):
                if st.button("+ New Chat", key=f"new_{pname}", use_container_width=True):
                    _clear_session(pname)
                    st.rerun()

                if not sessions:
                    st.caption("No conversations yet.")
                    continue

                st.divider()
                for s in sessions:
                    col_name, col_del = st.columns([5, 1])
                    with col_name:
                        if s.get("model"):
                            st.caption(s["model"])
                        if st.button(s["name"], key=f"sess_{s['id']}", use_container_width=True):
                            _load_session(pname, s["id"])
                            st.rerun()
                    if col_del.button("✕", key=f"del_{s['id']}"):
                        db.delete_session(s["id"])
                        if _is_active(pname, s["id"]):
                            _clear_session(pname)
                        st.rerun()


# ---------------------------------------------------------------------------
# Ollama tab — streaming, model selector, DB persistence
# ---------------------------------------------------------------------------

def _init_ollama_state():
    if _MSGS not in st.session_state:
        st.session_state[_MSGS] = []
    if _SID not in st.session_state:
        st.session_state[_SID] = None
    if _MDL not in st.session_state:
        models  = ollama_chat.list_models()
        default = ollama_chat.DEFAULT_MODEL
        st.session_state[_MDL] = default if default in models else (models[0] if models else default)


def render_ollama_tab():
    _init_ollama_state()

    models = ollama_chat.list_models()
    if models:
        current  = st.session_state[_MDL]
        idx      = models.index(current) if current in models else 0
        selected = st.selectbox("Model", options=models, index=idx, label_visibility="collapsed")
        st.session_state[_MDL] = selected
    else:
        st.warning("No Ollama models found. Pull one first:\n```\nollama pull qwen3:8b-q4_K_M\n```")
        selected = st.session_state[_MDL]

    for msg in st.session_state[_MSGS]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Message Ollama...", key="input_Ollama"):
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state[_SID] is None:
            session_id = db.create_session("Ollama", prompt, model=selected)
            st.session_state[_SID] = session_id
        else:
            session_id = st.session_state[_SID]

        db.save_message(session_id, "user", prompt)
        st.session_state[_MSGS].append({"role": "user", "content": prompt})

        history_so_far = st.session_state[_MSGS][:-1]
        with st.chat_message("assistant"):
            response: str = st.write_stream(
                ollama_chat.stream_chat(prompt, history_so_far, selected)
            )

        db.save_message(session_id, "assistant", response)
        st.session_state[_MSGS].append({"role": "assistant", "content": response})
        st.rerun()


# ---------------------------------------------------------------------------
# OpenAI tab — model selector, DB persistence
# ---------------------------------------------------------------------------

def _init_openai_state():
    if _OAI_MSGS not in st.session_state:
        st.session_state[_OAI_MSGS] = []
    if _OAI_SID not in st.session_state:
        st.session_state[_OAI_SID] = None
    if _OAI_MDL not in st.session_state:
        st.session_state[_OAI_MDL] = openai_chat.DEFAULT_MODEL


def render_openai_tab():
    _init_openai_state()

    selected = st.selectbox(
        "Model",
        options=openai_chat.MODELS,
        index=openai_chat.MODELS.index(st.session_state[_OAI_MDL])
               if st.session_state[_OAI_MDL] in openai_chat.MODELS else 0,
        label_visibility="collapsed",
    )
    st.session_state[_OAI_MDL] = selected

    for msg in st.session_state[_OAI_MSGS]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Message OpenAI...", key="input_OpenAI"):
        history = list(st.session_state[_OAI_MSGS])  # snapshot before appending

        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state[_OAI_SID] is None:
            session_id = db.create_session("OpenAI", prompt, model=selected)
            st.session_state[_OAI_SID] = session_id
        else:
            session_id = st.session_state[_OAI_SID]

        db.save_message(session_id, "user", prompt)
        st.session_state[_OAI_MSGS].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = openai_chat.chat(prompt, history=history, model=selected)
            st.write(response)

        db.save_message(session_id, "assistant", response)
        st.session_state[_OAI_MSGS].append({"role": "assistant", "content": response})
        st.rerun()


# ---------------------------------------------------------------------------
# Generic tab — in-memory history only (Gemini, HuggingFace)
# ---------------------------------------------------------------------------

def render_tab(provider):
    key = f"history_{provider.NAME}"
    if key not in st.session_state:
        st.session_state[key] = []

    for role, text in st.session_state[key]:
        with st.chat_message(role):
            st.write(text)

    if prompt := st.chat_input(f"Message {provider.NAME}...", key=f"input_{provider.NAME}"):
        history = [{"role": r, "content": t} for r, t in st.session_state[key]]
        st.session_state[key].append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = provider.chat(prompt, history=history)
            st.write(response)

        st.session_state[key].append(("assistant", response))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db.init_db()

    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
    st.title("AI Chatbot")

    available = [p for p in ALL_PROVIDERS if p.is_configured()]
    if not available:
        st.error("No providers configured. Add API keys to your .env file.")
        return

    render_sidebar(available)

    tabs = st.tabs([p.NAME for p in available])
    for tab, provider in zip(tabs, available):
        with tab:
            if provider is ollama_chat:
                render_ollama_tab()
            elif provider is openai_chat:
                render_openai_tab()
            else:
                render_tab(provider)


if __name__ == "__main__":
    main()
