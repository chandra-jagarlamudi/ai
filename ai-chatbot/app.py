import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

import openai_chat
import gemini_chat
import huggingface_chat
import ollama_chat

ALL_PROVIDERS = [openai_chat, gemini_chat, huggingface_chat, ollama_chat]


def render_tab(provider):
    # Session state key per provider
    key = f"history_{provider.NAME}"
    if key not in st.session_state:
        st.session_state[key] = []

    # Display chat history
    for role, text in st.session_state[key]:
        with st.chat_message(role):
            st.write(text)

    # Chat input
    if prompt := st.chat_input(f"Message {provider.NAME}...", key=f"input_{provider.NAME}"):
        st.session_state[key].append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = provider.chat(prompt)
            st.write(response)

        st.session_state[key].append(("assistant", response))


def main():
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
    st.title("🤖 AI Chatbot")

    available = [p for p in ALL_PROVIDERS if p.is_configured()]

    if not available:
        st.error("No providers configured. Add API keys to your .env file.")
        return

    tabs = st.tabs([p.NAME for p in available])
    for tab, provider in zip(tabs, available):
        with tab:
            render_tab(provider)


if __name__ == "__main__":
    main()
