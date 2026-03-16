"""
AI Chatbot with mandatory Langfuse tracing.
Four tabs: OpenAI, Gemini, Hugging Face, Ollama. Each tab is a model-specific chatbot.
All configured providers load at startup; every request is traced in Langfuse.
"""
import streamlit as st

import config

# Config loads .env and Langfuse; fail fast if Langfuse keys missing
PROVIDER_KEYS = ("openai", "gemini", "huggingface", "ollama")
TAB_LABELS = {"openai": "OpenAI", "gemini": "Gemini", "huggingface": "Hugging Face", "ollama": "Ollama"}
CONFIG_HINTS = {
    "openai": "Set OPENAI_API_KEY and OPENAI_MODEL in .env to use this model.",
    "gemini": "Set GOOGLE_API_KEY and GEMINI_MODEL in .env to use this model.",
    "huggingface": "Set HF_API_TOKEN (or USE_LOCAL_MODEL and HF_CHAT_MODEL) in .env to use this model.",
    "ollama": "Set OLLAMA_MODEL (and OLLAMA_BASE_URL if needed) in .env. Ensure Ollama is running.",
}


def init_session_state():
    """Load all providers once and init per-tab history."""
    if "providers" not in st.session_state:
        providers, errors = config.get_all_providers()
        st.session_state.providers = providers
        st.session_state.provider_errors = errors
    for key in PROVIDER_KEYS:
        if f"history_{key}" not in st.session_state:
            st.session_state[f"history_{key}"] = []


def render_chat_tab(provider_key: str):
    """Render one tab: either config hint or chat UI."""
    provider = st.session_state.providers.get(provider_key)
    history = st.session_state[f"history_{provider_key}"]

    if provider is None:
        st.info(CONFIG_HINTS[provider_key])
        if provider_key in st.session_state.provider_errors:
            st.caption(f"Error: {st.session_state.provider_errors[provider_key]}")
        return

    model_info = getattr(provider, "model_name", None)
    caption = f"Using {provider.name}" + (f" ({model_info})" if model_info else "") + ". All requests are traced in Langfuse."
    st.caption(caption)

    for role, text in history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

    with st.form(f"input_form_{provider_key}", clear_on_submit=True):
        user_input = st.text_input("You:", key=f"input_{provider_key}")
        submit = st.form_submit_button("Send")

    if submit and user_input:
        st.session_state[f"history_{provider_key}"].append(("user", user_input))
        with st.spinner("Generating response..."):
            response = config.query_with_trace(provider_key, user_input)
        st.session_state[f"history_{provider_key}"].append(("bot", response))
        st.rerun()


def main():
    st.title("AI Powered Chatbot (Langfuse)")

    init_session_state()

    tab_names = [TAB_LABELS[k] for k in PROVIDER_KEYS]
    tabs = st.tabs(tab_names)

    for i, provider_key in enumerate(PROVIDER_KEYS):
        with tabs[i]:
            render_chat_tab(provider_key)


if __name__ == "__main__":
    main()
