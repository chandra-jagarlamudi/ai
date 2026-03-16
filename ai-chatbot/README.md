# AI Chatbot

A simple Streamlit chatbot with a tab per AI provider. All providers load on startup — any provider with a key set in `.env` gets its own tab.

## Providers

| Tab         | Key required     | Model env var   | Default                    |
|-------------|------------------|-----------------|----------------------------|
| OpenAI      | `OPENAI_API_KEY` | `OPENAI_MODEL`  | `gpt-4o-mini`              |
| Gemini      | `GOOGLE_API_KEY` | `GEMINI_MODEL`  | `gemini-pro`               |
| HuggingFace | `HF_API_TOKEN`   | `HF_CHAT_MODEL` | `Qwen/Qwen2.5-7B-Instruct` |
| Ollama      | _(none — local)_ | `OLLAMA_MODEL`  | `llama2`                   |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `.env` with your API keys, then run:

```bash
streamlit run app.py
```

## File Structure

```
ai-chatbot/
├── app.py                # Streamlit UI — one tab per provider
├── openai_chat.py        # OpenAI
├── gemini_chat.py        # Google Gemini
├── huggingface_chat.py   # HuggingFace Inference API
├── ollama_chat.py        # Ollama (local)
├── .env                  # API keys and model config
└── requirements.txt
```

## Notes

- Tabs are only shown for providers that have their key configured
- Ollama tab always shows — requires `ollama serve` running locally and the model pulled (`ollama pull <model>`)
- HuggingFace uses the Inference API only (no local model download)

## HuggingFace — Inference Provider Setup

HuggingFace now routes inference through third-party compute providers (featherless-ai, Together AI, Fireworks, etc.).
Two things must be configured:

1. **Enable a provider** on your HF account: [huggingface.co/settings/inference-providers](https://huggingface.co/settings/inference-providers)
2. **Set `HF_PROVIDER` in `.env`** to the provider you enabled (e.g. `featherless-ai`)
3. **Set `HF_CHAT_MODEL`** to a model hosted by that provider

```env
HF_API_TOKEN=hf_...
HF_PROVIDER=featherless-ai
HF_CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
```
