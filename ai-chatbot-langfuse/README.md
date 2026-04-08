# AI Chatbot (Langfuse)

A Streamlit chatbot with **mandatory Langfuse tracing** and four provider tabs: OpenAI, Gemini, HuggingFace, Ollama.
All providers load at startup; every request is traced in Langfuse.

See [ai-chatbot](../ai-chatbot/) for the simpler version without tracing.

## Prerequisites

- Python 3.12
- A Langfuse project (cloud or self-hosted) with its API keys

## Setup

```bash
cd ai-chatbot-langfuse
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Run:

```bash
streamlit run ai_chatbot.py
```

## Dev Container

This project includes a VS Code dev container for a consistent Python 3.12 setup.

1. Open the project in VS Code.
2. Run "Dev Containers: Reopen in Container".
3. Wait for the post-create step to create `.venv` and install dependencies from `requirements.txt`.

Then copy environment variables and run the app inside the container:

```bash
cp .env.example .env
. .venv/bin/activate
streamlit run ai_chatbot.py
```

The Streamlit app is available on forwarded port `8501`.
Langfuse does not need a forwarded port when you use Langfuse Cloud because the app sends traces outbound over HTTPS.

If you run self-hosted Langfuse on your host machine while the app runs inside the dev container, set `LANGFUSE_BASE_URL=http://host.docker.internal:3000` in `.env`.
Port `3000` only needs to be exposed if you also want to open the Langfuse UI in your browser.

## Environment Variables

### Required — Langfuse

| Variable               | Description                                               |
|------------------------|-----------------------------------------------------------|
| `LANGFUSE_PUBLIC_KEY`  | From your Langfuse project settings                       |
| `LANGFUSE_SECRET_KEY`  | From your Langfuse project settings                       |
| `LANGFUSE_BASE_URL`    | e.g. `http://host.docker.internal:3000` from the dev container, or your cloud URL   |

The app will **not start** if Langfuse keys are missing or invalid.

### Per-provider (optional)

Tabs for unconfigured providers show a setup hint instead of a chat UI.

| Tab         | Required key     | Model var       | Default                    |
|-------------|------------------|-----------------|----------------------------|
| OpenAI      | `OPENAI_API_KEY` | `OPENAI_MODEL`  | `gpt-4.1`                  |
| Gemini      | `GOOGLE_API_KEY` | `GEMINI_MODEL`  | `gemini-2.0-flash`         |
| HuggingFace | `HF_API_TOKEN`   | `HF_CHAT_MODEL` | `Qwen/Qwen2.5-7B-Instruct` |
| Ollama      | _(none)_         | `OLLAMA_MODEL`  | `llama2`                   |

## HuggingFace — Inference Provider Setup

HuggingFace now routes inference through third-party compute providers (featherless-ai, Together AI, Fireworks, etc.).
Two things must be configured:

1. **Enable a provider** on your HF account: [huggingface.co/settings/inference-providers](https://huggingface.co/settings/inference-providers)
2. **Set `HF_PROVIDER`** in `.env` to match your enabled provider (e.g. `featherless-ai`)
3. **Set `HF_CHAT_MODEL`** to a model hosted by that provider

```env
HF_API_TOKEN=hf_...
HF_PROVIDER=featherless-ai
HF_CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
```

**Local mode** (`USE_LOCAL_MODEL=true`): runs the model on your machine via `transformers`. Does not use the inference provider. Requires `HF_CHAT_MODEL` to be an instruction-tuned model (e.g. names ending in `-it` or `-Instruct`).

```env
USE_LOCAL_MODEL=true
HF_CHAT_MODEL=google/gemma-2-2b-it
HF_DEVICE=auto        # auto / cuda / cpu
HF_MAX_TOKENS=512
HF_USE_8BIT=false     # true reduces memory, needs bitsandbytes
```

## Ollama Setup

```bash
ollama serve
ollama pull llama2   # or any other model
```

Set `OLLAMA_MODEL` (and optionally `OLLAMA_BASE_URL`) in `.env`.

## Observability (Langfuse)

- **One trace per message** — every send in every tab creates a trace in Langfuse
- **OpenAI**: LangChain callback captures token usage and spans per trace
- **Gemini / HuggingFace / Ollama**: one generation span per message with input, output, model name, and latency

Inspect traces at your Langfuse URL to debug prompts, responses, and performance.

## File Structure

```
ai-chatbot-langfuse/
├── ai_chatbot.py       # Streamlit UI — 4 tabs, loads all providers at startup
├── config.py           # .env loading, Langfuse init, get_all_providers(), query_with_trace()
├── openai_chat.py      # OpenAI provider
├── gemini_chat.py      # Gemini provider
├── huggingface_chat.py # HuggingFace provider (API or local)
├── ollama_chat.py      # Ollama provider
├── .env                # API keys and config (do not commit)
├── .env.example        # Template
└── requirements.txt
```
