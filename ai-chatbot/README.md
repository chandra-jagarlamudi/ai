# AI Chatbot

A Streamlit chatbot with a tab per AI provider. Providers with a key configured in `.env` get their own tab automatically. The Ollama tab is always shown — no key needed, runs fully local.

---

## Providers

| Tab         | Key required     | Model env var   | Default                    | Model selector |
|-------------|------------------|-----------------|----------------------------|----------------|
| OpenAI      | `OPENAI_API_KEY` | `OPENAI_MODEL`  | `gpt-4o-mini`              | Yes — comma-separated list |
| Gemini      | `GOOGLE_API_KEY` | `GEMINI_MODEL`  | `gemini-pro`               | No             |
| HuggingFace | `HF_API_TOKEN`   | `HF_CHAT_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | No             |
| Ollama      | _(none — local)_ | `OLLAMA_MODEL`  | `qwen3:8b-q4_K_M`          | Yes — lists all pulled models |

All providers maintain full **conversation history** — each request includes all previous messages so the model has context of the entire conversation.

---

## File Structure

```
ai-chatbot/
├── app.py                # Streamlit UI — tabs, sidebar, streaming, model selectors
├── openai_chat.py        # OpenAI provider — multi-model support, history
├── gemini_chat.py        # Google Gemini provider — history via start_chat()
├── huggingface_chat.py   # HuggingFace Inference API provider — history
├── ollama_chat.py        # Ollama provider — streaming, model listing, history
├── db.py                 # SQLite helpers — sessions + messages
├── Dockerfile            # Builds the app into a container image
├── docker-compose.yml    # Runs the container with correct networking + volumes
├── .dockerignore         # Keeps secrets and build noise out of the image
├── .env                  # Your local secrets (never committed)
├── .env.example          # Template — copy to .env and fill in
└── requirements.txt      # Python dependencies
```

---

## Option 1 — Run locally (no Docker)

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **What is a virtual environment?**
> A `.venv` folder that holds an isolated copy of Python and your packages.
> This prevents version conflicts between projects on your machine.

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in any API keys you want to use
```

> **What is `.env`?**
> A plain text file of `KEY=VALUE` pairs loaded into the process at startup by
> `python-dotenv`. It keeps secrets out of code. Never commit this file.

### 3. Pull the Ollama model

```bash
ollama pull qwen3:8b-q4_K_M
```

> **What is Ollama?**
> Ollama is a local server that downloads and runs open-weight LLMs on your
> machine. It exposes an HTTP API at `http://localhost:11434`. The Python
> `ollama` package is a thin wrapper around that API.
>
> `qwen3:8b-q4_K_M` means:
> - **Qwen3** — model family by Alibaba
> - **8b** — 8 billion parameters (mid-size, runs well on consumer hardware)
> - **q4_K_M** — 4-bit quantized, "K_M" quality level (smaller file, modest
>   quality loss vs full precision)

### 4. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Option 2 — Run in Docker

Docker packages the app and all its dependencies into an isolated **container**
so the environment is identical everywhere — your machine, a teammate's machine,
or a cloud server.

### Concepts

**Image vs Container**
An image is the blueprint (built from `Dockerfile`). A container is a running
instance of that image — like a class vs an object in code. You can run many
containers from one image.

**Layer caching**
Docker builds images in layers, one per `Dockerfile` instruction. If a layer's
input hasn't changed, Docker reuses the cached version. That's why
`requirements.txt` is copied and installed *before* the rest of the code — a
code change won't force a full `pip install` again.

**env_file vs volume-mounting `.env`**
`env_file` is the Docker-native way to pass secrets. Docker Compose reads your
`.env` on the host and injects each `KEY=VALUE` as a real environment variable
inside the container. The file itself is never copied into the image.
This is how `OPENAI_API_KEY`, `HF_API_TOKEN`, `GOOGLE_API_KEY`, etc. reach
the container — no special networking, just normal environment variables.

**Why cloud APIs (OpenAI, HuggingFace, Gemini) work from Docker without any extra setup**
These are HTTP APIs on the public internet. Docker containers have outbound
internet access by default (they share the host's network via NAT). The
container calls `api.openai.com` the same way your terminal does — the only
thing needed is the API key in the environment.

**`environment:` vs `env_file:` precedence**
Both can set the same variable. `environment:` always wins. That's why
`OLLAMA_BASE_URL=http://host.docker.internal:11434` is set under `environment:`
— it overrides whatever value might be in `.env`, ensuring the container always
talks to host Ollama even if you have a different URL in your local `.env`.

**Volumes**
A volume mounts a host path into the container. Used here for:
- `./data` — directory holding `chat_history.db`, persists across container
  restarts (otherwise the database would be lost when the container stops)

> **Why mount a directory and not the `.db` file directly?**
> If you mount a file path that doesn't exist on the host yet, Docker creates
> it as a *directory* instead of a file. SQLite then fails because it can't
> open a directory as a database. Mounting a directory sidesteps this — Docker
> creates `./data/` as a directory, and the app creates `chat_history.db`
> inside it at runtime.

**Ports**
Containers have their own network. `"8501:8501"` in `docker-compose.yml` means
*forward port 8501 on your Mac to port 8501 inside the container*, making the
app reachable at `http://localhost:8501`.

**`host.docker.internal`**
This is a special DNS hostname that Docker gives containers so they can reach
services running on the host machine (your Mac). It resolves to your host's
internal IP address from inside the container.

Why does this matter? Ollama runs on your Mac natively so it can use the Apple
Silicon GPU (Metal). Docker containers on macOS cannot access Metal directly,
so running Ollama inside a container would force it to use CPU — roughly 10×
slower. The correct setup is:

```
Your Mac
  ├── Ollama (native, uses Metal/GPU)  ← http://host.docker.internal:11434
  └── Docker container
        └── Streamlit app  →  calls Ollama via host.docker.internal
```

The `extra_hosts: host.docker.internal:host-gateway` line in
`docker-compose.yml` makes this work on Linux too (Docker Desktop on Mac sets
it up automatically).

### Steps

**1. Make sure Ollama is running on your host**

```bash
ollama serve          # starts the Ollama server if it isn't already running
ollama pull qwen3:8b-q4_K_M
```

**2. Point the app at the host Ollama**

In your `.env`, set:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

**3. Build and start**

```bash
docker compose up --build
```

`--build` forces a fresh image build. Omit it on subsequent runs if the code
hasn't changed — it will reuse the cached image and start faster.

**4. Open the app**

```
http://localhost:8501
```

**Useful Docker commands**

```bash
docker compose up --build        # build image + start container
docker compose up                # start with existing image
docker compose down              # stop and remove container
docker compose logs -f           # tail logs in real time
docker compose ps                # show running containers
docker image ls                  # list all built images on your machine
docker system prune              # clean up unused images/containers/volumes
```

---

## OpenAI tab — multiple models

Set `OPENAI_MODEL` in `.env` as a comma-separated list to enable a model
selector dropdown in the OpenAI tab:

```env
OPENAI_MODEL=gpt-4o-mini-2024-07-18,gpt-4.1-nano-2025-04-14,gpt-5-nano-2025-08-07
```

- The first model in the list is the default selected on page load.
- The selected model is remembered in the browser session (switching tabs and
  returning keeps your selection).
- To use a single model, just set one value with no commas — the dropdown is
  hidden and that model is used directly.

---

## Ollama tab features

- **Model selector** — dropdown lists every model you have pulled locally (`ollama list`)
- **Streaming** — tokens appear word-by-word instead of waiting for the full response
- **Persistent history** — conversations are saved to SQLite (`data/chat_history.db`)
- **Session sidebar** — past conversations show the model name and the opening message; click to reload, ✕ to delete

```
Sidebar session entry:
  qwen3:8b-q4_K_M          ← model used for this conversation
  [Tell me about Python...] [✕]
```

---

## Conversation history (all providers)

Every provider passes the full conversation history to the API on each request.
The model sees all previous turns — not just the current message.

```python
# What gets sent to the API on every request:
messages = [
    {"role": "user",      "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user",      "content": "How does it compare to JavaScript?"},  # current
]
```

**Gemini role name difference**

Gemini uses different field names compared to OpenAI and HuggingFace:

| OpenAI / HuggingFace | Gemini |
|---|---|
| `role: "assistant"` | `role: "model"` |
| `{"role": "user", "content": "hi"}` | `{"role": "user", "parts": ["hi"]}` |
| `client.chat.completions.create(messages=...)` | `model.start_chat(history=...).send_message(prompt)` |

The app converts the shared history format to each provider's expected format
automatically before sending.

---

## Persistent chat history (SQLite — Ollama only)

Chat history for the Ollama tab is persisted to disk in a SQLite database so
conversations survive page refreshes and container restarts.

> **What is SQLite?**
> An embedded database — the entire database lives in a single `.db` file on
> disk. No server process, no installation. Python ships with `sqlite3` in its
> standard library. It's the right choice for local apps that need structured
> storage without the overhead of Postgres or MySQL.

**Schema**

```sql
sessions
  id          TEXT PRIMARY KEY   -- UUID, e.g. "f3a1…"
  provider    TEXT               -- "Ollama"
  name        TEXT               -- first 50 chars of the opening message
  model       TEXT               -- model used (e.g. "qwen3:8b-q4_K_M")
  created_at  DATETIME
  updated_at  DATETIME           -- bumped on every new message

messages
  id          INTEGER PRIMARY KEY AUTOINCREMENT
  session_id  TEXT  →  sessions.id  (ON DELETE CASCADE)
  role        TEXT               -- "user" or "assistant"
  content     TEXT
  created_at  DATETIME
```

`ON DELETE CASCADE` means deleting a session automatically deletes all its
messages — no orphaned rows.

**Schema migrations**

The `model` column was added after the initial schema. `init_db()` runs this
migration on every startup using SQLite's `ALTER TABLE`:

```python
try:
    conn.execute("ALTER TABLE sessions ADD COLUMN model TEXT NOT NULL DEFAULT ''")
except sqlite3.OperationalError:
    pass  # column already exists — safe to ignore
```

This is the standard SQLite migration pattern. Unlike Postgres, SQLite has no
`ADD COLUMN IF NOT EXISTS`, so the try/except is the idiomatic alternative.

---

## HuggingFace — Inference Provider Setup

HuggingFace routes inference through third-party compute providers
(featherless-ai, Together AI, Fireworks, etc.):

1. Enable a provider on your HF account: [huggingface.co/settings/inference-providers](https://huggingface.co/settings/inference-providers)
2. Set `HF_PROVIDER` in `.env` to match (e.g. `featherless-ai`)
3. Set `HF_CHAT_MODEL` to a model hosted by that provider

```env
HF_API_TOKEN=hf_...
HF_PROVIDER=featherless-ai
HF_CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
```

---

## How streaming works

Most LLMs generate text token-by-token internally. Without streaming, the
server waits until all tokens are ready before sending the response — you see
nothing, then everything at once.

With streaming, each token is sent to the client as soon as it's generated.
In this app:

1. `ollama_chat.stream_chat()` calls the Ollama API with `stream=True`, which
   returns a generator that yields one chunk per token.
2. Streamlit's `st.write_stream()` consumes that generator and renders each
   token to the screen immediately.
3. At the end, `st.write_stream()` returns the complete assembled string, which
   is then saved to the database.
4. `st.rerun()` is called so the page rerenders cleanly from state, placing the
   input box below all messages ready for the next question.

```
Ollama server
  └── yields chunk {"message": {"content": "Hello"}}
      yields chunk {"message": {"content": " there"}}
      ...
        └── st.write_stream() renders each chunk live
              └── returns full string when generator is exhausted
                    └── saved to SQLite
                          └── st.rerun() → input resets below responses
```

---

## Why LLMs are stateless

Every LLM API — Ollama, OpenAI, Gemini, HuggingFace — has no memory between
calls. Each request must include the *entire* conversation history. The model
reads the full context window on every call and produces the next reply.

```
Turn 1:  send [user: "What is Python?"]
         ← "Python is a programming language..."

Turn 2:  send [user: "What is Python?", assistant: "Python is...", user: "Compare to JS?"]
         ← "Compared to JavaScript..."

Turn 3:  send [all 4 messages so far, user: "Which is faster?"]
         ← "It depends on the use case..."
```

The app stores all messages in `st.session_state` (in-memory, current browser
session) for all providers, and additionally in SQLite (persists across
refreshes) for the Ollama tab. On every submission, the full accumulated history
is passed to the API before the new prompt.
