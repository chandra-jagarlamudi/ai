# Personal Financial AI Agent

A conversational AI assistant for personal finance, built with **LangChain**, **LiteLLM**, and **Streamlit**. Switch between OpenAI, Anthropic, and Ollama with a single `.env` change.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                        │
│                                                                     │
│  ┌──────────────┐    ┌───────────────────────────────────────────┐  │
│  │   Sidebar    │    │              Main Chat Area               │  │
│  │              │    │                                           │  │
│  │ Tool Guides  │    │  User: "EMI for ₹10L at 8.5% for 20 yrs"  │  │
│  │ Examples     │    │  ──────────────────────────────────────── │  │
│  │              │    │  Assistant: 📊 EMI = ₹8,678 / month ...   │  │
│  │ [Clear Chat] │    │                                           │  │
│  └──────────────┘    └───────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  User message + Chat history
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangChain Agent (agent.py)                       │
│                                                                     │
│   System Prompt + Chat History + User Input                         │
│                         │                                           │
│                         ▼                                           │
│              ┌─────────────────────┐                                │
│              │       LiteLLM       │  ◄── LLM_MODEL in .env         │
│              │  (model router)     │                                │
│              │                     │                                │
│              │  openai/gpt-4o      │                                │
│              │  anthropic/claude-* │                                │
│              │  ollama/llama3      │                                │
│              └──────────┬──────────┘                                │
│                         │  OpenAI-style function calling            │
│                         ▼                                           │
│         ┌───────────────────────────────────┐                       │
│         │         Tool Selection            │                       │
│         └───┬───────┬───────┬───────┬───────┘                       │
│             │       │       │       │                               │
│             ▼       ▼       ▼       ▼       ▼                       │
│          ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                    │
│          │ EMI │ │ SIP │ │Budg │ │Stock│ │Calc │                    │
│          └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                    │
│             └───────┴───────┴───────┴───────┘                       │
│                         │  Tool result                              │
│                         ▼                                           │
│              Final response assembled by LLM                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                     Streamlit chat bubble
                     (with tool-call trace)
```

---

## Features

| Tool | Description | Key Formula |
|------|-------------|-------------|
| 📊 **EMI Calculator** | Monthly loan installment | `EMI = P·r·(1+r)^n / ((1+r)^n−1)` |
| 📈 **SIP Calculator** | Future value of monthly investments | `FV = P·((1+r)^n−1) / r` |
| 💼 **Budget Planner** | 50-30-20 income allocation | 50% Needs · 30% Wants · 20% Savings |
| 📉 **Stock Price** | Live quotes via Yahoo Finance | yfinance (no API key needed) |
| 🧮 **Calculator** | Safe NumPy math evaluator | sqrt, log, sin, cos, pi, e … |

---

## Project Structure

```
personal-financial-ai-agent/
├── app.py              ← Streamlit UI entry point
├── agent.py            ← LangChain agent + LiteLLM setup
├── tools/
│   ├── __init__.py     ← Exports ALL_TOOLS list
│   ├── emi_calculator.py
│   ├── sip_calculator.py
│   ├── budget_planner.py
│   ├── stock_price.py
│   └── calculator.py
├── requirements.txt
├── .env.example        ← Copy to .env and fill in your keys
└── README.md
```

---

## Setup

### 1. Create virtual environment

```bash
cd personal-financial-ai-agent
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your API key and desired model:

```env
LLM_MODEL=openai/gpt-4o
OPENAI_API_KEY=sk-...
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Switching Models

Change only the `LLM_MODEL` value in `.env` — no code changes needed.

| Provider | LLM_MODEL value | Key required |
|----------|----------------|--------------|
| OpenAI GPT-4o | `openai/gpt-4o` | `OPENAI_API_KEY` |
| OpenAI GPT-4o mini | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic Claude | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Ollama (local) | `ollama/llama3` | none (runs locally) |
| Ollama Mistral | `ollama/mistral` | none (runs locally) |

> **Note:** For Ollama, make sure the Ollama daemon is running (`ollama serve`) and the model is pulled (`ollama pull llama3`).

---

## Example Queries

### EMI Calculator
```
Calculate EMI for a ₹50,00,000 home loan at 8.5% interest for 20 years
What is the monthly payment on a $25,000 car loan at 7% for 4 years?
```

### SIP Calculator
```
If I invest ₹10,000 per month at 12% annual return for 15 years, how much will I accumulate?
SIP projection: $500/month, 10% return, 30 years
```

### Budget Planner
```
Help me plan my budget for a monthly income of ₹75,000
Create a 50-30-20 budget for $4,500 monthly take-home pay
```

### Stock Price
```
What is the current price of Apple stock?
Get me RELIANCE.NS stock info
Show Tesla stock price and 52-week range
```

### Calculator
```
What is sqrt(144) + 15% of 80000?
Calculate log10(1000) divided by pi
What is 2 to the power of 16?
```

---

## How the Agent Routes Queries

The agent uses **OpenAI-style function calling** (via LiteLLM):

1. The LLM reads the user message and the **docstring** of each tool.
2. It decides which tool (if any) best matches the query.
3. It calls the tool with structured arguments extracted from the message.
4. The tool returns a formatted result string.
5. The LLM assembles a final response and adds context/explanation.

Tool calls are visible in the chat UI as expandable "thought" sections (powered by `StreamlitCallbackHandler`).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Agent orchestration, tool framework |
| `langchain-community` | `ChatLiteLLM`, `StreamlitCallbackHandler` |
| `litellm` | Universal LLM proxy (OpenAI / Anthropic / Ollama) |
| `streamlit` | Web UI |
| `yfinance` | Yahoo Finance stock data (free, no API key) |
| `numpy` | Safe math expression evaluation |
| `python-dotenv` | `.env` file loading |
