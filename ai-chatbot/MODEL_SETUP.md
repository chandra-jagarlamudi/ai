# Multi-Model Chatbot Configuration Guide

This chatbot now supports switching between multiple AI model providers. Configure your preferred model type in `.env` using the `MODEL_TYPE` setting.

## Quick Start

To switch models, edit the `.env` file and change:

```
MODEL_TYPE=huggingface
```

Choose one of: `openai`, `gemini`, `huggingface`, `ollama`

### Quick Switch: Local vs API (Hugging Face)

**To run google/gemma-3-1b-pt LOCALLY (no API key needed):**

```env
MODEL_TYPE=huggingface
USE_LOCAL_MODEL=true
HF_CHAT_MODEL=google/gemma-3-1b-pt
```

**To run via HF Inference API (requires API key):**

```env
MODEL_TYPE=huggingface
USE_LOCAL_MODEL=false
HF_CHAT_MODEL=google/gemma-3-1b-pt
HF_API_TOKEN=hf_...
```

---

## Model Configuration

### 1. **OpenAI** (GPT-3.5/GPT-4)

**Setup:**

```env
MODEL_TYPE=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4, etc.
```

**Requirements:**

- Install: `pip install langchain-openai`
- Get API key: https://platform.openai.com/api-keys
- Add to `.env`: `OPENAI_API_KEY=sk-...`

**Cost:** Pay-per-use API calls

---

### 2. **Google Gemini** (Gemini Pro)

**Setup:**

```env
MODEL_TYPE=gemini
GOOGLE_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-pro
```

**Requirements:**

- Install: `pip install google-generativeai`
- Get API key: https://makersuite.google.com/app/apikey
- Add to `.env`: `GOOGLE_API_KEY=AIzaSy...`

**Cost:** Free tier available + pay-per-use

---

### 3. **Hugging Face** (API or Local)

#### 3a. Using Hugging Face Inference API (Remote)

**Setup:**

```env
MODEL_TYPE=huggingface
USE_LOCAL_MODEL=false
HF_API_TOKEN=your-token-here
HF_CHAT_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**Requirements:**

- Install: `pip install huggingface-hub`
- Get token: https://huggingface.co/settings/tokens
- Add to `.env`: `HF_API_TOKEN=hf_...`

**Cost:** Free tier available + pay-per-use

**Popular Models:**

- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-7b-chat-hf`
- `tiiuae/falcon-7b-instruct`

#### 3b. Running Locally (Transformers + PyTorch)

**Setup:**

```env
MODEL_TYPE=huggingface
USE_LOCAL_MODEL=true
HF_CHAT_MODEL=google/gemma-3-1b-pt
HF_DEVICE=auto
HF_MAX_TOKENS=512
HF_USE_8BIT=false
```

**Configuration Options:**

- `USE_LOCAL_MODEL`: Set to `true` to run locally, `false` for API
- `HF_DEVICE`: Device to run model on:
  - `auto` (recommended): Automatically use GPU if available
  - `cuda`: Force NVIDIA GPU
  - `cpu`: Force CPU (slower)
- `HF_MAX_TOKENS`: Maximum tokens to generate (512 is good default)
- `HF_USE_8BIT`: Enable 8-bit quantization (reduces memory, requires bitsandbytes)

**Requirements:**

- Install: `pip install transformers torch accelerate`
- Optional for 8-bit: `pip install bitsandbytes`
- First run will download the model (~2-15GB depending on model size)
- GPU recommended (NVIDIA CUDA or Apple Metal)

**Cost:** Free (runs locally, no API calls)

**Recommended Lightweight Models for Local:**

- `google/gemma-3-1b-pt` (~3GB) - Small, fast ⭐ Recommended
- `microsoft/phi-2` (~5GB) - Very capable
- `microsoft/phi-1.5` (~3GB) - Good balance
- `mistralai/Mistral-7B-Instruct-v0.2` (~14GB) - Better quality, slower
- `meta-llama/Llama-2-7b-chat-hf` (~14GB) - High quality

**Google Gemma 3 1B Specific:**

```env
HF_CHAT_MODEL=google/gemma-3-1b-pt
HF_MAX_TOKENS=512
HF_USE_8BIT=false  # Set to true if running out of memory
```

This is a lightweight model (~3GB) optimized for low-resource environments.

**Memory Optimization:**

If you encounter out-of-memory errors:

```env
# Option 1: Use 8-bit quantization
HF_USE_8BIT=true

# Option 2: Reduce max tokens
HF_MAX_TOKENS=256

# Option 3: Reduce device to CPU (slower but more stable)
HF_DEVICE=cpu
```

**Switching Between Local and API:**

```env
# To use LOCAL model (no API key needed, runs on your machine)
USE_LOCAL_MODEL=true
# HF_API_TOKEN=... (not needed)

# To use HF INFERENCE API (needs API key, runs on HF servers)
USE_LOCAL_MODEL=false
HF_API_TOKEN=hf_... (required)
```

---

### 4. **Ollama** (Local LLMs)

**Setup:**

```env
MODEL_TYPE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

**Requirements:**

1. Install Ollama: https://ollama.ai
2. Start Ollama server: `ollama serve`
3. Pull a model: `ollama pull llama2` (or other models)
4. Install Python package: `pip install ollama`

**Cost:** Free (runs locally)

**Available Models:**

- `ollama pull llama2` - Llama 2 (7B, ~4GB)
- `ollama pull mistral` - Mistral (7B, ~4GB)
- `ollama pull neural-chat` - Neural Chat (7B, ~4GB)
- `ollama pull dolphin-mixtral` - Dolphin Mixtral (8x7B, ~26GB)

**Typical Workflow:**

```bash
# Terminal 1 - Start Ollama server
ollama serve

# Terminal 2 - Pull a model
ollama pull llama2

# Terminal 3 - Run chatbot
streamlit run ai_chatbot.py
```

---

## Installation

Install all dependencies for all providers:

```bash
pip install -r requirements.txt
```

Or install selectively:

```bash
# OpenAI only
pip install langchain-openai

# Gemini only
pip install google-generativeai

# Hugging Face only
pip install huggingface-hub transformers torch accelerate

# Ollama only
pip install ollama
```

---

## Troubleshooting

### "OPENAI_API_KEY not set in environment"

- Ensure `.env` file exists and contains your key
- Check `.env` file is in the same directory as `ai_chatbot.py`
- Run: `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"`

### "Failed to initialize Gemini provider"

- Verify API key is valid: https://makersuite.google.com/app/apikey
- Ensure `google-generativeai` is installed

### "Ollama server not responding"

- Make sure Ollama server is running: `ollama serve`
- Check `OLLAMA_BASE_URL` is correct (default: `http://localhost:11434`)
- Ensure model is downloaded: `ollama pull llama2`

### "Access to model is restricted" or "401 Unauthorized" for Gated Models

This error occurs when using gated models like `google/gemma-3-1b-pt`. Follow these steps:

1. **Accept the model license on Hugging Face:**
   - Go to https://huggingface.co/google/gemma-3-1b-pt
   - Click "Agree and access repository"
   - Log in if needed

2. **Use your Hugging Face token locally:**
   ```bash
   huggingface-cli login
   # Paste your HF token when prompted
   ```

3. **Add token to `.env` file:**
   ```env
   HF_TOKEN=hf_...
   HF_API_TOKEN=hf_...
   ```

4. **Try running again:**
   ```bash
   streamlit run ai_chatbot.py
   ```

The code now supports both `HF_TOKEN` and `HF_API_TOKEN` for authentication with local models.

### "Out of memory" with local Hugging Face models

- Try a smaller model: `google/gemma-3-1b-pt` or `microsoft/phi-2`
- Enable 8-bit quantization in `.env`: `HF_USE_8BIT=true`
- Reduce max tokens in `.env`: `HF_MAX_TOKENS=256`
- Use CPU instead of GPU: `HF_DEVICE=cpu` (slower, more stable)

### Model takes too long to load

- First load downloads model (~4-15GB) - subsequent loads are faster
- Consider using Ollama or API-based solutions for faster startup

---

## Performance Comparison

| Provider         | Speed     | Cost | Local | Quality   | Best For         |
| ---------------- | --------- | ---- | ----- | --------- | ---------------- |
| OpenAI           | Very Fast | $$$  | ❌    | Excellent | Production       |
| Gemini           | Very Fast | $$   | ❌    | Excellent | Budget + Quality |
| HF API           | Fast      | $$   | ❌    | Good      | Testing          |
| HF Local (Gemma) | Medium    | Free | ✅    | Good      | Low-Resource     |
| HF Local (7B+)   | Slow\*    | Free | ✅    | Very Good | Desktop/Server   |
| Ollama           | Medium    | Free | ✅    | Good      | Easy Setup       |

\*First load is slow (downloads model); subsequent loads faster

---

## Recommended Setup by Use Case

**For Local Execution (Gemma 3 1B - Recommended for this setup):**

```env
MODEL_TYPE=huggingface
USE_LOCAL_MODEL=true
HF_CHAT_MODEL=google/gemma-3-1b-pt
HF_DEVICE=auto
HF_MAX_TOKENS=512
HF_USE_8BIT=false
```

**For API Execution (via Hugging Face Inference API):**

```env
MODEL_TYPE=huggingface
USE_LOCAL_MODEL=false
HF_CHAT_MODEL=google/gemma-3-1b-pt
HF_API_TOKEN=hf_...  # (required)
```

**For Development/Testing (API):**

```env
MODEL_TYPE=huggingface
USE_LOCAL_MODEL=false
HF_API_TOKEN=hf_...
```

**For Production (Best Quality):**

```env
MODEL_TYPE=openai
OPENAI_API_KEY=sk-...
```

**For Local Deployment (No API Keys):**

```env
MODEL_TYPE=ollama
```

**For Budget-Conscious:**

```env
MODEL_TYPE=gemini
GOOGLE_API_KEY=AIzaSy...
```

---

## File Structure

```
rag_chatbot/
├── ai_chatbot.py        # Main chatbot with multi-provider support
├── .env                 # Configuration (add your API keys here)
├── requirements.txt     # Python dependencies
└── MODEL_SETUP.md      # This guide
```

---

## Next Steps

1. Choose a model provider from above
2. Update `.env` with your choice and credentials
3. Install dependencies: `pip install -r requirements.txt`
4. Run the chatbot: `streamlit run ai_chatbot.py`
