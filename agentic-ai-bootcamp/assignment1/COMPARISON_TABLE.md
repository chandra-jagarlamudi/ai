# Model Comparison: Testing & Comparison

Use **at least 3 test questions** covering:
- **Leave policies**
- **Remote work rules**
- **Startup pricing model**

Run the comparison script to get response times and full answers, then fill this table (and optionally add per-question notes below).

```bash
python run_comparison.py
```

---

## Comparison table

| Model | Accuracy | Hallucination | Response Speed | Cost per Query | Ease of Setup |
|-------|----------|---------------|----------------|----------------|---------------|
| **OpenAI (gpt-4o-mini)** | *(1–5 or Good/Fair/Poor)* | *(None / Minor / Major)* | *(from script, e.g. ~2s)* | *(API cost; estimate from usage)* | *(e.g. API key only)* |
| **Ollama (gemma)** | | | | *$0 (local)* | *(install Ollama + pull model)* |
| **Gemini (gemini-2.5-flash)** | | | | *(API cost)* | *(API key)* |

### How to fill

- **Accuracy**: Rate how correct and complete the answers are (e.g. 1–5, or Good/Fair/Poor) after reading `comparison_results.json`.
- **Hallucination**: Note if the model invented facts not in the documents: None / Minor / Major.
- **Response Speed**: Copy from the script’s summary table or from `comparison_results.json` (`time_seconds`).
- **Cost per Query**: OpenAI/Gemini: estimate from token usage if available; Ollama: $0 (local).
- **Ease of Setup**: Short note (e.g. “API key”, “Ollama install + pull”, “API key”).

---

## Test questions used

1. **Leave policies** – What are the leave policies? How many days of annual leave do employees get?
2. **Remote work rules** – What are the remote work rules? Is remote work allowed and under what conditions?
3. **Startup pricing model** – What is the startup's pricing model? How does the company price its product?
4. *(optional)* Sick leave – What is the policy on sick leave and how should employees report it?
5. *(optional)* Company values – What are the company's core values or mission mentioned in the documents?

---

## Per-question notes (optional)

Add short notes after reviewing `comparison_results.json`:

| Question | OpenAI | Ollama | Gemini |
|----------|--------|--------|--------|
| Leave policies | | | |
| Remote work rules | | | |
| Startup pricing model | | | |
