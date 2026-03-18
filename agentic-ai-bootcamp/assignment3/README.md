# Meeting Notes Analyzer

> An AI-powered LangGraph multi-agent workflow that converts any meeting transcript or audio recording into a structured meeting report — with a Streamlit UI and full LangSmith observability.

---

## Output

| Section | Description |
|---------|-------------|
| 📝 **Meeting Summary** | Concise 3–5 sentence overview |
| 🏷️ **Key Topics** | Labelled list of main discussion themes |
| ✅ **Action Items** | Every task with its assigned owner |
| 🚦 **Priority Level** | High / Medium / Low based on urgency signals |

---

## Workflow Architecture

The system is built as a **LangGraph state machine** with 6 nodes and 2 conditional routing points.

```
                    ┌─────────────────────────────────────────────────┐
                    │                    INPUT                        │
                    │        transcript text  /  audio file           │
                    └───────────────────┬─────────────────────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │           START             │
                         └──────────────┬──────────────┘
                                        │
                  ╔═════════════════════╪══════════════════════╗
                  ║  Conditional Route  │  Is audio file?      ║
                  ╚═════════════════════╪══════════════════════╝
                       YES ◄────────────┤────────────► NO
                        │               │              │
           ┌────────────▼───────────┐   │              │
           │   transcribe_audio     │   │              │
           │   ─────────────────    │   │              │
           │   OpenAI Whisper API   │   │              │
           │   audio → text         │   │              │
           └────────────┬───────────┘   │              │
                        │               │              │
                        └───────────────┘──────────────┘
                                        │
                           ┌────────────▼────────────-┐
                           │      extract_topics      │
                           │      ──────────────      │
                           │   Agent 1 · GPT-4o-mini  │
                           │   Identifies 3–7 key     │
                           │   discussion topics      │
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼────────────-┐
                           │    summarize_meeting     │
                           │    ─────────────────     │
                           │   Agent 2 · GPT-4o-mini  │
                           │   Generates a concise    │
                           │   3–5 sentence summary   │
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼───────────-─┐
                           │  extract_action_items    │
                           │  ────────────────────    │
                           │   Agent 3 · GPT-4o-mini  │
                           │   Finds tasks + owners   │
                           └────────────┬─────────────┘
                                        │
                  ╔═════════════════════╪══════════════════════╗
                  ║  Conditional Route  │  Action items found? ║
                  ╚═════════════════════╪══════════════════════╝
                       YES ◄────────────┤────────────► NO
                        │                              │
           ┌────────────▼───────────┐                  │
           │    classify_priority   │                  │
           │    ─────────────────   │                  │
           │   Agent 4 · GPT-4o-mini│                  │
           │   Rates urgency:       │                  │
           │   High / Medium / Low  │                  │
           └────────────┬───────────┘                  │
                        │                              │
                        └──────────────────────────────┘
                                        │
                           ┌────────────▼────────────-┐
                           │      generate_report     │
                           │      ───────────────     │
                           │   Assembles all agent    │
                           │   outputs into the       │
                           │   final structured report│
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼────────────┐
                           │           END           │
                           └─────────────────────────┘
```

### Conditional routing

| Decision point | Condition | Route |
|----------------|-----------|-------|
| After `START` | `audio_path` is set | `transcribe_audio` → pipeline |
| After `START` | Plain text provided | Skip straight to `extract_topics` |
| After `extract_action_items` | Tasks found | → `classify_priority` |
| After `extract_action_items` | No tasks found | Skip to `generate_report` |

---

## Project Structure

```
assignment3/
├── app.py                   # Streamlit web UI
├── meeting_analyzer.py      # LangGraph workflow + all agent nodes
├── test_meeting_analyzer.py # Test suite — 4 scenarios
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── .env                     # Your local secrets (never commit)
```

---

## Setup

```bash
# 1. Create and activate the virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

---

## Running the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### What you can upload

| Type | Extensions | Notes |
|------|------------|-------|
| Plain text | `.txt` | Raw transcript or notes |
| PDF document | `.pdf` | Meeting notes, exported docs |
| Audio recording | `.mp3` `.mp4` `.m4a` `.wav` `.webm` | Transcribed via Whisper, max 25 MB |

### UI features
- Upload a file and click **Analyze Meeting**
- Results render above the uploader — no scrolling needed
- **🗑️ Clear** button resets the results
- 🔴🟡🟢 Priority badge with one-line reasoning
- Topics and action items displayed side by side
- **⬇️ Download** the full report as a `.txt` file
- Collapsible transcript viewer (useful after audio transcription)

---

## Python API

```python
# Text transcript
from meeting_analyzer import analyze_meeting
print(analyze_meeting("John: Fix the login bug. Sarah: I'll handle it by tomorrow."))

# Audio recording
from meeting_analyzer import analyze_audio_meeting
print(analyze_audio_meeting("recordings/standup.mp3"))

# Get the full structured state (for custom rendering)
from meeting_analyzer import analyze_meeting_state
state = analyze_meeting_state("...")
# state keys: transcript, topics, summary, action_items, priority, final_report
```

---

## Running the Tests

```bash
# 3 text scenarios
python test_meeting_analyzer.py

# Include audio test (pass your file as an argument)
python test_meeting_analyzer.py path/to/meeting.mp3
```

| Test | Scenario | Expected routing |
|------|----------|-----------------|
| 1 | Clear task assignments | text → priority → **Medium** |
| 2 | Discussion only, no tasks | text → skip priority → **N/A** |
| 3 | Multiple tasks with deadlines | text → priority → **High** |
| 4 | Audio recording | audio → Whisper → pipeline |

---

## LangSmith Observability

Add these variables to `.env` to enable full tracing in [LangSmith](https://smith.langchain.com):

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-api-key-here
LANGSMITH_PROJECT=meeting-analyzer
```

Every run will appear in the LangSmith UI with:
- Each node as a labelled step in the trace
- LLM inputs / outputs and token usage per call
- Latency breakdown per agent
- The exact conditional routing path taken

---

## Configuration

```env
# .env
OPENAI_API_KEY=...          # Required
LLM_MODEL=gpt-4o-mini       # Default — fast and cost-effective
# LLM_MODEL=gpt-4o          # Higher capability
```
