"""
Meeting Notes Analyzer — Streamlit UI
======================================
Upload a meeting document (TXT, PDF) or audio recording (MP3, MP4, M4A, WAV, WEBM)
and get a structured meeting report powered by the LangGraph multi-agent workflow.
"""

import logging
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Meeting Notes Analyzer",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_TYPES = {"mp3", "mp4", "m4a", "wav", "webm", "mpeg", "mpga"}
TEXT_TYPES = {"txt", "pdf"}
ALL_TYPES = sorted(TEXT_TYPES | AUDIO_TYPES)

PRIORITY_COLORS = {
    "high priority": "🔴",
    "medium priority": "🟡",
    "low priority": "🟢",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from a TXT or PDF upload."""
    ext = Path(uploaded_file.name).suffix.lower().lstrip(".")

    if ext == "txt":
        return uploaded_file.read().decode("utf-8", errors="replace")

    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(f"Unsupported text format: .{ext}")


def save_audio_tmp(uploaded_file) -> str:
    """Save an audio upload to a temp file and return the path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def render_results(state: dict) -> None:
    """Render the structured meeting report from the LangGraph state."""

    # ── Priority badge ─────────────────────────────────────────────────
    priority = state.get("priority", "")
    if priority:
        icon = PRIORITY_COLORS.get(priority.lower(), "⚪")
        st.markdown(
            f"<div style='text-align:right; font-size:1.1rem; margin-bottom:0.5rem'>"
            f"{icon} <strong>{priority}</strong></div>",
            unsafe_allow_html=True,
        )
        reasoning = state.get("priority_reasoning", "")
        if reasoning:
            st.caption(f"_{reasoning}_")

    st.divider()

    # ── Meeting Summary ────────────────────────────────────────────────
    st.subheader("📝 Meeting Summary")
    st.write(state.get("summary", "No summary available."))

    st.divider()

    col_topics, col_actions = st.columns(2)

    with col_topics:
        st.subheader("🏷️ Key Topics")
        topics = state.get("topics", [])
        if topics:
            for topic in topics:
                st.markdown(f"- {topic}")
        else:
            st.write("No topics identified.")

    with col_actions:
        st.subheader("✅ Action Items")
        action_items = state.get("action_items", [])
        if action_items:
            for item in action_items:
                task_priority = item.get("priority", "").lower()
                task_icon = PRIORITY_COLORS.get(f"{task_priority} priority", "⚪")
                with st.container(border=True):
                    col_task, col_badge = st.columns([5, 1])
                    with col_task:
                        st.markdown(f"**{item['task']}**")
                        st.caption(f"Owner: {item['owner']}")
                    with col_badge:
                        st.markdown(
                            f"<div style='text-align:right; padding-top:4px'>"
                            f"{task_icon} {item.get('priority', '–')}</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.info("No action items identified in this meeting.")

    st.divider()

    # ── Download + transcript ──────────────────────────────────────────
    report = state.get("final_report", "")
    if report:
        st.download_button(
            label="⬇️ Download Full Report (.txt)",
            data=report,
            file_name="meeting_report.txt",
            mime="text/plain",
        )

    transcript = state.get("transcript", "")
    if transcript:
        with st.expander("📄 View Transcript", expanded=False):
            st.text(transcript)


# ---------------------------------------------------------------------------
# Sidebar — instructions
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📋 Meeting Notes Analyzer")
    st.markdown("---")

    st.markdown("### How it works")
    st.markdown(
        """
1. Upload a meeting document or audio recording
2. Click **Analyze Meeting**
3. The LangGraph pipeline runs 4 AI agents:
   - 🏷️ **Topic Extraction**
   - 📝 **Meeting Summary**
   - ✅ **Action Items**
   - 🚦 **Priority Classification**
4. Download the structured report
"""
    )

    st.markdown("---")
    st.markdown("### Supported file types")
    st.markdown(
        """
| Type | Formats |
|------|---------|
| 📄 Text | TXT, PDF |
| 🎙️ Audio | MP3, MP4, M4A, WAV, WEBM |
"""
    )
    st.caption("Audio transcribed via **OpenAI Whisper**. Max 25 MB.")

    st.markdown("---")
    st.markdown("### Agent pipeline")
    st.markdown(
        """
```
START
 ├── audio? → Whisper
 └── text  ──────────►
         ↓
   extract_topics
         ↓
  summarize_meeting
         ↓
 extract_action_items
         ↓
  tasks? → classify_priority
  none  ──────────────────►
         ↓
   generate_report
         ↓
        END
```
"""
    )

# ---------------------------------------------------------------------------
# Main area — results first, then uploader at the bottom
# ---------------------------------------------------------------------------

st.header("Meeting Notes Analyzer")

# Show results from a previous run before the uploader so the user never
# has to scroll up after getting their report.
if "result_state" in st.session_state:
    col_title, col_clear = st.columns([6, 1])
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            del st.session_state["result_state"]
            st.rerun()
    render_results(st.session_state["result_state"])
    st.divider()

# ── File uploader & analyze button (always at the bottom) ─────────────────
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload a new meeting file",
        type=ALL_TYPES,
        help="Text/document: TXT, PDF  |  Audio: MP3, MP4, M4A, WAV, WEBM (max 25 MB)",
    )

    if uploaded_file:
        ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
        is_audio = ext in AUDIO_TYPES
        file_icon = "🎙️" if is_audio else "📄"
        st.success(f"{file_icon} Loaded **{uploaded_file.name}**")

        if st.button("Analyze Meeting", type="primary", use_container_width=True):

            from meeting_analyzer import analyze_audio_meeting_state, analyze_meeting_state

            try:
                if is_audio:
                    with st.spinner("🎙️ Transcribing audio with Whisper…"):
                        tmp_path = save_audio_tmp(uploaded_file)

                    with st.spinner("🤖 Analyzing with LangGraph agents…"):
                        state = analyze_audio_meeting_state(tmp_path)

                    Path(tmp_path).unlink(missing_ok=True)

                else:
                    with st.spinner("📄 Reading document…"):
                        transcript = extract_text_from_file(uploaded_file)

                    if not transcript.strip():
                        st.error("The file appears to be empty or could not be read.")
                        st.stop()

                    with st.spinner("🤖 Analyzing with LangGraph agents…"):
                        state = analyze_meeting_state(transcript)

                # Persist results in session state and rerun so they render above
                st.session_state["result_state"] = state
                st.rerun()

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                logging.exception("Analysis error")

    else:
        st.markdown(
            "<div style='color: grey; text-align: center; padding: 0.5rem'>"
            "TXT · PDF · MP3 · MP4 · M4A · WAV · WEBM</div>",
            unsafe_allow_html=True,
        )
