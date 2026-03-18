"""
Meeting Notes Analyzer — LangGraph Multi-Agent Workflow
========================================================
Converts a meeting transcript (or audio file) into structured insights via a
sequential multi-agent pipeline with conditional routing.

Agents / Nodes
--------------
0. Audio Transcription Node  — (optional) Whisper → transcript text
1. Topic Extraction Agent    — identifies key discussion topics
2. Meeting Summary Agent     — generates a concise 3-5 sentence summary
3. Action Item Agent         — extracts tasks and task owners
4. Priority Classification Agent — rates urgency (High / Medium / Low)
5. Final Report Node         — assembles the structured meeting report

Conditional logic
-----------------
- If an audio_path is provided at START → transcribe_audio → pipeline.
  Otherwise the pipeline starts directly from extract_topics.
- If no action items are found → priority classification is skipped.

Observability
-------------
LangSmith tracing is enabled automatically when these env vars are set:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<your-key>
  LANGCHAIN_PROJECT=meeting-analyzer
Every node invocation appears as a step inside the LangSmith trace.
"""

import logging
import os
from typing import Any

import openai
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("meeting_analyzer")

# Log whether LangSmith tracing is active so the user knows at a glance.
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    logger.info("LangSmith tracing ENABLED — project: %s", project)
else:
    logger.info(
        "LangSmith tracing DISABLED "
        "(set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY to enable)"
    )

# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class TopicsOutput(BaseModel):
    topics: list[str] = Field(description="List of key discussion topics")


class SummaryOutput(BaseModel):
    summary: str = Field(description="Concise 3-5 sentence meeting summary")


class ActionItem(BaseModel):
    task: str = Field(description="Task description")
    owner: str = Field(description="Person responsible, or 'Not specified'")
    priority: str = Field(
        description="Task priority: 'High', 'Medium', or 'Low' based on urgency signals"
    )


class ActionItemsOutput(BaseModel):
    action_items: list[ActionItem] = Field(
        description="List of action items extracted from the transcript"
    )


class PriorityOutput(BaseModel):
    priority: str = Field(
        description="Priority level: 'High Priority', 'Medium Priority', or 'Low Priority'"
    )
    reasoning: str = Field(description="Brief explanation of the priority rating")


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class MeetingState(TypedDict):
    audio_path: str          # path to audio file; empty string = text-only input
    transcript: str          # raw meeting text (populated by transcription or caller)
    topics: list[str]
    summary: str
    action_items: list[dict[str, str]]
    priority: str
    priority_reasoning: str
    final_report: str


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0.5) -> ChatOpenAI:
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# Node 0 — Audio transcription (Whisper)
# ---------------------------------------------------------------------------


def transcribe_audio(state: MeetingState) -> dict[str, Any]:
    """
    Transcribe an audio meeting recording to text using OpenAI Whisper.
    Supports any format accepted by the Whisper API:
    mp3, mp4, mpeg, mpga, m4a, wav, webm (max 25 MB).
    """
    audio_path = state["audio_path"]
    logger.info("Audio node: Transcribing '%s' via Whisper …", audio_path)

    client = openai.OpenAI()
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    transcript = transcription if isinstance(transcription, str) else transcription.text
    logger.info("Transcription complete — %d chars", len(transcript))
    return {"transcript": transcript}


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


def extract_topics(state: MeetingState) -> dict[str, Any]:
    """Agent 1 — identify the main discussion topics."""
    logger.info("Agent 1: Extracting topics …")

    llm = _get_llm().with_structured_output(TopicsOutput)
    result: TopicsOutput = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert meeting analyst. "
                    "Extract the key discussion topics from the transcript. "
                    "Return 3–7 concise topic labels (2-5 words each)."
                )
            ),
            HumanMessage(content=f"Meeting transcript:\n\n{state['transcript']}"),
        ]
    )

    logger.info("Topics found: %s", result.topics)
    return {"topics": result.topics}


def summarize_meeting(state: MeetingState) -> dict[str, Any]:
    """Agent 2 — generate a concise meeting summary."""
    logger.info("Agent 2: Summarizing meeting …")

    llm = _get_llm().with_structured_output(SummaryOutput)
    result: SummaryOutput = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert meeting summarizer. "
                    "Write a clear, concise summary of the meeting in 3–5 sentences. "
                    "Focus on what was discussed and any decisions made."
                )
            ),
            HumanMessage(content=f"Meeting transcript:\n\n{state['transcript']}"),
        ]
    )

    logger.info("Summary generated (%d chars)", len(result.summary))
    return {"summary": result.summary}


def extract_action_items(state: MeetingState) -> dict[str, Any]:
    """Agent 3 — extract tasks and their owners."""
    logger.info("Agent 3: Extracting action items …")

    llm = _get_llm().with_structured_output(ActionItemsOutput)
    result: ActionItemsOutput = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert at identifying action items in meeting transcripts. "
                    "Extract every concrete task or commitment made. "
                    "For each item provide: "
                    "(1) the task description, "
                    "(2) the owner — person responsible, or 'Not specified' if not mentioned, "
                    "(3) the task priority — 'High', 'Medium', or 'Low' based on urgency signals "
                    "such as 'urgent', 'ASAP', 'by tomorrow', hard deadlines, or blocking work. "
                    "If there are no action items at all, return an empty list."
                )
            ),
            HumanMessage(content=f"Meeting transcript:\n\n{state['transcript']}"),
        ]
    )

    items = [item.model_dump() for item in result.action_items]
    logger.info("Action items found: %d", len(items))
    return {"action_items": items}


def classify_priority(state: MeetingState) -> dict[str, Any]:
    """Agent 4 — classify priority based on urgency signals."""
    logger.info("Agent 4: Classifying priority …")

    action_text = "\n".join(
        f"- {item['task']} (Owner: {item['owner']})"
        for item in state["action_items"]
    )

    llm = _get_llm().with_structured_output(PriorityOutput)
    result: PriorityOutput = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a priority assessment expert. "
                    "Analyse the meeting action items and transcript for urgency signals "
                    "such as: 'urgent', 'ASAP', 'by tomorrow', 'this week', hard deadlines, "
                    "customer-facing issues, revenue impact, or critical blockers. "
                    "Return exactly one of: 'High Priority', 'Medium Priority', or 'Low Priority' "
                    "and a brief one-sentence reasoning."
                )
            ),
            HumanMessage(
                content=(
                    f"Meeting transcript:\n\n{state['transcript']}\n\n"
                    f"Action items:\n{action_text}"
                )
            ),
        ]
    )

    logger.info("Priority: %s — %s", result.priority, result.reasoning)
    return {"priority": result.priority, "priority_reasoning": result.reasoning}


def generate_report(state: MeetingState) -> dict[str, Any]:
    """Final node — assemble the structured meeting report."""
    logger.info("Final node: Generating report …")

    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("MEETING NOTES — STRUCTURED REPORT")
    lines.append("=" * 60)

    # Show source so it's clear whether the input was audio or text
    if state.get("audio_path"):
        lines.append(f"Source: audio — {state['audio_path']}")
    lines.append("")

    lines.append("MEETING SUMMARY")
    lines.append("-" * 40)
    lines.append(state.get("summary", "No summary available."))
    lines.append("")

    lines.append("KEY TOPICS")
    lines.append("-" * 40)
    for i, topic in enumerate(state.get("topics", []), 1):
        lines.append(f"{i}. {topic}")
    lines.append("")

    lines.append("ACTION ITEMS")
    lines.append("-" * 40)
    action_items = state.get("action_items", [])
    if action_items:
        for i, item in enumerate(action_items, 1):
            lines.append(f"{i}. {item['task']}")
            lines.append(f"   Owner:    {item['owner']}")
            lines.append(f"   Priority: {item.get('priority', 'Not specified')}")
    else:
        lines.append("No action items identified in this meeting.")
    lines.append("")

    lines.append("PRIORITY LEVEL")
    lines.append("-" * 40)
    priority = state.get("priority", "")
    if priority:
        lines.append(priority)
        reasoning = state.get("priority_reasoning", "")
        if reasoning:
            lines.append(f"Reason: {reasoning}")
    else:
        lines.append("N/A — No action items to prioritise.")
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    logger.info("Report assembled (%d chars)", len(report))
    return {"final_report": report}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def route_input(state: MeetingState) -> str:
    """Route to audio transcription if an audio file path is provided."""
    if state.get("audio_path"):
        logger.info("Input is audio — routing → transcribe_audio")
        return "transcribe_audio"
    logger.info("Input is text — routing → extract_topics")
    return "extract_topics"


def route_after_action_items(state: MeetingState) -> str:
    """Skip priority classification when no action items were found."""
    if state.get("action_items"):
        logger.info("Routing → classify_priority")
        return "classify_priority"
    logger.info("No action items — routing → generate_report (skipping priority)")
    return "generate_report"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    graph = StateGraph(MeetingState)

    # Register nodes
    graph.add_node("transcribe_audio", transcribe_audio)
    graph.add_node("extract_topics", extract_topics)
    graph.add_node("summarize_meeting", summarize_meeting)
    graph.add_node("extract_action_items", extract_action_items)
    graph.add_node("classify_priority", classify_priority)
    graph.add_node("generate_report", generate_report)

    # Conditional entry: audio path provided → transcribe first
    graph.add_conditional_edges(
        START,
        route_input,
        {
            "transcribe_audio": "transcribe_audio",
            "extract_topics": "extract_topics",
        },
    )

    # After transcription, continue with the standard pipeline
    graph.add_edge("transcribe_audio", "extract_topics")

    # Sequential analysis pipeline
    graph.add_edge("extract_topics", "summarize_meeting")
    graph.add_edge("summarize_meeting", "extract_action_items")

    # Conditional: skip priority when no action items
    graph.add_conditional_edges(
        "extract_action_items",
        route_after_action_items,
        {
            "classify_priority": "classify_priority",
            "generate_report": "generate_report",
        },
    )

    graph.add_edge("classify_priority", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _run(audio_path: str = "", transcript: str = "") -> MeetingState:
    """Internal helper — invoke the compiled graph, returns full state."""
    logger.info("Starting meeting analysis …")
    app = build_graph()

    initial_state: MeetingState = {
        "audio_path": audio_path,
        "transcript": transcript,
        "topics": [],
        "summary": "",
        "action_items": [],
        "priority": "",
        "priority_reasoning": "",
        "final_report": "",
    }

    final_state = app.invoke(initial_state)
    logger.info("Analysis complete.")
    return final_state


def analyze_meeting(transcript: str) -> str:
    """Analyze a plain-text meeting transcript. Returns the formatted report."""
    return _run(transcript=transcript)["final_report"]


def analyze_meeting_state(transcript: str) -> MeetingState:
    """Analyze a plain-text meeting transcript. Returns the full state dict."""
    return _run(transcript=transcript)


def analyze_audio_meeting(audio_path: str) -> str:
    """Transcribe an audio recording then analyze it. Returns the formatted report."""
    return _run(audio_path=audio_path)["final_report"]


def analyze_audio_meeting_state(audio_path: str) -> MeetingState:
    """Transcribe an audio recording then analyze it. Returns the full state dict."""
    return _run(audio_path=audio_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """
    John: We need to improve the website performance.
    Sarah: Yes, page load time is too slow.
    David: I will optimize the database queries this week.
    Sarah: I will redesign the homepage layout.
    John: Let's try to finish these tasks before Friday.
    """
    print(analyze_meeting(sample))
