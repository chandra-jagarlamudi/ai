"""
Tests for the Meeting Notes Analyzer — 4 scenario coverage
===========================================================

Test 1: Meeting with clear task assignments (example from the brief)
Test 2: Discussion-only meeting with no tasks
Test 3: Meeting with multiple action items and deadlines
Test 4: Audio meeting recording (requires an .mp3/.wav/etc. file)
"""

import logging
import os
import sys

from meeting_analyzer import analyze_audio_meeting, analyze_meeting

logger = logging.getLogger("test_meeting_analyzer")

# ---------------------------------------------------------------------------
# Test transcripts
# ---------------------------------------------------------------------------

TRANSCRIPT_1_CLEAR_TASKS = """
John: We need to improve the website performance.
Sarah: Yes, page load time is too slow.
David: I will optimize the database queries this week.
Sarah: I will redesign the homepage layout.
John: Let's try to finish these tasks before Friday.
"""

TRANSCRIPT_2_NO_TASKS = """
Alice: Good morning everyone. Today I wanted to catch up on where the product
       roadmap stands.
Bob: Sure, I think we've made good progress on the authentication module.
     The new OAuth2 flow is working well in staging.
Alice: That's great. The UX team also shared their research on user drop-off —
       the main pain point seems to be the onboarding wizard.
Bob: Interesting. We should keep that in mind when planning the next quarter.
Carol: Agreed. The data also shows that mobile users have a 20% higher churn rate
       than desktop users.
Alice: Good discussion everyone. Let's pick this up in the next planning session.
"""

TRANSCRIPT_3_MULTIPLE_DEADLINES = """
Manager: Alright team, we have a critical product launch in two weeks.
         We need to move FAST.
Dev Lead: I'll finish the API integration by tomorrow EOD — it's blocking QA.
QA Lead: Once I get the API, I'll run the full regression suite. I need two days.
Designer: I have to deliver the final landing page assets ASAP — marketing is
          waiting on me.
Dev Lead: Also, James needs to deploy the new CDN configuration by Wednesday
          or we'll miss our performance targets.
Manager: Security audit is urgent too — Lisa, can you complete the OWASP checklist
         by end of this week?
Lisa: Yes, I'll prioritise that. I'll also coordinate with the DevOps team to
      patch the two open CVEs before launch.
Manager: Great. Remember, the launch is non-negotiable. Everything needs to be
         done before next Friday.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_test(name: str, transcript: str) -> None:
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  {name}")
    print(separator)
    report = analyze_meeting(transcript)
    print(report)


def run_audio_test(audio_path: str) -> None:
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  TEST 4 — Audio meeting recording")
    print(f"  File: {audio_path}")
    print(separator)

    if not os.path.exists(audio_path):
        print(
            f"[SKIPPED] Audio file not found: {audio_path}\n"
            "  To run this test, provide a real audio file path:\n"
            "    python test_meeting_analyzer.py path/to/meeting.mp3"
        )
        return

    report = analyze_audio_meeting(audio_path)
    print(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    run_test("TEST 1 — Meeting with clear task assignments", TRANSCRIPT_1_CLEAR_TASKS)
    run_test("TEST 2 — Discussion-only meeting (no tasks)", TRANSCRIPT_2_NO_TASKS)
    run_test("TEST 3 — Multiple action items with deadlines", TRANSCRIPT_3_MULTIPLE_DEADLINES)

    # Test 4 — audio: pass a file path as the first CLI argument, e.g.:
    #   python test_meeting_analyzer.py meeting.mp3
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "sample_meeting.mp3"
    run_audio_test(audio_path)
