from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentResult:
    """Result returned by the intent detector."""

    name: str
    confidence: float
    matched_phrases: tuple[str, ...] = ()


INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "session_summary": (
        "summarize my session",
        "summarize session",
        "latest session",
        "how did i do",
        "session summary",
        "today's session",
        "today session",
        "session result",
        "session performance",
    ),
    "flag_explanation": (
        "why was",
        "flagged",
        "flag",
        "alert",
        "warning",
        "below target",
        "needs attention",
        "what went wrong",
    ),
    "trend_analysis": (
        "am i improving",
        "improving",
        "improvement",
        "progress",
        "trend",
        "compare sessions",
        "better than",
        "worse than",
        "over time",
    ),
    "exercise_guidance": (
        "how should i",
        "how do i",
        "correct posture",
        "correct form",
        "perform the exercise",
        "exercise guidance",
        "exercise instructions",
        "knee extension",
        "sit to stand",
        "deep squat",
        "hurdle step",
        "side lunge",
    ),
    "metric_explanation": (
        "what is",
        "what does",
        "meaning of",
        "explain",
        "jerk score",
        "hold duration",
        "lift duration",
        "shin angle",
        "knee angle",
        "symmetry",
        "angular velocity",
        "trunk lean",
        "tremor severity",
        "grip force",
        "tap rate",
    ),
    "dashboard_navigation": (
        "where is",
        "where can i find",
        "which page",
        "navigate",
        "show me the page",
        "open reports",
        "find my alerts",
    ),
    "report_generation": (
        "generate report",
        "create report",
        "make a report",
        "clinical summary",
        "clinician note",
        "doctor note",
        "download report",
    ),
    "care_team_escalation": (
        "contact clinician",
        "contact doctor",
        "tell my doctor",
        "notify caregiver",
        "send to clinician",
        "escalate",
    ),
}


def normalize_text(text: str) -> str:
    """Normalize text for deterministic keyword-based intent detection."""
    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s'-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized)


def detect_intent(question: str) -> IntentResult:
    """Detect the most likely chatbot intent.

    The initial implementation is deterministic and testable. It can later be
    replaced by an ML classifier without changing the orchestrator interface.
    """
    normalized = normalize_text(question)

    if not normalized:
        return IntentResult("empty", 1.0)

    best_intent = "general_help"
    best_matches: tuple[str, ...] = ()
    best_score = 0.0

    for intent, phrases in INTENT_PATTERNS.items():
        matches = tuple(phrase for phrase in phrases if phrase in normalized)
        if not matches:
            continue

        phrase_coverage = sum(len(match.split()) for match in matches)
        score = min(0.55 + (0.08 * phrase_coverage) + (0.04 * len(matches)), 0.99)

        if score > best_score:
            best_intent = intent
            best_matches = matches
            best_score = score

    if best_intent == "general_help":
        return IntentResult("general_help", 0.35)

    return IntentResult(best_intent, round(best_score, 2), best_matches)
