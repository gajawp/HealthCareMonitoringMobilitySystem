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
        "summarize my latest session",
        "summarize latest session",
        "summarize my mobility session",
        "summarize mobility session",
        "summarize my latest mobility session",
        "summarize latest mobility session",
        "latest session",
        "latest mobility session",
        "my latest session",
        "my latest mobility session",
        "recent session",
        "recent mobility session",
        "recorded session",
        "recorded mobility session",
        "how did i do",
        "session summary",
        "mobility session summary",
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


def _detect_session_summary_by_keywords(
    normalized: str,
) -> IntentResult | None:
    """
    Detect session-summary requests even when extra words appear between
    phrases, such as 'summarize my latest mobility session'.
    """

    session_terms = (
        "session",
        "mobility session",
        "recorded session",
    )

    summary_terms = (
        "summarize",
        "summary",
        "latest",
        "recent",
        "today",
        "how did i do",
        "performance",
        "result",
    )

    matched_session_terms = tuple(
        term
        for term in session_terms
        if term in normalized
    )

    matched_summary_terms = tuple(
        term
        for term in summary_terms
        if term in normalized
    )

    if matched_session_terms and matched_summary_terms:
        matched = (
            matched_session_terms
            + matched_summary_terms
        )

        return IntentResult(
            name="session_summary",
            confidence=0.90,
            matched_phrases=matched,
        )

    return None


def detect_intent(question: str) -> IntentResult:
    """
    Detect the most likely chatbot intent.

    The implementation uses deterministic phrase matching with an additional
    keyword-combination rule for session summaries.
    """

    normalized = normalize_text(question)

    if not normalized:
        return IntentResult(
            name="empty",
            confidence=1.0,
        )

    session_keyword_result = (
        _detect_session_summary_by_keywords(
            normalized
        )
    )

    if session_keyword_result is not None:
        return session_keyword_result

    best_intent = "general_help"
    best_matches: tuple[str, ...] = ()
    best_score = 0.0

    for intent, phrases in INTENT_PATTERNS.items():
        matches = tuple(
            phrase
            for phrase in phrases
            if phrase in normalized
        )

        if not matches:
            continue

        phrase_coverage = sum(
            len(match.split())
            for match in matches
        )

        score = min(
            0.55
            + (0.08 * phrase_coverage)
            + (0.04 * len(matches)),
            0.99,
        )

        if score > best_score:
            best_intent = intent
            best_matches = matches
            best_score = score

    if best_intent == "general_help":
        return IntentResult(
            name="general_help",
            confidence=0.35,
        )

    return IntentResult(
        name=best_intent,
        confidence=round(best_score, 2),
        matched_phrases=best_matches,
    )