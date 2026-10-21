from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class IntentResult:
    """Result returned by the intent detector."""

    name: str
    confidence: float
    matched_phrases: tuple[str, ...] = ()


INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "privacy_sensitive_request": (
        "show me the exact prompt",
        "show the exact prompt",
        "prompt sent to the language model",
        "show the prompt sent to the language model",
        "reveal the system prompt",
        "show the system prompt",
        "print the system prompt",
        "show hidden instructions",
        "reveal hidden instructions",
        "show internal instructions",
        "reveal internal instructions",
        "show model context",
        "show the full context",
        "show raw context",
        "reveal the context",
        "developer message",
        "show developer message",
        "ignore privacy rules",
        "ignore previous instructions",
        "tell me the full name and patient id",
        "full name and patient id",
        "show patient id",
        "reveal patient id",
        "what is my patient id",
        "show patient identifier",
        "reveal patient identifier",
        "show patient token",
        "reveal patient token",
        "patient token secret",
        "show api key",
        "reveal api key",
        "show environment variables",
        "reveal environment variables",
        "show password hash",
        "reveal password hash",
        "show authorization mapping",
        "reveal authorization mapping",
        "show internal file paths",
        "reveal internal file paths",
        "show credentials",
        "reveal credentials",
        "list all patients",
        "show all patients",
        "identify the patient",
        "infer the patient identity",
    ),
    "exercise_recommendation": (
        "suggest exercise",
        "suggest exercises",
        "suggested exercise",
        "suggested exercises",
        "recommend exercise",
        "recommend exercises",
        "recommended exercise",
        "recommended exercises",
        "what exercise should i do",
        "what exercises should i do",
        "which exercise should i do",
        "which exercises should i do",
        "show my exercises",
        "show suggested exercises",
        "show recommended exercises",
        "exercise recommendation",
        "exercise recommendations",
        "assigned exercises",
        "my exercise plan",
    ),
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
        "repetition performance",
        "rep performance",
        "repetition summary",
        "rep summary",
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
    "have i improved",
    "has my balance improved",
    "has my mobility improved",
    "has my movement improved",
    "did i improve",
    "improved",
    "improving",
    "improvement",
    "progress",
    "trend",
    "compare sessions",
    "compare my sessions",
    "better than",
    "worse than",
    "over time",
    "getting better",
    "getting worse",
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
        "jerk",
        "jerk value",
        "movement jerk",
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



def _detect_recorded_mobility_metric_request(
    normalized: str,
) -> IntentResult | None:
    """
    Detect requests for values from the patient's recorded mobility session.

    This is intentionally separate from ``metric_explanation``:

    - "What is jerk score?" -> metric_explanation
    - "What was my average jerk score?" -> session_summary
    - "How many reps did I do?" -> session_summary
    - "How many frames were recorded?" -> session_summary

    The orchestrator can then retrieve the latest/session-specific mobility
    record before generation.
    """

    recorded_metric_terms = (
        "repetition",
        "repetitions",
        "rep",
        "reps",
        "rep count",
        "frames",
        "frame count",
        "jerk score",
        "hold duration",
        "lift duration",
        "lower duration",
        "duration",
        "shin angle",
        "shin lift angle",
        "knee angle",
        "symmetry",
        "toe-off symmetry",
        "toe off symmetry",
        "angular velocity",
        "peak angular velocity",
        "trunk lean",
        "tremor severity",
        "grip force",
        "tap rate",
    )

    value_request_terms = (
        "how many",
        "how much",
        "what was",
        "what were",
        "what is my",
        "what are my",
        "tell me",
        "show me",
        "give me",
        "my average",
        "my mean",
        "my latest",
        "my total",
        "total",
        "average",
        "mean",
        "latest",
        "recorded",
        "completed",
        "did i",
    )

    side_terms = (
        "left",
        "right",
        "flagged",
        "unflagged",
    )

    matched_metrics = tuple(
        term
        for term in recorded_metric_terms
        if term in normalized
    )

    if not matched_metrics:
        return None

    matched_value_terms = tuple(
        term
        for term in value_request_terms
        if term in normalized
    )

    # Side-specific repetition questions such as
    # "How many left leg repetitions did I complete?"
    matched_side_terms = tuple(
        term
        for term in side_terms
        if term in normalized
    )

    side_only_repetition_request = (
        matched_side_terms
        and any(
            phrase in normalized
            for phrase in (
                "how many",
                "did i complete",
                "did i do",
                "i complete",
                "i do",
                "completed with",
                "with the left leg",
                "with the right leg",
                "left leg",
                "right leg",
                "left side",
                "right side",
            )
        )
    )

    if matched_value_terms or side_only_repetition_request:
        return IntentResult(
            name="session_summary",
            confidence=0.96,
            matched_phrases=(
                matched_metrics
                + matched_value_terms
                + matched_side_terms
            ),
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

    privacy_phrases = INTENT_PATTERNS[
        "privacy_sensitive_request"
    ]
    privacy_matches = tuple(
        phrase
        for phrase in privacy_phrases
        if phrase in normalized
    )
    if privacy_matches:
        return IntentResult(
            name="privacy_sensitive_request",
            confidence=0.99,
            matched_phrases=privacy_matches,
        )

    exercise_recommendation_phrases = INTENT_PATTERNS[
        "exercise_recommendation"
    ]
    exercise_matches = tuple(
        phrase
        for phrase in exercise_recommendation_phrases
        if phrase in normalized
    )
    if exercise_matches:
        return IntentResult(
            name="exercise_recommendation",
            confidence=0.98,
            matched_phrases=exercise_matches,
        )

    recorded_metric_result = (
        _detect_recorded_mobility_metric_request(
            normalized
        )
    )

    if recorded_metric_result is not None:
        return recorded_metric_result

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


def _as_date(value: str | date | None) -> date | None:
    """Parse an ISO date or datetime string into a date."""

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    try:
        return datetime.fromisoformat(str(value).strip()).date()
    except ValueError:
        return None


def _format_date(value: str | date | None) -> str | None:
    parsed = _as_date(value)
    if parsed is None:
        return None
    return f"{parsed:%b} {parsed.day}, {parsed.year}"


def describe_interpretation(
    *,
    intent: str,
    session_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    selected_session_id: str | None = None,
    resolved_dates: list[str] | None = None,
) -> str | None:
    """
    Describe, in plain words, which sessions a question was resolved to.

    ``resolved_dates`` are the session dates the retriever actually used.
    The text is shown above the answer so users can spot a misread date or
    a misunderstood "latest" before relying on the answer.
    """

    dates = sorted(
        parsed
        for parsed in (_as_date(value) for value in resolved_dates or [])
        if parsed is not None
    )

    if session_date:
        return f"Sessions on {_format_date(session_date)}"

    if start_date and end_date:
        return (
            f"Sessions from {_format_date(start_date)} "
            f"to {_format_date(end_date)}"
        )

    if start_date:
        return f"Sessions from {_format_date(start_date)} onward"

    if end_date:
        return f"Sessions up to {_format_date(end_date)}"

    if intent == "trend_analysis":
        if len(dates) >= 2:
            return (
                f"Trend across {len(dates)} sessions "
                f"({_format_date(dates[0])} to {_format_date(dates[-1])})"
            )
        return None

    if intent in {
        "session_summary",
        "flag_explanation",
        "metric_explanation",
    } and dates:
        label = (
            "Selected session"
            if selected_session_id
            else "Latest session"
        )
        return f"{label} ({_format_date(dates[-1])})"

    return None


def closest_sessions(
    target: str | date | None,
    available_dates: list[str],
    *,
    limit: int = 3,
) -> list[str]:
    """
    Return the recorded session dates nearest to ``target``.

    Without a target, the most recent sessions are returned. Results are
    ISO ``YYYY-MM-DD`` strings in chronological order.
    """

    unique_dates = sorted(
        {
            parsed
            for parsed in (_as_date(value) for value in available_dates)
            if parsed is not None
        }
    )

    if not unique_dates or limit <= 0:
        return []

    target_date = _as_date(target)

    if target_date is None:
        nearest = unique_dates[-limit:]
    else:
        nearest = sorted(
            unique_dates,
            key=lambda value: (abs((value - target_date).days), value),
        )[:limit]

    return [value.isoformat() for value in sorted(nearest)]
