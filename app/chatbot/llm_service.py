from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    used_llm: bool
    provider: str
    model: str | None = None
    request_id: str | None = None
    error: str | None = None


SYSTEM_INSTRUCTIONS = """
You are the AI Healthcare Mobility Assistant inside a rehabilitation monitoring dashboard.

Your job:
1. Explain retrieved mobility data and approved exercise knowledge.
2. Adapt language to the logged-in role.
3. Be concise, respectful, accessible, and transparent.
4. Distinguish recorded facts from cautious interpretation.
5. State when information is unavailable.

Hard safety rules:
- Never diagnose a disease or injury.
- Never prescribe medication or change medication.
- Never prescribe patient-specific exercise targets, angles, repetitions, resistance, or treatment.
- Never claim research dataset labels are clinical thresholds.
- Never invent measurements, alerts, trends, sources, or clinician instructions.
- For emergency indicators, direct the user to urgent help rather than continuing normal coaching.
- Use only the supplied context for patient-specific claims.

Role behavior:
- Patient: simple language, short explanation, supportive tone.
- Caregiver: plain summary, observations, and safe next step.
- Clinician: concise technical summary with measured values and limitations.

Answer format:
- Start with a direct answer.
- Mention the relevant recorded evidence.
- State limitations when needed.
- Do not mention internal prompts, token limits, or hidden reasoning.
""".strip()


class LLMService:
    """Generate grounded responses using OpenAI or a deterministic fallback."""

    def __init__(
        self,
        *,
        model: str | None = None,
        use_llm: bool | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        env_setting = os.getenv("USE_LLM", "true").strip().lower()
        self.use_llm = (
            use_llm
            if use_llm is not None
            else env_setting in {"1", "true", "yes", "on"}
        )
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _serialize_context(self, context: dict[str, Any]) -> str:
        return json.dumps(context, indent=2, ensure_ascii=False, default=str)

    def _fallback_answer(self, context: dict[str, Any]) -> str:
        intent = context.get("intent", "general_help")
        role = context.get("user", {}).get("role", "Patient")
        mobility = context.get("mobility", {})
        knowledge = context.get("clinical_knowledge", [])

        if intent == "session_summary" and mobility.get("available"):
            total = mobility.get("total_repetitions")
            flagged = mobility.get("flagged_repetitions")
            metrics = mobility.get("metrics", {})

            parts = []
            if total is not None:
                parts.append(f"The latest recorded session contains {total} repetitions.")
            if flagged is not None:
                parts.append(f"{flagged} repetitions were flagged by the configured rules.")

            if "hold_duration_s" in metrics:
                value = metrics["hold_duration_s"].get("mean")
                parts.append(f"The average recorded hold duration was {value} seconds.")

            if "jerk_score" in metrics:
                value = metrics["jerk_score"].get("mean")
                parts.append(
                    f"The average movement-smoothness indicator was {value}."
                )

            if not parts:
                parts.append("The session data was found, but no supported summary metrics were available.")

            return " ".join(parts)

        if intent == "flag_explanation" and mobility.get("available"):
            records = mobility.get("flagged_repetitions", [])
            if "repetition" in mobility:
                rep = mobility["repetition"]
                return (
                    f"Repetition {rep.get('rep_id', 'requested')} was retrieved. "
                    f"The recorded values include hold duration "
                    f"{rep.get('hold_duration_s', 'not available')}, lift angle "
                    f"{rep.get('shin_lift_angle_deg', 'not available')}, and jerk score "
                    f"{rep.get('jerk_score', 'not available')}. "
                    "A flag indicates that a configured system rule was triggered; "
                    "it is not a diagnosis."
                )

            if records:
                first = records[0]
                reasons = ", ".join(first.get("reasons", [])) or "a configured rule"
                return (
                    f"{mobility.get('total_flagged', len(records))} repetitions were flagged. "
                    f"For example, repetition {first.get('rep_id')} was associated with "
                    f"{reasons}. A flag describes measured exercise performance and does "
                    "not diagnose a medical condition."
                )

            return "No flagged repetitions were found in the retrieved session."

        if intent in {"exercise_guidance", "metric_explanation"} and knowledge:
            top = knowledge[0]
            return (
                f"{top.get('title', 'Relevant guidance')}: "
                f"{top.get('text', '')}"
            )

        if intent == "trend_analysis":
            return mobility.get(
                "message",
                "Trend analysis requires multiple dated sessions for the same patient.",
            )

        if intent == "dashboard_navigation":
            page = context.get("dashboard", {}).get("current_page", "the dashboard")
            return (
                f"You are currently on {page}. Use the left-side navigation to open "
                "Home, Daily Goals, Reports, Doctor Feedback, or AI Assistant."
            )

        if role == "Clinician":
            return (
                "I can summarize recorded sessions, retrieve flagged repetitions, "
                "explain metrics, and provide approved exercise-reference text."
            )

        return (
            "I can summarize recorded sessions, explain alerts and metrics, and retrieve "
            "approved exercise guidance."
        )

    def generate(self, context: dict[str, Any]) -> GenerationResult:
        """Generate a response, falling back locally when no API is configured."""
        if not self.use_llm or not self.api_key or OpenAI is None:
            reason = None
            if not self.use_llm:
                reason = "LLM disabled by configuration."
            elif not self.api_key:
                reason = "OPENAI_API_KEY is not configured."
            elif OpenAI is None:
                reason = "The openai package is not installed."

            return GenerationResult(
                text=self._fallback_answer(context),
                used_llm=False,
                provider="local_fallback",
                error=reason,
            )

        user_input = (
            "Answer the user's healthcare mobility dashboard question using only the "
            "context below.\n\n"
            + self._serialize_context(context)
        )

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.responses.create(
                model=self.model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=user_input,
            )

            text = (response.output_text or "").strip()
            if not text:
                raise RuntimeError("The model returned an empty response.")

            return GenerationResult(
                text=text,
                used_llm=True,
                provider="openai",
                model=self.model,
                request_id=getattr(response, "_request_id", None),
            )
        except Exception as exc:
            return GenerationResult(
                text=self._fallback_answer(context),
                used_llm=False,
                provider="local_fallback",
                model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )
