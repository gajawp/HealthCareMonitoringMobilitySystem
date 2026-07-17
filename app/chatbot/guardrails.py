from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAFETY_DIR = PROJECT_ROOT / "knowledge_base" / "safety"


@dataclass(frozen=True)
class SafetyDecision:
    action: str
    safe_response: str | None = None
    reason: str | None = None
    escalation_required: bool = False


DIAGNOSIS_PATTERNS = (
    r"\bdo i have\b",
    r"\bam i diagnosed\b",
    r"\bdiagnose me\b",
    r"\bdoes this mean i have\b",
    r"\bis this parkinson",
    r"\bis this arthritis",
    r"\bis this a stroke",
)

TREATMENT_PATTERNS = (
    r"\bwhat medication\b",
    r"\bchange my medication\b",
    r"\bstop taking\b",
    r"\bincrease my dose\b",
    r"\bdecrease my dose\b",
    r"\bhow many repetitions should i do\b",
    r"\bwhat angle is safe for me\b",
    r"\bshould i continue after surgery\b",
)

UNSAFE_OUTPUT_PATTERNS = (
    r"\byou have (parkinson'?s?|arthritis|a stroke|a disease|a disorder)\b",
    r"\byou are diagnosed with\b",
    r"\bthis confirms (a|the|that you have)\b",
    r"\bthe results confirm (a|the|that you have)\b",
    r"\bstop taking your\b",
    r"\bincrease your dose\b",
    r"\bdecrease your dose\b",
    r"\byou do not need (a|to see) (doctor|clinician)\b",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


class HealthcareGuardrails:
    """Apply input and output safety policies around chatbot generation."""

    def __init__(self) -> None:
        self.boundaries = _load_json(SAFETY_DIR / "chatbot_boundaries.json")
        self.escalation_rules = _load_json(SAFETY_DIR / "escalation_rules.json")

    @staticmethod
    def _matches(text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def check_input(self, question: str) -> SafetyDecision:
        normalized = question.lower().strip()

        for rule in self.escalation_rules.get("emergency_patterns", []):
            keywords = [str(keyword).lower() for keyword in rule.get("keywords", [])]
            if any(keyword in normalized for keyword in keywords):
                return SafetyDecision(
                    action="block_and_escalate",
                    safe_response=rule.get(
                        "response",
                        "This may require immediate medical attention. "
                        "Contact local emergency services now.",
                    ),
                    reason=rule.get("id", "possible_emergency"),
                    escalation_required=True,
                )

        for rule in self.escalation_rules.get(
            "non_emergency_escalation_patterns", []
        ):
            keywords = [str(keyword).lower() for keyword in rule.get("keywords", [])]
            if any(keyword in normalized for keyword in keywords):
                return SafetyDecision(
                    action="allow_with_escalation",
                    safe_response=rule.get("response"),
                    reason=rule.get("id", "care_team_review"),
                    escalation_required=True,
                )

        if self._matches(normalized, DIAGNOSIS_PATTERNS):
            response = self.boundaries.get("required_response_behavior", {}).get(
                "diagnosis_request",
                "I can explain recorded mobility data, but I cannot diagnose a medical condition.",
            )
            return SafetyDecision(
                action="block",
                safe_response=response,
                reason="diagnosis_request",
            )

        if self._matches(normalized, TREATMENT_PATTERNS):
            response = self.boundaries.get("required_response_behavior", {}).get(
                "treatment_request",
                "Exercise targets and treatment decisions must come from the patient's clinician or care plan.",
            )
            return SafetyDecision(
                action="block",
                safe_response=response,
                reason="treatment_request",
            )

        return SafetyDecision(action="allow")

    def check_output(
        self,
        answer: str,
        *,
        sufficient_evidence: bool,
    ) -> SafetyDecision:
        if not answer.strip():
            return SafetyDecision(
                action="replace",
                safe_response="I could not generate a response.",
                reason="empty_output",
            )

        if self._matches(answer, UNSAFE_OUTPUT_PATTERNS):
            return SafetyDecision(
                action="replace",
                safe_response=(
                    "I can explain the recorded mobility information, but I cannot "
                    "diagnose a condition or recommend treatment changes. Please "
                    "discuss medical decisions with the care team."
                ),
                reason="unsafe_medical_claim",
                escalation_required=False,
            )

        if not sufficient_evidence:
            return SafetyDecision(
                action="replace",
                safe_response=self.boundaries.get(
                    "required_response_behavior", {}
                ).get(
                    "missing_data",
                    "I do not have enough recorded information to answer that reliably.",
                ),
                reason="insufficient_evidence",
            )

        return SafetyDecision(action="allow")

    def add_standard_disclaimer(self, answer: str, *, role: str) -> str:
        if role == "Clinician":
            return answer

        disclaimer = (
            "\n\n*This assistant explains recorded dashboard information and approved "
            "exercise material. It does not provide a diagnosis or change a care plan.*"
        )

        if disclaimer.strip() in answer:
            return answer

        return answer.rstrip() + disclaimer
