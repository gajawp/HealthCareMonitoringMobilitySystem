from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_ROLES = {"Patient", "Caregiver", "Clinician"}


@dataclass
class UserContext:
    """Identity and authorization context supplied by the dashboard."""

    user_id: str
    user_name: str
    role: str
    patient_id: str
    authorized_patient_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"Unsupported role: {self.role}")

        if not self.user_id:
            raise ValueError("user_id is required.")

        if not self.patient_id:
            raise ValueError("patient_id is required.")

        allowed = set(self.authorized_patient_ids)

        if self.role == "Patient" and self.patient_id != self.user_id:
            raise PermissionError("A patient may access only their own records.")

        if self.role in {"Caregiver", "Clinician"} and allowed:
            if self.patient_id not in allowed:
                raise PermissionError(
                    f"{self.role} is not authorized for patient {self.patient_id}."
                )


@dataclass
class DashboardContext:
    """State from the currently displayed dashboard page."""

    current_page: str = "AI Assistant"
    selected_session_id: str | None = None
    selected_exercise_id: str | None = None
    selected_date_range: str | None = None
    visible_metrics: dict[str, Any] = field(default_factory=dict)
    active_alert: dict[str, Any] | None = None


@dataclass
class RetrievalContext:
    """Combined context passed to the LLM and fallback response generator."""

    question: str
    intent: str
    intent_confidence: float
    user: dict[str, Any]
    dashboard: dict[str, Any]
    mobility: dict[str, Any]
    clinical_knowledge: list[dict[str, Any]]
    sources: list[str]
    sufficient_evidence: bool


def build_user_context(
    *,
    user_id: str,
    user_name: str,
    role: str,
    patient_id: str,
    authorized_patient_ids: list[str] | None = None,
) -> UserContext:
    context = UserContext(
        user_id=user_id,
        user_name=user_name,
        role=role,
        patient_id=patient_id,
        authorized_patient_ids=authorized_patient_ids or [],
    )
    context.validate()
    return context


def build_dashboard_context(
    *,
    current_page: str = "AI Assistant",
    selected_session_id: str | None = None,
    selected_exercise_id: str | None = None,
    selected_date_range: str | None = None,
    visible_metrics: dict[str, Any] | None = None,
    active_alert: dict[str, Any] | None = None,
) -> DashboardContext:
    return DashboardContext(
        current_page=current_page,
        selected_session_id=selected_session_id,
        selected_exercise_id=selected_exercise_id,
        selected_date_range=selected_date_range,
        visible_metrics=visible_metrics or {},
        active_alert=active_alert,
    )


def assemble_retrieval_context(
    *,
    question: str,
    intent: str,
    intent_confidence: float,
    user_context: UserContext,
    dashboard_context: DashboardContext,
    mobility_data: dict[str, Any] | None,
    clinical_knowledge: list[dict[str, Any]] | None,
) -> RetrievalContext:
    mobility = mobility_data or {}
    knowledge = clinical_knowledge or []

    source_ids: list[str] = []
    for result in knowledge:
        for source in result.get("sources", []):
            if source not in source_ids:
                source_ids.append(source)

    sufficient_evidence = bool(
        mobility.get("available")
        or mobility.get("sessions")
        or mobility.get("flagged_repetitions")
        or knowledge
        or dashboard_context.visible_metrics
        or dashboard_context.active_alert
    )

    return RetrievalContext(
        question=question,
        intent=intent,
        intent_confidence=intent_confidence,
        user=asdict(user_context),
        dashboard=asdict(dashboard_context),
        mobility=mobility,
        clinical_knowledge=knowledge,
        sources=source_ids,
        sufficient_evidence=sufficient_evidence,
    )
