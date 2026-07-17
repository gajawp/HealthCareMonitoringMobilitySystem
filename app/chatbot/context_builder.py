from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.security.privacy_filter import PrivacyFilter


VALID_ROLES = {
    "Patient",
    "Caregiver",
    "Clinician",
}


@dataclass
class UserContext:
    """
    Internal identity and authorization context.

    This object may contain identifiers for backend authorization and
    application logic. It must be sanitized before being sent to the LLM.
    """

    user_id: str
    user_name: str
    role: str
    patient_id: str
    authorized_patient_ids: list[str] = field(
        default_factory=list
    )

    def validate(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"Unsupported role: {self.role}"
            )

        if not self.user_id:
            raise ValueError("user_id is required.")

        if not self.patient_id:
            raise ValueError("patient_id is required.")

        authorized_ids = set(
            self.authorized_patient_ids
        )

        if self.role == "Patient":
            if self.patient_id != self.user_id:
                raise PermissionError(
                    "A patient may access only their own records."
                )

        if self.role in {
            "Caregiver",
            "Clinician",
        }:
            if (
                authorized_ids
                and self.patient_id
                not in authorized_ids
            ):
                raise PermissionError(
                    f"{self.role} is not authorized "
                    f"for patient {self.patient_id}."
                )


@dataclass
class DashboardContext:
    """State from the currently displayed dashboard."""

    current_page: str = "AI Assistant"
    selected_session_id: str | None = None
    selected_exercise_id: str | None = None
    selected_date_range: str | None = None
    visible_metrics: dict[str, Any] = field(
        default_factory=dict
    )
    active_alert: dict[str, Any] | None = None


@dataclass
class RetrievalContext:
    """
    Combined retrieval result used by the response generator.

    The `user` field may contain backend identifiers. The orchestrator must
    sanitize this dataclass before passing it to the LLM.
    """

    question: str
    intent: str
    intent_confidence: float
    user: dict[str, Any]
    dashboard: dict[str, Any]
    mobility: dict[str, Any]
    clinical_knowledge: list[dict[str, Any]]
    sources: list[str]
    sufficient_evidence: bool


def _normalize_role(role: str) -> str:
    normalized_role = role.strip().lower()

    role_mapping = {
        "patient": "Patient",
        "caregiver": "Caregiver",
        "care giver": "Caregiver",
        "clinician": "Clinician",
        "doctor": "Clinician",
        "physician": "Clinician",
    }

    return role_mapping.get(
        normalized_role,
        role,
    )


def build_user_context(
    *,
    user_id: str,
    user_name: str,
    role: str,
    patient_id: str,
    authorized_patient_ids: list[str] | None = None,
) -> UserContext:
    """
    Build and validate internal authorization context.

    This function is retained for backward compatibility with the original
    Phase 2 tests and the current Streamlit integration.
    """

    context = UserContext(
        user_id=user_id.strip().upper(),
        user_name=user_name.strip(),
        role=_normalize_role(role),
        patient_id=patient_id.strip().upper(),
        authorized_patient_ids=[
            patient.strip().upper()
            for patient in (
                authorized_patient_ids or []
            )
            if patient and patient.strip()
        ],
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
    """Build the current dashboard state."""

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
    clinical_knowledge: list[
        dict[str, Any]
    ] | None,
) -> RetrievalContext:
    """Combine user, dashboard, mobility and knowledge results."""

    mobility = mobility_data or {}
    knowledge = clinical_knowledge or []

    source_ids: list[str] = []

    for result in knowledge:
        for source in result.get(
            "sources",
            [],
        ):
            source_text = str(source)

            if source_text not in source_ids:
                source_ids.append(
                    source_text
                )

    latest_session = mobility.get("latest_session")

    latest_session_available = (
        isinstance(latest_session, dict)
        and bool(latest_session.get("available"))
    )

    sufficient_evidence = bool(
        mobility.get("available")
        or latest_session_available
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


class ChatbotContextBuilder:
    """
    Builds a context that is safe to send to the LLM.

    This class is used by new code. The standalone functions above are kept
    for compatibility with the existing Phase 2 implementation.
    """

    def __init__(
        self,
        privacy_filter: PrivacyFilter | None = None,
    ) -> None:
        self._privacy_filter = (
            privacy_filter
            or PrivacyFilter()
        )

    def build_context(
        self,
        *,
        role: str,
        question_intent: str,
        mobility_data: dict[str, Any],
        clinical_knowledge: list[
            dict[str, Any]
        ],
        dashboard_data: (
            dict[str, Any] | None
        ) = None,
    ) -> dict[str, Any]:
        raw_context = {
            "role": _normalize_role(role),
            "patient_reference": (
                "current_patient"
            ),
            "question_intent": (
                question_intent
            ),
            "dashboard": (
                dashboard_data or {}
            ),
            "mobility_data": (
                mobility_data
            ),
            "clinical_knowledge": (
                clinical_knowledge
            ),
        }

        safe_context = (
            self._privacy_filter
            .sanitize_context(
                raw_context
            )
        )

        self._privacy_filter.assert_safe_context(
            safe_context
        )

        return safe_context