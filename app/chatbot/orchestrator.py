"""
Secure healthcare chatbot orchestrator.

This module coordinates:

1. Input guardrails.
2. User and patient authorization.
3. Patient-specific mobility retrieval.
4. Approved clinical knowledge retrieval.
5. Removal of identifiers before LLM generation.
6. LLM response privacy validation.
7. Safe response formatting.

Important privacy rule:
The patient ID and opaque patient key may be used by trusted backend services,
but neither value is included in the LLM generation context.
"""

from __future__ import annotations

import inspect
import re
from datetime import date, timedelta

from dateutil import parser as date_parser
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from app.auth.models import AuthenticatedUser, UserRole
from app.security.audit import AuditLogger
from app.security.authorization import AuthorizationService
from app.security.patient_identity import PatientIdentityService
from app.security.privacy_filter import PrivacyFilter

from .context_builder import (
    assemble_retrieval_context,
    build_dashboard_context,
    build_user_context,
)
from .guardrails import HealthcareGuardrails
from .exercise_recommender import ExerciseRecommendationService
from .intent import detect_intent
from .knowledge_retriever import ClinicalKnowledgeRetriever
from .llm_service import LLMService
from .mobility_retriever import MobilityRetriever


@dataclass(frozen=True)
class ChatbotResponse:
    """Standard response returned by the chatbot."""

    answer: str
    intent: str
    intent_confidence: float
    sources: list[str]
    used_llm: bool
    provider: str
    model: str | None
    safety_action: str
    escalation_required: bool
    metadata: dict[str, Any]
    response_type: str = "text"
    exercise_data: dict[str, Any] | None = None


PATIENT_REFERENCE_PATTERN = re.compile(
    r"\bP\d{3,10}\b",
    flags=re.IGNORECASE,
)


def _extract_patient_references(question: str) -> list[str]:
    """
    Extract explicit patient identifiers from a question.

    The identifiers are used only for authorization validation and are never
    added to LLM context, chatbot responses, or audit details.
    """

    matches = PATIENT_REFERENCE_PATTERN.findall(question or "")

    return list(
        dict.fromkeys(
            match.strip().upper()
            for match in matches
            if match.strip()
        )
    )


def _validate_question_patient_references(
    *,
    question: str,
    selected_patient_id: str,
    authorized_patient_ids: list[str] | None,
) -> tuple[bool, str | None, int]:
    """
    Validate explicit patient references in the question.

    A question may refer to the currently selected patient. References to an
    unassigned patient are denied. References to an authorized but non-selected
    patient are also denied so that the user must explicitly change the patient
    through the dashboard before requesting patient-specific data.
    """

    referenced_ids = _extract_patient_references(question)

    if not referenced_ids:
        return True, None, 0

    selected_id = selected_patient_id.strip().upper()

    authorized_ids = {
        value.strip().upper()
        for value in (authorized_patient_ids or [])
        if isinstance(value, str) and value.strip()
    }
    authorized_ids.add(selected_id)

    unauthorized_count = sum(
        reference not in authorized_ids
        for reference in referenced_ids
    )

    if unauthorized_count:
        return (
            False,
            "unauthorized_patient_reference",
            len(referenced_ids),
        )

    if len(referenced_ids) > 1:
        return (
            False,
            "multiple_patient_references_not_supported",
            len(referenced_ids),
        )

    if referenced_ids[0] != selected_id:
        return (
            False,
            "referenced_patient_not_selected",
            len(referenced_ids),
        )

    return True, None, len(referenced_ids)


def _extract_rep_id(question: str) -> int | None:
    """Extract a repetition number from the user question."""

    patterns = (
        r"\brep(?:etition)?\s*#?\s*(\d+)\b",
        r"\bnumber\s+(\d+)\b",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


def _parse_date_text(
    text: str,
    *,
    default_year: int | None = None,
) -> str | None:
    """Parse a user-provided date and normalize it to YYYY-MM-DD."""

    cleaned = re.sub(
        r"\b(?:on|from|to|through|until|between|and)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    ).strip(" ,.")

    if not cleaned:
        return None

    default_value = date(
        default_year or date.today().year,
        1,
        1,
    )

    try:
        parsed = date_parser.parse(
            cleaned,
            fuzzy=True,
            default=default_value,
        )
    except (ValueError, TypeError, OverflowError):
        return None

    return parsed.strftime("%Y-%m-%d")


def _extract_date_filters(
    question: str,
) -> tuple[str | None, str | None, str | None]:
    """
    Extract one date or an inclusive date range from a question.

    Returns: (session_date, start_date, end_date)
    """

    text = question.strip()
    relative_days_match = re.search(
        r"\b(?:last|past)\s+(\d+)\s+days?\b",
        text,
        flags=re.IGNORECASE,
    )

    if relative_days_match:
        number_of_days = int(relative_days_match.group(1))

        if number_of_days <= 0:
            return None, None, None

        end_value = date.today()
        start_value = end_value - timedelta(
            days=number_of_days - 1
        )

        return (
            None,
            start_value.strftime("%Y-%m-%d"),
            end_value.strftime("%Y-%m-%d"),
        )

    range_patterns = (
        r"\bfrom\s+(.+?)\s+(?:to|through|until)\s+(.+?)(?:[?.!]|$)",
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
    )

    for pattern in range_patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        left_text = match.group(1).strip()
        right_text = match.group(2).strip()

        right_date = _parse_date_text(right_text)
        inferred_year = (
            int(right_date[:4])
            if right_date is not None
            else None
        )
        left_date = _parse_date_text(
            left_text,
            default_year=inferred_year,
        )

        if left_date and right_date:
            if left_date > right_date:
                left_date, right_date = right_date, left_date
            return None, left_date, right_date

    explicit_patterns = (
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        (
            r"\b(?:january|february|march|april|may|june|july|"
            r"august|september|october|november|december|"
            r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b"
        ),
        (
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:january|february|march|april|may|june|july|"
            r"august|september|october|november|december|"
            r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
            r"(?:\s+\d{4})?\b"
        ),
    )

    for pattern in explicit_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed = _parse_date_text(match.group(0))
            if parsed:
                return parsed, None, None

    return None, None, None


def _is_session_count_question(question: str) -> bool:
    normalized = question.casefold()
    return (
        ("how many" in normalized or "number of" in normalized)
        and "session" in normalized
    )


def _normalize_role(role: str) -> UserRole:
    """Normalize role names into the UserRole enum."""

    normalized_role = role.strip().lower()

    aliases = {
        "doctor": "clinician",
        "physician": "clinician",
        "care giver": "caregiver",
    }

    normalized_role = aliases.get(
        normalized_role,
        normalized_role,
    )

    try:
        return UserRole(normalized_role)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported user role: {role!r}"
        ) from exc


def _create_authenticated_user(
    *,
    user_id: str,
    user_name: str,
    role: str,
) -> AuthenticatedUser:
    """
    Build the authenticated-user model expected by AuthorizationService.

    In the final UI architecture, the AuthenticatedUser object should ideally
    come directly from the authentication/session service rather than being
    recreated here.
    """

    normalized_username = user_id.strip().upper()

    if not normalized_username:
        raise ValueError("user_id cannot be empty.")

    return AuthenticatedUser(
        user_key=f"user:{normalized_username}",
        username=normalized_username,
        display_name=user_name.strip() or "Authenticated user",
        role=_normalize_role(role),
    )


def _convert_to_dictionary(value: Any) -> Any:
    """Convert nested dataclasses into dictionaries."""

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    if isinstance(value, dict):
        return {
            str(key): _convert_to_dictionary(nested_value)
            for key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [
            _convert_to_dictionary(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _convert_to_dictionary(item)
            for item in value
        ]

    return value


def _remove_sensitive_fields(value: Any) -> Any:
    """
    Remove identifying fields recursively.

    This is a defense-in-depth layer in addition to PrivacyFilter.
    """

    blocked_keys = {
        "patient_id",
        "patientid",
        "patient_key",
        "patientkey",
        "patient_name",
        "patientname",
        "user_id",
        "userid",
        "user_key",
        "userkey",
        "user_name",
        "username",
        "authorized_patient_ids",
        "authorized_patient_keys",
        "authorized_patients",
        "available_patient_ids",
        "display_name",
        "full_name",
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "date_of_birth",
        "dob",
        "filename",
        "file_name",
        "source_file",
        "sourcefile",
        "file_path",
        "filepath",
        "source_path",
        "raw_path",
        "checked_paths",
    }

    if isinstance(value, dict):
        safe_dictionary: dict[str, Any] = {}

        for key, nested_value in value.items():
            normalized_key = (
                str(key)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized_key in blocked_keys:
                continue

            safe_dictionary[str(key)] = (
                _remove_sensitive_fields(
                    nested_value
                )
            )

        return safe_dictionary

    if isinstance(value, list):
        return [
            _remove_sensitive_fields(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _remove_sensitive_fields(item)
            for item in value
        ]

    return value


def _sanitize_sources(
    *,
    sources: list[Any],
    privacy_filter: PrivacyFilter,
    prohibited_values: list[str],
) -> list[str]:
    """Remove identifiers from source labels returned to the UI."""

    safe_sources: list[str] = []

    for source in sources:
        source_text = str(source)

        safe_source = privacy_filter.sanitize_answer(
            source_text,
            prohibited_values=prohibited_values,
        )

        privacy_filter.assert_safe_answer(
            safe_source,
            prohibited_values=prohibited_values,
        )

        safe_sources.append(safe_source)

    return safe_sources


class HealthcareChatbotOrchestrator:
    """
    Coordinates the secure healthcare chatbot workflow.

    Authorization is performed before mobility retrieval. Identifying values
    are removed before the final generation context reaches the LLM.
    """

    def __init__(
        self,
        *,
        knowledge_retriever: ClinicalKnowledgeRetriever | None = None,
        mobility_retriever: MobilityRetriever | None = None,
        llm_service: LLMService | None = None,
        guardrails: HealthcareGuardrails | None = None,
        identity_service: PatientIdentityService | None = None,
        privacy_filter: PrivacyFilter | None = None,
        authorization_service: AuthorizationService | None = None,
        audit_logger: AuditLogger | None = None,
        exercise_recommender: ExerciseRecommendationService | None = None,
    ) -> None:
        self.identity_service = (
            identity_service
            or PatientIdentityService()
        )

        self.privacy_filter = (
            privacy_filter
            or PrivacyFilter()
        )

        self.authorization_service = (
            authorization_service
            or AuthorizationService(
                identity_service=self.identity_service
            )
        )

        self.knowledge_retriever = (
            knowledge_retriever
            or ClinicalKnowledgeRetriever()
        )

        self.mobility_retriever = (
            mobility_retriever
            or MobilityRetriever()
        )

        self.llm_service = (
            llm_service
            or LLMService()
        )

        self.guardrails = (
            guardrails
            or HealthcareGuardrails()
        )

        self.audit_logger = (
            audit_logger
            or AuditLogger()
        )

        self.exercise_recommender = (
            exercise_recommender
            or ExerciseRecommendationService()
        )

    def answer(
        self,
        *,
        question: str,
        user_id: str,
        user_name: str,
        role: str,
        patient_id: str,
        patient_condition: str | None = None,
        authorized_patient_ids: list[str] | None = None,
        current_page: str = "AI Assistant",
        selected_session_id: str | None = None,
        selected_exercise_id: str | None = None,
        selected_date_range: str | None = None,
        visible_metrics: dict[str, Any] | None = None,
        active_alert: dict[str, Any] | None = None,
    ) -> ChatbotResponse:
        """
        Answer a question using an authorized, de-identified context.

        patient_id is accepted temporarily for compatibility with the current
        Streamlit application. It is converted to an opaque patient key before
        authorization and is never included in the LLM context.
        """

        clean_question = question.strip()

        if not clean_question:
            return ChatbotResponse(
                answer="Please enter a question.",
                intent="empty_question",
                intent_confidence=1.0,
                sources=[],
                used_llm=False,
                provider="system",
                model=None,
                safety_action="block",
                escalation_required=False,
                metadata={
                    "safety_reason": "empty_question",
                },
            )

        # ------------------------------------------------------------------
        # 1. Apply input guardrails before retrieving patient data.
        # ------------------------------------------------------------------

        input_decision = self.guardrails.check_input(
            clean_question
        )

        if input_decision.action in {
            "block",
            "block_and_escalate",
        }:
            return ChatbotResponse(
                answer=(
                    input_decision.safe_response
                    or "I cannot answer that request."
                ),
                intent="safety_intervention",
                intent_confidence=1.0,
                sources=[],
                used_llm=False,
                provider="guardrail",
                model=None,
                safety_action=input_decision.action,
                escalation_required=(
                    input_decision.escalation_required
                ),
                metadata={
                    "safety_reason": input_decision.reason,
                },
            )

        # ------------------------------------------------------------------
        # 2. Build authenticated-user model.
        # ------------------------------------------------------------------

        try:
            authenticated_user = _create_authenticated_user(
                user_id=user_id,
                user_name=user_name,
                role=role,
            )
        except ValueError as exc:
            return ChatbotResponse(
                answer=(
                    "The current account has an invalid or "
                    "unsupported user role."
                ),
                intent="invalid_user",
                intent_confidence=1.0,
                sources=[],
                used_llm=False,
                provider="authentication",
                model=None,
                safety_action="block",
                escalation_required=False,
                metadata={
                    "safety_reason": str(exc),
                },
            )

        request_id = self.audit_logger.create_request_id()

        # ------------------------------------------------------------------
        # 3. Convert selected patient ID to an opaque key.
        # ------------------------------------------------------------------

        try:
            requested_patient_key = (
                self.identity_service.create_patient_key(
                    patient_id
                )
            )
        except ValueError as exc:
            return ChatbotResponse(
                answer="No valid patient record was selected.",
                intent="invalid_patient",
                intent_confidence=1.0,
                sources=[],
                used_llm=False,
                provider="authorization",
                model=None,
                safety_action="block",
                escalation_required=False,
                metadata={
                    "safety_reason": str(exc),
                },
            )

        # ------------------------------------------------------------------
        # 4. Authorize access using the tested AuthorizationService.
        # ------------------------------------------------------------------

        access_decision = (
            self.authorization_service
            .authorize_patient_access(
                authenticated_user,
                requested_patient_key,
            )
        )

        if (
            not access_decision.allowed
            or access_decision.patient_key is None
        ):
            self.audit_logger.record(
                event_type="authorization",
                action="chatbot_patient_access",
                user_key=authenticated_user.user_key,
                role=authenticated_user.role.value,
                patient_key=requested_patient_key,
                allowed=False,
                request_id=request_id,
                details={
                    "reason_code": "patient_not_assigned",
                },
            )

            return ChatbotResponse(
                answer=(
                    "You are not authorized to access the "
                    "selected patient record."
                ),
                intent="authorization_denied",
                intent_confidence=1.0,
                sources=[],
                used_llm=False,
                provider="authorization",
                model=None,
                safety_action="block",
                escalation_required=False,
                metadata={
                    "safety_reason": access_decision.reason,
                },
            )

        authorized_patient_key = (
            access_decision.patient_key
        )

        # ------------------------------------------------------------------
        # 5. Validate patient identifiers explicitly mentioned in the
        #    question before retrieving any patient-specific data.
        # ------------------------------------------------------------------

        (
            patient_reference_allowed,
            patient_reference_reason,
            patient_reference_count,
        ) = _validate_question_patient_references(
            question=clean_question,
            selected_patient_id=patient_id,
            authorized_patient_ids=authorized_patient_ids,
        )

        if not patient_reference_allowed:
            self.audit_logger.record(
                event_type="authorization",
                action="question_patient_reference_denied",
                user_key=authenticated_user.user_key,
                role=authenticated_user.role.value,
                patient_key=authorized_patient_key,
                allowed=False,
                request_id=request_id,
                details={
                    "reason_code": patient_reference_reason,
                    "referenced_patient_count": (
                        patient_reference_count
                    ),
                },
            )

            return ChatbotResponse(
                answer=(
                    "I cannot provide information for that patient in "
                    "the current session. Select an authorized patient "
                    "from the dashboard before asking patient-specific "
                    "questions."
                ),
                intent="authorization_denied",
                intent_confidence=1.0,
                sources=[],
                used_llm=False,
                provider="authorization",
                model=None,
                safety_action="block",
                escalation_required=False,
                metadata={
                    "authorization_denied": True,
                    "reason": patient_reference_reason,
                    "authorization_checked": True,
                },
            )

        # Values that must never appear in the LLM prompt or answer.
        prohibited_values = [
            value
            for value in [
                user_id,
                user_name,
                patient_id,
                requested_patient_key,
                authorized_patient_key,
                *(authorized_patient_ids or []),
            ]
            if value
        ]

        # ------------------------------------------------------------------
        # 6. Detect the user's intent.
        # ------------------------------------------------------------------

        intent_result = detect_intent(clean_question)
        rep_id = _extract_rep_id(clean_question)
        session_date, start_date, end_date = _extract_date_filters(
            clean_question
        )

        effective_intent = intent_result.name
        normalized_question = clean_question.casefold()

        has_date_filter = bool(
            session_date or start_date or end_date
        )

        mentions_doctor_feedback = any(
            phrase in normalized_question
            for phrase in (
                "doctor feedback",
                "doctor's feedback",
                "feedback from my doctor",
                "feedback from doctor",
                "what did my doctor say",
                "what has my doctor said",
                "doctor comments",
                "doctor notes",
                "clinician feedback",
                "physician feedback",
            )
        )

        mentions_caregiver_feedback = any(
            phrase in normalized_question
            for phrase in (
                "caregiver feedback",
                "feedback from my caregiver",
                "what did my caregiver say",
                "caregiver comments",
                "caregiver notes",
            )
        )

        mentions_care_team_feedback = any(
            phrase in normalized_question
            for phrase in (
                "care team feedback",
                "feedback from my care team",
                "all feedback",
                "doctor and caregiver feedback",
            )
        )

        mentions_mobility_history = any(
            phrase in normalized_question
            for phrase in (
                "session",
                "sessions",
                "mobility",
                "report",
                "summary",
                "repetitions",
                "metrics",
            )
        )
        

        mentions_trend = any(
            phrase in normalized_question
            for phrase in (
                "improving",
            "improvement",
            "improved",
            "have i improved",
            "has my balance improved",
            "has my mobility improved",
            "has my movement improved",
            "getting better",
            "getting worse",
            "trend",
            "changed",
            "change over time",
            "progress",
            "compare",
            "better than",
            "worse than",
            )
        )

        # Feedback intent has priority so words such as "summary" or a date
        # cannot accidentally reroute a feedback request to mobility history.
        if mentions_doctor_feedback:
            effective_intent = "doctor_feedback"

        elif mentions_caregiver_feedback:
            effective_intent = "caregiver_feedback"

        elif mentions_care_team_feedback:
            effective_intent = "care_team_feedback"

        elif mentions_trend:
            effective_intent = "trend_analysis"

        elif has_date_filter and mentions_mobility_history:
            effective_intent = (
                "sessions_by_date_range"
                if start_date or end_date
                else "sessions_by_date"
            )

        elif _is_session_count_question(clean_question):
            effective_intent = "session_summary"

        # ------------------------------------------------------------------
        # 7. Return condition-based exercises from the controlled JSON.
        #
        # Exercise selection is deterministic and does not use the LLM.
        # ------------------------------------------------------------------

        if effective_intent == "exercise_recommendation":
            exercise_data = self.exercise_recommender.get_recommendations(
                condition=patient_condition,
            )

            profile_name = exercise_data.get(
                "profile_name",
                "General Lower-Limb Mobility",
            )
            exercise_count = len(
                exercise_data.get("exercises", [])
            )

            self.audit_logger.record(
                event_type="chatbot",
                action="retrieve_exercise_recommendations",
                user_key=authenticated_user.user_key,
                role=authenticated_user.role.value,
                patient_key=authorized_patient_key,
                allowed=True,
                request_id=request_id,
                details={
                    "intent": effective_intent,
                    "profile_key": exercise_data.get("profile_key"),
                    "exercise_count": exercise_count,
                    "used_llm": False,
                },
            )

            return ChatbotResponse(
                answer=(
                    f"Here are the clinician-configured exercises for the "
                    f"{profile_name} profile."
                    if exercise_count
                    else (
                        "No exercises are currently configured for this "
                        "patient profile."
                    )
                ),
                intent=effective_intent,
                intent_confidence=intent_result.confidence,
                sources=[],
                used_llm=False,
                provider="exercise_recommendation_service",
                model=None,
                safety_action="allow",
                escalation_required=False,
                metadata={
                    "matched_phrases": list(
                        intent_result.matched_phrases
                    ),
                    "request_id": request_id,
                    "patient_context": "de_identified",
                    "authorization_checked": True,
                    "profile_key": exercise_data.get("profile_key"),
                    "clinical_review_status": exercise_data.get(
                        "clinical_review_status"
                    ),
                    "exercise_count": exercise_count,
                },
                response_type="exercise_recommendations",
                exercise_data=exercise_data,
            )

        # ------------------------------------------------------------------
        # 7. Retrieve mobility data only after authorization.
        # ------------------------------------------------------------------

        mobility_data = self._retrieve_mobility_data(
            intent=effective_intent,
            patient_id=patient_id,
            patient_key=authorized_patient_key,
            session_id=selected_session_id,
            rep_id=rep_id,
            session_date=session_date,
            start_date=start_date,
            end_date=end_date,
        )

        # ------------------------------------------------------------------
        # 8. Retrieve approved clinical knowledge only.
        # ------------------------------------------------------------------

        knowledge_query_intents = {
            "exercise_guidance",
            "metric_explanation",
            "flag_explanation",
            "general_help",
        }

        clinical_knowledge: list[Any] = []

        if effective_intent in knowledge_query_intents:
            clinical_knowledge = (
                self.knowledge_retriever.search(
                    clean_question,
                    top_k=5,
                    exercise_id=selected_exercise_id,
                    approved_only=True,
                )
            )

        # ------------------------------------------------------------------
        # 9. Build existing application contexts.
        #
        # These may temporarily contain identifying backend information.
        # Everything is removed before generation.
        # ------------------------------------------------------------------

        user_context = build_user_context(
            user_id=user_id,
            user_name=user_name,
            role=role,
            patient_id=patient_id,
            authorized_patient_ids=(
                authorized_patient_ids
            ),
        )

        dashboard_context = build_dashboard_context(
            current_page=current_page,
            selected_session_id=selected_session_id,
            selected_exercise_id=selected_exercise_id,
            selected_date_range=selected_date_range,
            visible_metrics=visible_metrics,
            active_alert=active_alert,
        )

        retrieval_context = assemble_retrieval_context(
            question=clean_question,
            intent=effective_intent,
            intent_confidence=intent_result.confidence,
            user_context=user_context,
            dashboard_context=dashboard_context,
            mobility_data=mobility_data,
            clinical_knowledge=clinical_knowledge,
        )

        # ------------------------------------------------------------------
        # 10. Create the exact de-identified LLM generation context.
        # ------------------------------------------------------------------

        raw_generation_context = _convert_to_dictionary(
            retrieval_context
        )

        safe_generation_context = _remove_sensitive_fields(
            raw_generation_context
        )

        safe_generation_context = (
            self.privacy_filter.sanitize_context(
                safe_generation_context
            )
        )

        safe_generation_context["patient_reference"] = (
            "current_patient"
        )

        self.privacy_filter.assert_safe_context(
            safe_generation_context,
            prohibited_values=prohibited_values,
        )

        # ------------------------------------------------------------------
        # 11. Generate the answer.
        # ------------------------------------------------------------------

        generation = self.llm_service.generate(
            safe_generation_context
        )

        generated_text = generation.text

        # ------------------------------------------------------------------
        # 12. Apply medical output guardrails.
        # ------------------------------------------------------------------

        sufficient_evidence = bool(
            getattr(
                retrieval_context,
                "sufficient_evidence",
                False,
            )
        )

        output_decision = self.guardrails.check_output(
            generated_text,
            sufficient_evidence=sufficient_evidence,
        )

        if output_decision.action == "replace":
            answer = (
                output_decision.safe_response
                or generated_text
            )
        else:
            answer = generated_text

        escalation_required = (
            input_decision.escalation_required
            or output_decision.escalation_required
        )

        if (
            input_decision.action
            == "allow_with_escalation"
            and input_decision.safe_response
        ):
            answer = (
                answer.rstrip()
                + "\n\n"
                + input_decision.safe_response
            )

        # ------------------------------------------------------------------
        # 13. Sanitize and validate generated output.
        # ------------------------------------------------------------------

        answer = self.privacy_filter.sanitize_answer(
            answer,
            prohibited_values=prohibited_values,
        )

        self.privacy_filter.assert_safe_answer(
            answer,
            prohibited_values=prohibited_values,
        )

        answer = self.guardrails.add_standard_disclaimer(
            answer,
            role=role,
        )

        answer = self.privacy_filter.sanitize_answer(
            answer,
            prohibited_values=prohibited_values,
        )

        self.privacy_filter.assert_safe_answer(
            answer,
            prohibited_values=prohibited_values,
        )

        # ------------------------------------------------------------------
        # 14. Sanitize sources returned to the UI.
        # ------------------------------------------------------------------

        raw_sources = list(
            getattr(
                retrieval_context,
                "sources",
                [],
            )
        )

        safe_sources = _sanitize_sources(
            sources=raw_sources,
            privacy_filter=self.privacy_filter,
            prohibited_values=prohibited_values,
        )

        # ------------------------------------------------------------------
        # 15. Record a privacy-safe successful chatbot audit event.
        # ------------------------------------------------------------------

        self.audit_logger.record(
            event_type="chatbot",
            action="answer_question",
            user_key=authenticated_user.user_key,
            role=authenticated_user.role.value,
            patient_key=authorized_patient_key,
            allowed=True,
            request_id=request_id,
            details={
                "intent": effective_intent,
                "used_llm": generation.used_llm,
                "provider": generation.provider,
                "safety_action": output_decision.action,
            },
        )

        # ------------------------------------------------------------------
        # 16. Return only privacy-safe metadata.
        # ------------------------------------------------------------------

        return ChatbotResponse(
            answer=answer,
            intent=effective_intent,
            intent_confidence=intent_result.confidence,
            sources=safe_sources,
            used_llm=generation.used_llm,
            provider=generation.provider,
            model=generation.model,
            safety_action=output_decision.action,
            escalation_required=escalation_required,
            metadata={
                "matched_phrases": list(
                    intent_result.matched_phrases
                ),
                "llm_error": generation.error,
                "request_id": request_id,
                "llm_request_id": generation.request_id,
                "mobility_available": bool(
                    mobility_data.get("available")
                ),
                "mobility_data_source": mobility_data.get(
                    "data_source"
                ),
                "session_date_filter": session_date,
                "start_date_filter": start_date,
                "end_date_filter": end_date,
                "knowledge_results": len(
                    clinical_knowledge
                ),
                "safety_reason": output_decision.reason,
                "patient_context": "de_identified",
                "authorization_checked": True,
                "approved_knowledge_only": True,
            },
        )

    def _retrieve_mobility_data(
        self,
        *,
        intent: str,
        patient_id: str,
        patient_key: str,
        session_id: str | None,
        rep_id: int | None,
        session_date: str | None,
        start_date: str | None,
        end_date: str | None,
    ) -> dict[str, Any]:
        """
        Retrieve mobility data using the new patient_key API when available.

        During the transition, the existing retriever may still accept
        patient_id. The patient ID is used only inside this trusted backend
        call and is removed from the result immediately.
        """

        retrieve_method = (
            self.mobility_retriever.retrieve_for_intent
        )

        try:
            parameters = inspect.signature(
                retrieve_method
            ).parameters
        except (TypeError, ValueError):
            parameters = {}

        arguments: dict[str, Any] = {
            "intent": intent,
            "session_id": session_id,
            "rep_id": rep_id,
        }

        if "session_date" in parameters:
            arguments["session_date"] = session_date
        if "start_date" in parameters:
            arguments["start_date"] = start_date
        if "end_date" in parameters:
            arguments["end_date"] = end_date

        if "patient_key" in parameters:
            arguments["patient_key"] = patient_key
        elif "patient_id" in parameters:
            arguments["patient_id"] = patient_id
        else:
            raise TypeError(
                "MobilityRetriever.retrieve_for_intent() must "
                "accept patient_key or patient_id."
            )

        result = retrieve_method(**arguments)

        if not isinstance(result, dict):
            return {
                "available": False,
                "error": (
                    "Mobility retrieval returned an "
                    "invalid result."
                ),
            }

        safe_result = _remove_sensitive_fields(
            _convert_to_dictionary(result)
        )

        safe_result = (
            self.privacy_filter.sanitize_context(
                safe_result
            )
        )

        return safe_result


def answer_question(
    question: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function used by the current Streamlit chatbot UI.

    This maintains backward compatibility with the existing dictionary-based
    chatbot context until Person 3 updates the UI to use AuthenticatedUser and
    opaque patient references directly.
    """

    required_fields = {
        "user_id",
        "role",
        "patient_id",
    }

    missing_fields = [
        field
        for field in required_fields
        if not context.get(field)
    ]

    if missing_fields:
        response = ChatbotResponse(
            answer=(
                "The chatbot session is missing required "
                "authentication information."
            ),
            intent="invalid_context",
            intent_confidence=1.0,
            sources=[],
            used_llm=False,
            provider="system",
            model=None,
            safety_action="block",
            escalation_required=False,
            metadata={
                "missing_fields": missing_fields,
            },
        )

        return asdict(response)

    orchestrator = HealthcareChatbotOrchestrator()

    response = orchestrator.answer(
        question=question,
        user_id=str(context["user_id"]),
        user_name=str(
            context.get(
                "user_name",
                "Authenticated user",
            )
        ),
        role=str(context["role"]),
        patient_id=str(context["patient_id"]),
        patient_condition=(
            str(context["patient_condition"])
            if context.get("patient_condition") is not None
            else None
        ),
        authorized_patient_ids=context.get(
            "authorized_patient_ids"
        ),
        current_page=str(
            context.get(
                "current_page",
                "AI Assistant",
            )
        ),
        selected_session_id=context.get(
            "selected_session_id"
        ),
        selected_exercise_id=context.get(
            "selected_exercise_id"
        ),
        selected_date_range=context.get(
            "selected_date_range"
        ),
        visible_metrics=context.get(
            "visible_metrics"
        ),
        active_alert=context.get(
            "active_alert"
        ),
    )

    return asdict(response)