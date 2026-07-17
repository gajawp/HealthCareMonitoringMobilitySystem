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
        "file_path",
        "filepath",
        "source_path",
        "raw_path",
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

    def answer(
        self,
        *,
        question: str,
        user_id: str,
        user_name: str,
        role: str,
        patient_id: str,
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

        # ------------------------------------------------------------------
        # 7. Retrieve mobility data only after authorization.
        # ------------------------------------------------------------------

        mobility_data = self._retrieve_mobility_data(
            intent=intent_result.name,
            patient_id=patient_id,
            patient_key=authorized_patient_key,
            session_id=selected_session_id,
            rep_id=rep_id,
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

        if intent_result.name in knowledge_query_intents:
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
            intent=intent_result.name,
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
                "intent": intent_result.name,
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
            intent=intent_result.name,
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