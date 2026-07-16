from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

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


@dataclass
class ChatbotResponse:
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


def _extract_rep_id(question: str) -> int | None:
    patterns = (
        r"\brep(?:etition)?\s*#?\s*(\d+)\b",
        r"\bnumber\s+(\d+)\b",
    )

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


class HealthcareChatbotOrchestrator:
    """Coordinate intent, retrieval, generation, and guardrails."""

    def __init__(
        self,
        *,
        knowledge_retriever: ClinicalKnowledgeRetriever | None = None,
        mobility_retriever: MobilityRetriever | None = None,
        llm_service: LLMService | None = None,
        guardrails: HealthcareGuardrails | None = None,
    ) -> None:
        self.knowledge_retriever = (
            knowledge_retriever or ClinicalKnowledgeRetriever()
        )
        self.mobility_retriever = mobility_retriever or MobilityRetriever()
        self.llm_service = llm_service or LLMService()
        self.guardrails = guardrails or HealthcareGuardrails()

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
        user_context = build_user_context(
            user_id=user_id,
            user_name=user_name,
            role=role,
            patient_id=patient_id,
            authorized_patient_ids=authorized_patient_ids,
        )

        dashboard_context = build_dashboard_context(
            current_page=current_page,
            selected_session_id=selected_session_id,
            selected_exercise_id=selected_exercise_id,
            selected_date_range=selected_date_range,
            visible_metrics=visible_metrics,
            active_alert=active_alert,
        )

        input_decision = self.guardrails.check_input(question)

        if input_decision.action in {"block", "block_and_escalate"}:
            return ChatbotResponse(
                answer=input_decision.safe_response or "I cannot answer that request.",
                intent="safety_intervention",
                intent_confidence=1.0,
                sources=[],
                used_llm=False,
                provider="guardrail",
                model=None,
                safety_action=input_decision.action,
                escalation_required=input_decision.escalation_required,
                metadata={"safety_reason": input_decision.reason},
            )

        intent_result = detect_intent(question)
        rep_id = _extract_rep_id(question)

        mobility_data = self.mobility_retriever.retrieve_for_intent(
            intent=intent_result.name,
            patient_id=patient_id,
            session_id=selected_session_id,
            rep_id=rep_id,
        )

        knowledge_query_intents = {
            "exercise_guidance",
            "metric_explanation",
            "flag_explanation",
            "general_help",
        }

        clinical_knowledge = []
        if intent_result.name in knowledge_query_intents:
            clinical_knowledge = self.knowledge_retriever.search(
                question,
                top_k=5,
                exercise_id=selected_exercise_id,
                approved_only=False,
            )

        retrieval_context = assemble_retrieval_context(
            question=question,
            intent=intent_result.name,
            intent_confidence=intent_result.confidence,
            user_context=user_context,
            dashboard_context=dashboard_context,
            mobility_data=mobility_data,
            clinical_knowledge=clinical_knowledge,
        )

        generation_context = asdict(retrieval_context)
        generation = self.llm_service.generate(generation_context)

        output_decision = self.guardrails.check_output(
            generation.text,
            sufficient_evidence=retrieval_context.sufficient_evidence,
        )

        if output_decision.action == "replace":
            answer = output_decision.safe_response or generation.text
        else:
            answer = generation.text

        escalation_required = (
            input_decision.escalation_required
            or output_decision.escalation_required
        )

        if (
            input_decision.action == "allow_with_escalation"
            and input_decision.safe_response
        ):
            answer = answer.rstrip() + "\n\n" + input_decision.safe_response

        answer = self.guardrails.add_standard_disclaimer(answer, role=role)

        return ChatbotResponse(
            answer=answer,
            intent=intent_result.name,
            intent_confidence=intent_result.confidence,
            sources=retrieval_context.sources,
            used_llm=generation.used_llm,
            provider=generation.provider,
            model=generation.model,
            safety_action=output_decision.action,
            escalation_required=escalation_required,
            metadata={
                "matched_phrases": list(intent_result.matched_phrases),
                "llm_error": generation.error,
                "request_id": generation.request_id,
                "mobility_available": bool(mobility_data.get("available")),
                "knowledge_results": len(clinical_knowledge),
                "safety_reason": output_decision.reason,
            },
        )


def answer_question(
    question: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Convenience function for Streamlit integration."""
    orchestrator = HealthcareChatbotOrchestrator()

    response = orchestrator.answer(
        question=question,
        user_id=str(context["user_id"]),
        user_name=str(context.get("user_name", context["user_id"])),
        role=str(context["role"]),
        patient_id=str(context["patient_id"]),
        authorized_patient_ids=context.get("authorized_patient_ids"),
        current_page=str(context.get("current_page", "AI Assistant")),
        selected_session_id=context.get("selected_session_id"),
        selected_exercise_id=context.get("selected_exercise_id"),
        selected_date_range=context.get("selected_date_range"),
        visible_metrics=context.get("visible_metrics"),
        active_alert=context.get("active_alert"),
    )

    return asdict(response)
