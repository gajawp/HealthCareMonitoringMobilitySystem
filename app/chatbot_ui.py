from __future__ import annotations

from typing import Any

import streamlit as st

from app.chatbot.orchestrator import (
    HealthcareChatbotOrchestrator,
)


def get_chatbot() -> HealthcareChatbotOrchestrator:
    """
    Create a fresh secure healthcare chatbot orchestrator.

    Caching is intentionally disabled while authentication, mobility retrieval,
    authorization, privacy filtering, and audit logging are being validated.
    """

    return HealthcareChatbotOrchestrator()


def _history_key(user_id: str, patient_id: str) -> str:
    return f"mobility_chat_history::{user_id}::{patient_id}"


def _initial_message(user_name: str, role: str) -> dict[str, str]:
    role_intro = {
        "Patient": (
            "I can summarize your recorded sessions, explain alerts and mobility "
            "metrics, and retrieve approved exercise guidance."
        ),
        "Caregiver": (
            "I can summarize the selected patient's recorded sessions, explain "
            "alerts, and retrieve approved exercise guidance."
        ),
        "Clinician": (
            "I can provide concise summaries of the selected patient's recorded "
            "sessions, flagged repetitions, mobility metrics, and source-backed "
            "exercise reference material."
        ),
    }

    return {
        "role": "assistant",
        "content": (
            f"Hello {user_name}. {role_intro.get(role, role_intro['Patient'])}"
        ),
    }


def _suggested_questions(
    role: str,
    selected_exercise: str,
) -> list[tuple[str, str]]:
    """Return role-specific prompts using the currently selected exercise."""

    exercise_prompt = f"How should I perform {selected_exercise}?"

    if role == "Clinician":
        return [
            (
                "📋 Session summary",
                "Summarize the latest recorded mobility session.",
            ),
            (
                "⚠️ Flagged repetitions",
                "Explain the flagged repetitions.",
            ),
            (
                "📐 Metric explanation",
                "What does jerk score mean?",
            ),
            (
                f"🦵 Guidance: {selected_exercise}",
                exercise_prompt,
            ),
        ]

    if role == "Caregiver":
        return [
            (
                "📋 Latest session",
                "Summarize the latest mobility session.",
            ),
            (
                "⚠️ Explain alerts",
                "Why were repetitions flagged?",
            ),
            (
                "📈 Progress",
                "Is the patient improving over time?",
            ),
            (
                f"🦵 Guidance: {selected_exercise}",
                exercise_prompt,
            ),
        ]

    return [
        (
            "📋 My latest session",
            "Summarize my latest mobility session.",
        ),
        (
            "⚠️ Explain my alerts",
            "Why were my repetitions flagged?",
        ),
        (
            "📈 Am I improving?",
            "Am I improving over time?",
        ),
        (
            f"🦵 Guidance: {selected_exercise}",
            exercise_prompt,
        ),
    ]

def _render_sources(sources: list[str]) -> None:
    if not sources:
        return

    with st.expander("Knowledge sources used"):
        for source in sources:
            st.markdown(f"- `{source}`")


def render_chatbot(
    *,
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
) -> None:
    """Render the Phase 3 chatbot inside the existing dashboard."""
    st.title("💬 AI Healthcare Mobility Assistant")

    st.info(
        "This assistant explains recorded dashboard information and approved "
        "exercise material. It does not diagnose conditions or change a care plan."
    )

    st.caption(
        f"Signed in as **{user_name}** · Role: **{role}** · "
        f"Selected patient: **{patient_id}**"
    )

    key = _history_key(user_id, patient_id)

    if key not in st.session_state:
        st.session_state[key] = [_initial_message(user_name, role)]

    st.markdown("### Suggested questions")

    exercise_options = [
        "seated knee extension",
        "sit-to-stand exercise",
        "seated marching",
        "finger tapping drill",
        "hand open-close repetitions",
        "resting tremor assessment",
        "leg raise exercise",
    ]

    default_exercise = (
        selected_exercise_id.strip().replace("_", " ")
        if isinstance(selected_exercise_id, str)
        and selected_exercise_id.strip()
        else exercise_options[0]
    )

    if default_exercise not in exercise_options:
        exercise_options.insert(0, default_exercise)

    selected_exercise = st.selectbox(
        "Choose an exercise for guidance",
        options=exercise_options,
        index=exercise_options.index(default_exercise),
        key=f"exercise_guidance::{user_id}::{patient_id}",
    )

    suggestions = _suggested_questions(
        role,
        selected_exercise,
    )
    columns = st.columns(2)
    selected_prompt: str | None = None

    for index, (label, prompt) in enumerate(suggestions):
        with columns[index % 2]:
            if st.button(
                label,
                key=f"suggest::{user_id}::{patient_id}::{index}",
                width="stretch",
            ):
                selected_prompt = prompt

    st.divider()

    for message in st.session_state[key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    typed_prompt = st.chat_input(
        "Ask about sessions, alerts, mobility metrics, or exercises"
    )
    prompt = typed_prompt or selected_prompt

    if prompt:
        st.session_state[key].append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                bot = get_chatbot()
                with st.spinner("Reviewing mobility data and approved guidance..."):
                    response = bot.answer(
                        question=prompt,
                        user_id=user_id,
                        user_name=user_name,
                        role=role,
                        patient_id=patient_id,
                        authorized_patient_ids=authorized_patient_ids,
                        current_page=current_page,
                        selected_session_id=selected_session_id,
                        selected_exercise_id=selected_exercise_id,
                        selected_date_range=selected_date_range,
                        visible_metrics=visible_metrics,
                        active_alert=active_alert,
                    )

                st.markdown(response.answer)
                _render_sources(response.sources)

                if response.escalation_required:
                    st.warning("This response includes a care-team or urgent escalation.")

                if role == "Clinician":
                    with st.expander("Technical response details"):
                        st.json(
                            {
                                "intent": response.intent,
                                "intent_confidence": response.intent_confidence,
                                "provider": response.provider,
                                "model": response.model,
                                "used_llm": response.used_llm,
                                "safety_action": response.safety_action,
                                "metadata": response.metadata,
                            }
                        )

                assistant_message = {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                }

            except Exception as exc:
                error_text = (
                    "The mobility assistant could not complete this request. "
                    "Please review the application terminal for the technical error."
                )
                st.error(error_text)
                st.exception(exc)
                assistant_message = {
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                }

        st.session_state[key].append(assistant_message)
        st.rerun()

    st.divider()
    clear_col, status_col = st.columns([1, 3])

    with clear_col:
        if st.button(
            "Clear conversation",
            key=f"clear::{user_id}::{patient_id}",
        ):
            st.session_state[key] = [_initial_message(user_name, role)]
            st.rerun()

    with status_col:
        st.caption(
            "Conversation history is stored only in the current Streamlit session."
        )