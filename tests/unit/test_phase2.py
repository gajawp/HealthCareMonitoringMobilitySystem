from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


from app.chatbot.context_builder import build_user_context
from app.chatbot.guardrails import HealthcareGuardrails
from app.chatbot.intent import detect_intent
from app.chatbot.llm_service import LLMService
from app.chatbot.mobility_retriever import MobilityRetriever
from app.chatbot.orchestrator import HealthcareChatbotOrchestrator
from app.chatbot.knowledge_retriever import ClinicalKnowledgeRetriever


def test_intent_detection():
    assert detect_intent("Summarize my latest session").name == "session_summary"
    assert detect_intent("Why was rep 4 flagged?").name == "flag_explanation"
    assert detect_intent("How should I do a knee extension?").name == "exercise_guidance"


def test_patient_authorization():
    build_user_context(
        user_id="P1001",
        user_name="John Smith",
        role="Patient",
        patient_id="P1001",
    )

    with pytest.raises(PermissionError):
        build_user_context(
            user_id="P1001",
            user_name="John Smith",
            role="Patient",
            patient_id="P1002",
        )


def test_mobility_summary_and_flags(tmp_path: Path):
    csv_path = tmp_path / "reps.csv"
    pd.DataFrame(
        [
            {
                "session_id": "S1",
                "rep_id": 1,
                "leg": "L",
                "hold_duration_s": 2.0,
                "jerk_score": 1.2,
                "flagged": False,
                "flag_hold": False,
                "flag_angle": False,
                "flag_jerk": False,
            },
            {
                "session_id": "S1",
                "rep_id": 2,
                "leg": "R",
                "hold_duration_s": 0.7,
                "jerk_score": 4.1,
                "flagged": True,
                "flag_hold": True,
                "flag_angle": False,
                "flag_jerk": True,
            },
        ]
    ).to_csv(csv_path, index=False)

    retriever = MobilityRetriever(
        reps_csv=csv_path,
        frames_csv=tmp_path / "missing_frames.csv",
        pose_csv=tmp_path / "missing_pose.csv",
        session_json=tmp_path / "missing.json",
        tremor_csv=tmp_path / "missing_tremor.csv",
    )

    summary = retriever.get_session_summary(patient_id="P1001")
    assert summary["total_repetitions"] == 2
    assert summary["flagged_repetitions"] == 1

    flags = retriever.get_flagged_repetitions(patient_id="P1001")
    assert flags["total_flagged"] == 1
    assert "hold duration" in flags["flagged_repetitions"][0]["reasons"]


def test_emergency_guardrail():
    decision = HealthcareGuardrails().check_input(
        "I fell and hit my head during the exercise."
    )
    assert decision.action == "block_and_escalate"
    assert decision.escalation_required is True


def test_orchestrator_fallback(tmp_path: Path):
    csv_path = tmp_path / "reps.csv"
    pd.DataFrame(
        [
            {
                "session_id": "S1",
                "rep_id": 1,
                "leg": "L",
                "hold_duration_s": 2.0,
                "jerk_score": 1.0,
                "flagged": False,
            }
        ]
    ).to_csv(csv_path, index=False)

    mobility = MobilityRetriever(
        reps_csv=csv_path,
        frames_csv=tmp_path / "missing_frames.csv",
        pose_csv=tmp_path / "missing_pose.csv",
        session_json=tmp_path / "missing.json",
        tremor_csv=tmp_path / "missing_tremor.csv",
    )

    orchestrator = HealthcareChatbotOrchestrator(
        mobility_retriever=mobility,
        knowledge_retriever=ClinicalKnowledgeRetriever(),
        llm_service=LLMService(use_llm=False),
    )

    response = orchestrator.answer(
        question="Summarize my latest session.",
        user_id="P1001",
        user_name="John Smith",
        role="Patient",
        patient_id="P1001",
    )

    assert response.intent == "session_summary"
    assert response.used_llm is False
    assert "1 repetitions" in response.answer
