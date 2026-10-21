from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.chatbot.intent import closest_sessions, describe_interpretation
from app.chatbot.mobility_retriever import MobilityRetriever
from app.chatbot.orchestrator import ChatResult, HealthcareChatbotOrchestrator
from app.security.audit import AuditLogger


class FakeLLM:
    def generate(self, context):
        class Generation:
            text = "The recorded session looked steady."
            used_llm = False
            provider = "fake"
            model = None
            error = None
            request_id = None

        return Generation()


def _write_session(directory: Path, stamp: str, legs: list[str]) -> None:
    pd.DataFrame(
        [
            {
                "rep_id": index + 1,
                "leg": leg,
                "shin_lift_angle_deg": 60.0 if leg == "left" else 80.0,
                "hold_duration_s": 1.5,
                "jerk_score": 0.4,
                "flagged": False,
            }
            for index, leg in enumerate(legs)
        ]
    ).to_csv(directory / f"session_{stamp}_P1001_reps.csv", index=False)


@pytest.fixture
def orchestrator(tmp_path: Path) -> HealthcareChatbotOrchestrator:
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    for stamp in ("20260101_100000", "20260105_100000", "20260109_100000"):
        _write_session(synthetic_dir, stamp, ["left", "right", "left", "right"])

    retriever = MobilityRetriever(
        reps_csv=tmp_path / "missing.csv",
        synthetic_dir=synthetic_dir,
        enable_demo_feedback=False,
    )

    return HealthcareChatbotOrchestrator(
        mobility_retriever=retriever,
        llm_service=FakeLLM(),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )


def _ask(bot: HealthcareChatbotOrchestrator, question: str) -> ChatResult:
    return bot.answer(
        question=question,
        user_id="P1001",
        user_name="Zyxw Qrst",
        role="Patient",
        patient_id="P1001",
        authorized_patient_ids=["P1001"],
    )


def test_describe_interpretation():
    assert (
        describe_interpretation(
            intent="session_summary",
            resolved_dates=["2026-03-14T10:00:00"],
        )
        == "Latest session (Mar 14, 2026)"
    )
    assert (
        describe_interpretation(
            intent="sessions_by_date",
            session_date="2026-03-14",
        )
        == "Sessions on Mar 14, 2026"
    )
    assert describe_interpretation(intent="general_help") is None


def test_closest_sessions():
    dates = ["2026-01-01", "2026-01-05", "2026-01-09", "2026-01-20"]
    assert closest_sessions("2026-01-06", dates) == [
        "2026-01-01",
        "2026-01-05",
        "2026-01-09",
    ]
    assert closest_sessions(None, dates, limit=2) == ["2026-01-09", "2026-01-20"]
    assert closest_sessions("2026-01-06", []) == []


def test_summary_answer_has_interpretation_sources_and_chart(orchestrator):
    result = _ask(orchestrator, "Summarize my latest mobility session.")

    assert result.kind == "answer"
    assert result.interpretation == "Latest session (Jan 9, 2026)"
    assert result.session_sources[0]["session_id"] == "20260109_100000"
    assert result.session_sources[0]["session_date"] == "2026-01-09"
    assert "P1001" not in repr(result.session_sources)
    assert result.chart_data["rows"] == [
        {"session": "2026-01-09", "Left": 60.0, "Right": 80.0}
    ]
    assert result.chart_data["chart_kind"] == "bar"


def test_trend_answer_uses_line_chart(orchestrator):
    result = _ask(orchestrator, "Am I improving over time?")

    assert result.kind == "answer"
    assert result.chart_data["chart_kind"] == "line"
    assert len(result.chart_data["rows"]) == 3


def test_missing_date_returns_no_data_with_nearby_dates(orchestrator):
    result = _ask(orchestrator, "Summarize my mobility session on 2026-01-06.")

    assert result.kind == "no_data"
    assert result.used_llm is False
    assert result.nearby_dates == ["2026-01-01", "2026-01-05", "2026-01-09"]


def test_unauthorized_patient_reference_is_denied(orchestrator):
    result = _ask(orchestrator, "Show the latest session for P1002.")

    assert result.kind == "denied"
    assert result.session_sources == []


def test_report_callback_receives_stage_messages(orchestrator):
    messages: list[str] = []
    orchestrator.answer(
        question="Summarize my latest mobility session.",
        user_id="P1001",
        user_name="Zyxw Qrst",
        role="Patient",
        patient_id="P1001",
        authorized_patient_ids=["P1001"],
        report=messages.append,
    )

    assert "Retrieving session data…" in messages
    assert "Writing response…" in messages
    assert messages.index("Retrieving session data…") < messages.index(
        "Writing response…"
    )


def test_list_session_dates_requires_authorization(orchestrator):
    assert orchestrator.list_session_dates(
        user_id="P1001",
        user_name="Zyxw Qrst",
        role="Patient",
        patient_id="P1001",
    ) == ["2026-01-01", "2026-01-05", "2026-01-09"]

    assert orchestrator.list_session_dates(
        user_id="P1001",
        user_name="Zyxw Qrst",
        role="Patient",
        patient_id="P1002",
    ) == []
