from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from app import dashboard_data as dd
from app.chatbot.mobility_retriever import MobilityRetriever


def _session(day: str, jerk: float, flagged: int = 0, total: int = 10, source: str = "synthetic"):
    return {
        "session_id": day.replace("-", "") + "_100000",
        "session_datetime": f"{day}T10:00:00",
        "data_sources": [source],
        "total_repetitions": total,
        "flagged_repetitions": flagged,
        "metric_means": {"jerk_score": jerk, "duration_s": 3.0},
        "metric_means_by_leg": {
            "Left": {"shin_lift_angle_deg": 60.0},
            "Right": {"shin_lift_angle_deg": 80.0},
        },
    }


def test_stable_seed_is_deterministic():
    assert dd.stable_seed("P1001", "Seated Marching") == dd.stable_seed(
        "P1001", "Seated Marching"
    )
    assert dd.stable_seed("P1001", "a") != dd.stable_seed("P1002", "a")


def test_jerk_trend():
    assert dd.jerk_trend([_session("2026-01-01", 0.5)]) is None
    assert dd.jerk_trend([_session("2026-01-01", 0.5), _session("2026-01-03", 0.4)]) == "improving"
    assert dd.jerk_trend([_session("2026-01-01", 0.4), _session("2026-01-03", 0.5)]) == "declining"
    assert dd.jerk_trend([_session("2026-01-01", 0.4), _session("2026-01-03", 0.41)]) == "steady"


def test_alerts_come_from_the_patients_own_sessions():
    today = datetime.date(2026, 1, 10)
    sessions = [_session("2026-01-01", 0.4), _session("2026-01-09", 0.6, flagged=6)]

    alerts = dd.build_alerts(sessions, today=today)
    types = {alert["type"]: alert for alert in alerts}

    assert types["Rep Quality Flags"]["severity"] == "high"
    assert types["Rep Quality Flags"]["time"] == "2026-01-09"
    assert "Smoothness Declining" in types
    assert "No Recent Session" not in types
    # No hardcoded alerts are added for every patient.
    assert "Fall Risk" not in types and "Inactivity" not in types


def test_stale_and_missing_sessions():
    stale = dd.build_alerts([_session("2026-01-01", 0.4)], today=datetime.date(2026, 3, 1))
    assert [alert["type"] for alert in stale] == ["No Recent Session"]

    empty = dd.build_alerts([])
    assert empty[0]["type"] == "No Sessions" and empty[0]["severity"] == "low"


def test_patient_status_sorts_urgent_first():
    calm = dd.patient_status([_session("2026-01-01", 0.4)], [], 0)
    urgent = dd.patient_status(
        [_session("2026-01-01", 0.4, flagged=8)],
        [{"severity": "high"}],
        2,
    )
    assert urgent["attention_score"] > calm["attention_score"]
    assert urgent["message_count"] == 2
    assert calm["last_session"] == "2026-01-01"


def test_history_and_date_filter():
    history = dd.chair_session_history(
        [_session("2026-01-01", 0.4), _session("2026-01-05", 0.3)],
        "Jerk Score",
    )
    assert list(history["Day"]) == [datetime.date(2026, 1, 1), datetime.date(2026, 1, 5)]
    assert list(history["Jerk Score"]) == [0.4, 0.3]

    filtered = dd.filter_history_by_dates(
        history, datetime.date(2026, 1, 2), datetime.date(2026, 1, 9)
    )
    assert list(filtered["Day"]) == [datetime.date(2026, 1, 5)]


def test_laterality_rows():
    assert dd.laterality_rows([_session("2026-01-01", 0.4)]) == [
        {"session": "2026-01-01", "Left": 60.0, "Right": 80.0}
    ]


def test_patient_sessions_are_patient_scoped(tmp_path: Path):
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    for patient, stamp in (("P1001", "20260101_100000"), ("P1002", "20260102_100000")):
        pd.DataFrame(
            [{"rep_id": 1, "leg": "left", "jerk_score": 0.4, "flagged": False}]
        ).to_csv(synthetic / f"session_{stamp}_{patient}_reps.csv", index=False)

    retriever = MobilityRetriever(reps_csv=tmp_path / "missing.csv", synthetic_dir=synthetic)

    sessions = dd.patient_sessions(retriever, "P1001")
    assert [s["session_id"] for s in sessions] == ["20260101_100000"]
    reps = retriever.get_session_repetitions(patient_id="P1002")
    assert "patient_id" not in reps.columns and len(reps) == 1


def test_report_pdf_is_generated():
    pdf = dd.build_report_pdf(
        patient_label="Test Patient (P1001)",
        start=datetime.date(2026, 1, 1),
        end=datetime.date(2026, 1, 31),
        exercise_rows=[{
            "exercise": "Seated Leg Raises",
            "latest": 0.4, "metric": "Jerk Score", "target": 0.35, "status": "On Track",
        }],
        sessions=[_session("2026-01-05", 0.4)],
    )
    assert pdf.startswith(b"%PDF")
