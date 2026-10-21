"""
Dashboard data built from the same recorded sessions the chatbot uses.

Everything here is plain Python/pandas (no Streamlit) so it can be unit
tested. main.py wraps these functions with Streamlit caching.
"""

from __future__ import annotations

import datetime
import zlib
from typing import Any

import pandas as pd

from app.chatbot.mobility_retriever import MobilityRetriever


# Sessions older than this produce an "no recent session" notice.
STALE_SESSION_DAYS = 14

# Relative change in mean jerk score treated as a real trend, not noise.
TREND_THRESHOLD = 0.10


def stable_seed(*parts: str) -> int:
    """
    Deterministic seed for demo data.

    Python's built-in hash() is randomized per process, which made mock
    charts change on every app restart.
    """

    return zlib.crc32("::".join(parts).encode("utf-8"))


def patient_sessions(
    retriever: MobilityRetriever,
    patient_id: str,
) -> list[dict[str, Any]]:
    """All recorded sessions for one patient, oldest first."""

    result = retriever.get_sessions_by_date_range(patient_id=patient_id)
    return list(result.get("sessions") or [])


def session_date(session: dict[str, Any]) -> datetime.date | None:
    value = str(session.get("session_datetime") or "")[:10]
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def chair_session_history(
    sessions: list[dict[str, Any]],
    metric_label: str,
) -> pd.DataFrame:
    """
    One row per recorded session, dated, for chair exercise charts.

    Columns: Day (date), Duration (min), <metric_label> (mean jerk score).
    """

    rows = []
    for session in sessions:
        day = session_date(session)
        means = session.get("metric_means") or {}
        if day is None or "jerk_score" not in means:
            continue

        mean_duration_s = means.get("duration_s") or 0.0
        rows.append(
            {
                "Day": day,
                "Duration (min)": round(
                    mean_duration_s * session.get("total_repetitions", 0) / 60.0,
                    2,
                ),
                metric_label: means["jerk_score"],
            }
        )

    return pd.DataFrame(rows, columns=["Day", "Duration (min)", metric_label])


def jerk_trend(sessions: list[dict[str, Any]]) -> str | None:
    """
    Compare the latest session's mean jerk score with up to three earlier.

    Lower jerk means smoother movement. Returns "improving", "declining",
    "steady", or None when there are fewer than two sessions.
    """

    values = [
        (session.get("metric_means") or {}).get("jerk_score")
        for session in sessions
    ]
    values = [value for value in values if value is not None]

    if len(values) < 2:
        return None

    latest = values[-1]
    baseline = sum(values[-4:-1]) / len(values[-4:-1])

    if baseline <= 0:
        return "steady"

    change = (latest - baseline) / baseline
    if change <= -TREND_THRESHOLD:
        return "improving"
    if change >= TREND_THRESHOLD:
        return "declining"
    return "steady"


TREND_ARROWS = {
    "improving": "↑ improving",
    "declining": "↓ declining",
    "steady": "→ steady",
}


def build_alerts(
    sessions: list[dict[str, Any]],
    *,
    glove_sessions: list[dict[str, Any]] | None = None,
    today: datetime.date | None = None,
) -> list[dict[str, Any]]:
    """
    Alerts derived from this patient's own recorded data only.

    Each alert: {"time", "type", "message", "severity", "source"} with
    severity "high", "medium" or "low".
    """

    today = today or datetime.date.today()
    alerts: list[dict[str, Any]] = []

    if not sessions:
        alerts.append(
            {
                "time": "—",
                "type": "No Sessions",
                "message": "No chair sensor sessions have been recorded yet.",
                "severity": "low",
                "source": "Chair",
            }
        )
    else:
        latest = sessions[-1]
        latest_day = session_date(latest)
        day_text = latest_day.isoformat() if latest_day else "Latest session"
        source = "Chair"

        flagged = int(latest.get("flagged_repetitions") or 0)
        total = int(latest.get("total_repetitions") or 0)
        if flagged and total:
            alerts.append(
                {
                    "time": day_text,
                    "type": "Rep Quality Flags",
                    "message": (
                        f"{flagged}/{total} repetitions flagged in the latest "
                        "session. Review hold time, lift angle and smoothness."
                    ),
                    "severity": "high" if flagged / total >= 0.5 else "medium",
                    "source": source,
                }
            )

        if jerk_trend(sessions) == "declining":
            alerts.append(
                {
                    "time": day_text,
                    "type": "Smoothness Declining",
                    "message": (
                        "Movement smoothness (jerk score) is worse than in "
                        "recent sessions."
                    ),
                    "severity": "medium",
                    "source": source,
                }
            )

        if latest_day and (today - latest_day).days > STALE_SESSION_DAYS:
            alerts.append(
                {
                    "time": day_text,
                    "type": "No Recent Session",
                    "message": (
                        f"No session recorded in the last "
                        f"{(today - latest_day).days} days."
                    ),
                    "severity": "low",
                    "source": source,
                }
            )

    if glove_sessions:
        latest_glove = glove_sessions[-1]
        severity = latest_glove["avg_severity"]
        if severity >= 2.5:
            alerts.append(
                {
                    "time": "Session",
                    "type": "Tremor Spike",
                    "message": (
                        f"Latest tremor severity {severity}/4 "
                        f"(session: {latest_glove['test_name']}). "
                        "Discuss with the care team."
                    ),
                    "severity": "high",
                    "source": "Glove",
                }
            )
        elif severity >= 1.5:
            alerts.append(
                {
                    "time": "Session",
                    "type": "Tremor Elevated",
                    "message": f"Tremor severity {severity}/4 -- monitor trend.",
                    "severity": "medium",
                    "source": "Glove",
                }
            )

    return alerts


def patient_status(
    sessions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    message_count: int,
) -> dict[str, Any]:
    """At-a-glance status row used by caregiver and clinician patient lists."""

    latest = sessions[-1] if sessions else {}
    latest_day = session_date(latest) if latest else None
    flagged = int(latest.get("flagged_repetitions") or 0)
    total = int(latest.get("total_repetitions") or 0)

    high = sum(1 for alert in alerts if alert["severity"] == "high")
    medium = sum(1 for alert in alerts if alert["severity"] == "medium")

    return {
        "last_session": latest_day.isoformat() if latest_day else None,
        "flagged": flagged,
        "total": total,
        "trend": jerk_trend(sessions),
        "high_alerts": high,
        "medium_alerts": medium,
        "message_count": message_count,
        # Higher sorts first: high alerts, then medium, then flag ratio.
        "attention_score": (
            high * 100
            + medium * 10
            + (flagged / total if total else 0)
        ),
    }


def laterality_rows(
    sessions: list[dict[str, Any]],
    metric: str = "shin_lift_angle_deg",
) -> list[dict[str, Any]]:
    """Left/right per-session rows in the shape ui_enhancements.render_chart expects."""

    rows = []
    for session in sessions:
        day = session_date(session)
        by_leg = session.get("metric_means_by_leg") or {}
        row: dict[str, Any] = {"session": day.isoformat() if day else None}
        for leg in ("Left", "Right"):
            value = (by_leg.get(leg) or {}).get(metric)
            if value is not None:
                row[leg] = value
        if row["session"] and len(row) > 1:
            rows.append(row)
    return rows


def filter_history_by_dates(
    history: pd.DataFrame,
    start: datetime.date | None,
    end: datetime.date | None,
) -> pd.DataFrame:
    """Keep dated history rows inside [start, end]; undated rows are kept."""

    if history.empty or (start is None and end is None):
        return history

    days = history["Day"]
    if not all(isinstance(value, datetime.date) for value in days):
        return history

    mask = pd.Series(True, index=history.index)
    if start is not None:
        mask &= days >= start
    if end is not None:
        mask &= days <= end
    return history[mask]


def _pdf_text(value: Any) -> str:
    """Core PDF fonts are Latin-1 only; replace anything else."""

    text = str(value)
    for original, replacement in (
        ("—", "-"), ("–", "-"), ("→", "->"), ("↑", "up"), ("↓", "down"),
        ("°", " deg"), ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
    ):
        text = text.replace(original, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_report_pdf(
    *,
    patient_label: str,
    start: datetime.date,
    end: datetime.date,
    exercise_rows: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
) -> bytes:
    """
    A short PDF report: per-exercise summary plus the sessions it used.

    exercise_rows: {"exercise", "latest", "metric", "target", "status"}. sessions: recorded session records within the period.
    """

    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _pdf_text(f"Mobility Report - {patient_label}"), ln=1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, _pdf_text(f"Period: {start.isoformat()} to {end.isoformat()}"), ln=1)
    pdf.cell(0, 6, _pdf_text(f"Generated: {datetime.date.today().isoformat()}"), ln=1)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Exercise summary", ln=1)
    widths = (76, 34, 34, 36)
    pdf.set_font("Arial", "B", 9)
    for width, heading in zip(widths, ("Exercise", "Latest", "Target", "Status")):
        pdf.cell(width, 7, heading, border=1)
    pdf.ln()
    pdf.set_font("Arial", "", 9)
    for row in exercise_rows:
        values = (
            row["exercise"][:46],
            f"{row['latest']}" if row["latest"] is not None else "-",
            f"{row['target']}",
            row["status"],
        )
        for width, value in zip(widths, values):
            pdf.cell(width, 7, _pdf_text(value), border=1)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Where this report comes from", ln=1)
    pdf.set_font("Arial", "", 9)
    if sessions:
        for session in sessions:
            day = session_date(session)
            pdf.multi_cell(
                0,
                5,
                _pdf_text(
                    f"- {day.isoformat() if day else 'undated'}: session "
                    f"{session.get('session_id')}, "
                    f"{session.get('total_repetitions', 0)} reps, "
                    f"{session.get('flagged_repetitions', 0)} flagged"
                ),
            )
    else:
        pdf.multi_cell(0, 5, "No recorded sensor sessions in this period.")
    pdf.ln(2)
    pdf.multi_cell(
        0,
        5,
        _pdf_text(
            "This report describes recorded data and does not provide a diagnosis."
        ),
    )

    output = pdf.output(dest="S")
    return output.encode("latin-1") if isinstance(output, str) else bytes(output)
