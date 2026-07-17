from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPS_CSV = (
    PROJECT_ROOT
    / "data"
    / "mobility"
    / "processed"
    / "session_20260314_131518_reps.csv"
)

DEFAULT_FRAMES_CSV = (
    PROJECT_ROOT
    / "data"
    / "mobility"
    / "processed"
    / "session_20260314_131518_frames.csv"
)

DEFAULT_POSE_CSV = (
    PROJECT_ROOT
    / "data"
    / "mobility"
    / "pose"
    / "session_20260314_131518_pose.csv"
)

DEFAULT_SESSION_JSON = (
    PROJECT_ROOT
    / "data"
    / "mobility"
    / "raw"
    / "session_20260314_131518.json"
)

DEFAULT_TREMOR_CSV = (
    PROJECT_ROOT
    / "data"
    / "tremor"
    / "tremor_validation_master.csv"
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _to_python(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _numeric_stats(df: pd.DataFrame, column: str) -> dict[str, float] | None:
    if column not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None

    return {
        "mean": round(float(values.mean()), 3),
        "minimum": round(float(values.min()), 3),
        "maximum": round(float(values.max()), 3),
        "latest": round(float(values.iloc[-1]), 3),
    }


class MobilityRetriever:
    """Retrieve structured mobility and tremor information from project files."""

    def __init__(
        self,
        *,
        reps_csv: str | Path | None = None,
        frames_csv: str | Path | None = None,
        pose_csv: str | Path | None = None,
        session_json: str | Path | None = None,
        tremor_csv: str | Path | None = None,
    ) -> None:
        self.reps_csv = Path(
            reps_csv or os.getenv("MOBILITY_REPS_CSV", str(DEFAULT_REPS_CSV))
        )
        self.frames_csv = Path(
            frames_csv or os.getenv("MOBILITY_FRAMES_CSV", str(DEFAULT_FRAMES_CSV))
        )
        self.pose_csv = Path(
            pose_csv or os.getenv("MOBILITY_POSE_CSV", str(DEFAULT_POSE_CSV))
        )
        self.session_json = Path(
            session_json or os.getenv("MOBILITY_SESSION_JSON", str(DEFAULT_SESSION_JSON))
        )
        self.tremor_csv = Path(
            tremor_csv or os.getenv("TREMOR_CSV", str(DEFAULT_TREMOR_CSV))
        )

    def get_session_summary(
        self,
        *,
        patient_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        reps = _read_csv(self.reps_csv)
        raw_session = _read_json(self.session_json)

        if not reps.empty and session_id and "session_id" in reps.columns:
            selected = reps[reps["session_id"].astype(str) == str(session_id)]
            if not selected.empty:
                reps = selected

        if reps.empty and not raw_session:
            return {
                "available": False,
                "patient_id": patient_id,
                "message": "No mobility session data was found.",
                "checked_paths": [str(self.reps_csv), str(self.session_json)],
            }

        result: dict[str, Any] = {
            "available": True,
            "patient_id": patient_id,
            "session_id": session_id,
        }

        if raw_session:
            result["session_id"] = result["session_id"] or raw_session.get("session_id")
            result["start_time"] = raw_session.get("start_time")
            result["sample_rate_hz"] = raw_session.get("sample_rate_hz")
            result["total_frames"] = raw_session.get("total_frames") or len(
                raw_session.get("frames", [])
            )

        if not reps.empty:
            if result["session_id"] is None and "session_id" in reps.columns:
                result["session_id"] = str(reps["session_id"].iloc[0])

            result["total_repetitions"] = int(len(reps))

            if "flagged" in reps.columns:
                flags = _to_bool(reps["flagged"])
                result["flagged_repetitions"] = int(flags.sum())
                result["unflagged_repetitions"] = int((~flags).sum())
            else:
                result["flagged_repetitions"] = 0

            if "leg" in reps.columns:
                result["repetitions_by_leg"] = {
                    str(key): int(value)
                    for key, value in reps["leg"]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .items()
                }

            metrics = {}
            for column in (
                "duration_s",
                "lift_duration_s",
                "hold_duration_s",
                "lower_duration_s",
                "shin_lift_angle_deg",
                "peak_angular_velocity",
                "jerk_score",
                "tof_symmetry",
            ):
                stats = _numeric_stats(reps, column)
                if stats:
                    metrics[column] = stats

            result["metrics"] = metrics

        return result

    def get_flagged_repetitions(
        self,
        *,
        patient_id: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        reps = _read_csv(self.reps_csv)

        if reps.empty:
            return {
                "available": False,
                "patient_id": patient_id,
                "flagged_repetitions": [],
                "message": "No repetition data was found.",
            }

        if session_id and "session_id" in reps.columns:
            selected = reps[reps["session_id"].astype(str) == str(session_id)]
            if not selected.empty:
                reps = selected

        if "flagged" in reps.columns:
            mask = _to_bool(reps["flagged"])
        else:
            mask = pd.Series(False, index=reps.index)
            for flag_column in ("flag_hold", "flag_angle", "flag_jerk"):
                if flag_column in reps.columns:
                    mask = mask | _to_bool(reps[flag_column])

        flagged = reps[mask].head(max(limit, 0))
        records: list[dict[str, Any]] = []

        for _, row in flagged.iterrows():
            reasons = []
            mappings = {
                "flag_hold": "hold duration",
                "flag_angle": "lift angle",
                "flag_jerk": "movement smoothness",
            }

            for column, label in mappings.items():
                if column in row.index and str(row[column]).strip().lower() in {
                    "true",
                    "1",
                    "yes",
                }:
                    reasons.append(label)

            records.append(
                {
                    "rep_id": _to_python(row.get("rep_id")),
                    "session_id": _to_python(row.get("session_id")),
                    "leg": _to_python(row.get("leg")),
                    "reasons": reasons,
                    "duration_s": _to_python(row.get("duration_s")),
                    "hold_duration_s": _to_python(row.get("hold_duration_s")),
                    "shin_lift_angle_deg": _to_python(
                        row.get("shin_lift_angle_deg")
                    ),
                    "peak_angular_velocity": _to_python(
                        row.get("peak_angular_velocity")
                    ),
                    "jerk_score": _to_python(row.get("jerk_score")),
                    "tof_symmetry": _to_python(row.get("tof_symmetry")),
                }
            )

        return {
            "available": True,
            "patient_id": patient_id,
            "total_flagged": int(mask.sum()),
            "returned_flagged": len(records),
            "flagged_repetitions": records,
        }

    def get_repetition(
        self,
        *,
        patient_id: str,
        rep_id: int,
    ) -> dict[str, Any]:
        reps = _read_csv(self.reps_csv)

        if reps.empty or "rep_id" not in reps.columns:
            return {
                "available": False,
                "patient_id": patient_id,
                "rep_id": rep_id,
                "message": "Repetition data was not found.",
            }

        rep_ids = pd.to_numeric(reps["rep_id"], errors="coerce")
        selected = reps[rep_ids == rep_id]

        if selected.empty:
            return {
                "available": False,
                "patient_id": patient_id,
                "rep_id": rep_id,
                "message": f"Repetition {rep_id} was not found.",
            }

        row = selected.iloc[0]
        return {
            "available": True,
            "patient_id": patient_id,
            "repetition": {
                str(column): _to_python(value)
                for column, value in row.items()
            },
        }

    def get_pose_summary(self) -> dict[str, Any]:
        pose = _read_csv(self.pose_csv)

        if pose.empty:
            return {"available": False, "message": "No pose data was found."}

        metrics = {}
        for column in (
            "knee_angle",
            "shin_angle_vertical",
            "trunk_lean",
            "ankle_disp_px",
            "near_visibility",
        ):
            stats = _numeric_stats(pose, column)
            if stats:
                metrics[column] = stats

        return {
            "available": True,
            "rows": int(len(pose)),
            "metrics": metrics,
        }

    def get_tremor_summary(self, *, patient_id: str) -> dict[str, Any]:
        tremor = _read_csv(self.tremor_csv)

        if tremor.empty:
            return {
                "available": False,
                "patient_id": patient_id,
                "message": "No tremor validation data was found.",
            }

        patient_column = next(
            (
                column
                for column in tremor.columns
                if column.lower() in {"patient_id", "participant_id", "subject_id"}
            ),
            None,
        )

        if patient_column:
            selected = tremor[
                tremor[patient_column].astype(str) == str(patient_id)
            ]
            if not selected.empty:
                tremor = selected

        numeric_columns = tremor.select_dtypes(include="number").columns[:12]
        metrics = {
            column: stats
            for column in numeric_columns
            if (stats := _numeric_stats(tremor, column))
        }

        return {
            "available": True,
            "patient_id": patient_id,
            "rows": int(len(tremor)),
            "metrics": metrics,
        }

    def retrieve_for_intent(
        self,
        *,
        intent: str,
        patient_id: str,
        session_id: str | None = None,
        rep_id: int | None = None,
    ) -> dict[str, Any]:
        if intent == "session_summary":
            return self.get_session_summary(
                patient_id=patient_id,
                session_id=session_id,
            )

        if intent == "flag_explanation":
            if rep_id is not None:
                return self.get_repetition(patient_id=patient_id, rep_id=rep_id)
            return self.get_flagged_repetitions(
                patient_id=patient_id,
                session_id=session_id,
            )

        if intent == "trend_analysis":
            return {
                "available": False,
                "patient_id": patient_id,
                "message": (
                    "Longitudinal trend analysis requires multiple dated sessions. "
                    "The current file retriever exposes one session unless additional "
                    "session files or a database adapter are configured."
                ),
                "latest_session": self.get_session_summary(
                    patient_id=patient_id,
                    session_id=session_id,
                ),
            }

        if intent == "metric_explanation":
            return {
                "available": True,
                "patient_id": patient_id,
                "session": self.get_session_summary(
                    patient_id=patient_id,
                    session_id=session_id,
                ),
                "pose": self.get_pose_summary(),
                "tremor": self.get_tremor_summary(patient_id=patient_id),
            }

        return {
            "available": False,
            "patient_id": patient_id,
            "message": "No structured mobility retrieval was required.",
        }
