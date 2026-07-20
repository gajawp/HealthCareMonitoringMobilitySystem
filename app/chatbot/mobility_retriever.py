from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


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

DEFAULT_SYNTHETIC_DIR = PROJECT_ROOT / "synthetic_data"

DEFAULT_FEEDBACK_JSON = (
    PROJECT_ROOT
    / "data"
    / "feedback"
    / "feedback.json"
)

DEFAULT_FEEDBACK_CSV = (
    PROJECT_ROOT
    / "data"
    / "feedback"
    / "feedback.csv"
)

# Demo records mirror the feedback currently shown for the demo patient.
# They are used only when no feedback file is available. Set
# MOBILITY_ENABLE_DEMO_FEEDBACK=false to disable this fallback.
_DEMO_FEEDBACK: tuple[dict[str, Any], ...] = (
    {
        "patient_id": "P1001",
        "feedback_type": "doctor",
        "author_name": "Dr. Sarah Lee",
        "feedback_date": "2026-02-24",
        "message": (
            "Tremor severity levels look stable this week. "
            "Keep up the exercises."
        ),
    },
    {
        "patient_id": "P1001",
        "feedback_type": "doctor",
        "author_name": "Dr. Michael Chen",
        "feedback_date": "2026-02-22",
        "message": (
            "Consider adjusting sitting breaks — prolonged sitting "
            "detected by chair sensors."
        ),
    },
    {
        "patient_id": "P1001",
        "feedback_type": "caregiver",
        "author_name": "Alice Martin",
        "feedback_date": "2026-02-23",
        "message": "Your tap rate has improved — keep it up!",
    },
    {
        "patient_id": "P1001",
        "feedback_type": "caregiver",
        "author_name": "Bob Taylor",
        "feedback_date": "2026-02-21",
        "message": "Remember to do your stretches after long sitting periods.",
    },
)

_REPS_FILENAME_PATTERN = re.compile(
    r"^session_"
    r"(?P<date>\d{8})_"
    r"(?P<time>\d{6})_"
    r"(?P<patient_id>[^_]+)_reps\.csv$",
    re.IGNORECASE,
)

_REAL_REPS_FILENAME_PATTERN = re.compile(
    r"^session_(?P<date>\d{8})_(?P<time>\d{6})_reps\.csv$",
    re.IGNORECASE,
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
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
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

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


def _normalise_identifier(value: Any) -> str:
    return str(value).strip().casefold()


def _session_datetime_from_parts(date_value: str, time_value: str) -> pd.Timestamp:
    return pd.to_datetime(
        f"{date_value}{time_value}",
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


def _derive_real_session_metadata(path: Path) -> tuple[str | None, pd.Timestamp]:
    match = _REAL_REPS_FILENAME_PATTERN.match(path.name)
    if not match:
        return None, pd.NaT

    date_value = match.group("date")
    time_value = match.group("time")
    session_id = f"{date_value}_{time_value}"
    session_datetime = _session_datetime_from_parts(date_value, time_value)
    return session_id, session_datetime


class MobilityRetriever:
    """
    Retrieve structured real and synthetic mobility information.

    Real repetition data is loaded from the existing processed repetition CSV.
    Synthetic repetition data is loaded from every matching CSV in
    ``synthetic_data``. Data is always filtered by patient before a session is
    selected, preventing records from different patients from being mixed.
    """

    def __init__(
        self,
        *,
        reps_csv: str | Path | None = None,
        frames_csv: str | Path | None = None,
        pose_csv: str | Path | None = None,
        session_json: str | Path | None = None,
        tremor_csv: str | Path | None = None,
        synthetic_dir: str | Path | None = None,
        feedback_json: str | Path | None = None,
        feedback_csv: str | Path | None = None,
        real_patient_id: str | None = None,
        enable_demo_feedback: bool | None = None,
    ) -> None:
        self._explicit_reps_csv = reps_csv is not None

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
        configured_synthetic_dir = synthetic_dir or os.getenv(
            "MOBILITY_SYNTHETIC_DIR"
        )

        # Unit tests and isolated callers that explicitly supply a repetition
        # CSV should use only that file unless they also explicitly supply a
        # synthetic directory. Normal application construction still loads the
        # project-level synthetic_data directory.
        if configured_synthetic_dir is not None:
            self.synthetic_dir: Path | None = Path(configured_synthetic_dir)
        elif self._explicit_reps_csv:
            self.synthetic_dir = None
        else:
            self.synthetic_dir = DEFAULT_SYNTHETIC_DIR

        self.feedback_json = Path(
            feedback_json
            or os.getenv("MOBILITY_FEEDBACK_JSON", str(DEFAULT_FEEDBACK_JSON))
        )
        self.feedback_csv = Path(
            feedback_csv
            or os.getenv("MOBILITY_FEEDBACK_CSV", str(DEFAULT_FEEDBACK_CSV))
        )

        if enable_demo_feedback is None:
            demo_setting = os.getenv(
                "MOBILITY_ENABLE_DEMO_FEEDBACK",
                "true",
            )
            self.enable_demo_feedback = (
                str(demo_setting).strip().casefold()
                not in {"false", "0", "no", "off"}
            )
        else:
            self.enable_demo_feedback = bool(enable_demo_feedback)

        self.real_patient_id = str(
            real_patient_id
            or os.getenv("MOBILITY_REAL_PATIENT_ID")
            or "REAL_PATIENT"
        ).strip()

    def _load_real_repetitions(self) -> pd.DataFrame:
        reps = _read_csv(self.reps_csv)
        if reps.empty:
            return reps

        reps = reps.copy()
        raw_session = _read_json(self.session_json)

        if "patient_id" not in reps.columns:
            # An explicitly injected CSV is commonly used by unit tests or
            # adapters and historically applied to the patient requested by
            # the caller. Mark it as a wildcard rather than forcing the
            # production REAL_PATIENT identifier.
            assigned_patient_id = (
                "*" if self._explicit_reps_csv else self.real_patient_id
            )
            reps.insert(0, "patient_id", assigned_patient_id)
        else:
            missing_patient = reps["patient_id"].isna() | (
                reps["patient_id"].astype(str).str.strip() == ""
            )
            reps.loc[missing_patient, "patient_id"] = self.real_patient_id

        derived_session_id, derived_datetime = _derive_real_session_metadata(
            self.reps_csv
        )
        raw_session_id = raw_session.get("session_id")

        if "session_id" not in reps.columns:
            reps["session_id"] = raw_session_id or derived_session_id
        else:
            missing_session = reps["session_id"].isna() | (
                reps["session_id"].astype(str).str.strip() == ""
            )
            reps.loc[missing_session, "session_id"] = (
                raw_session_id or derived_session_id
            )

        reps["data_source"] = "real"
        reps["source_file"] = self.reps_csv.name

        if "session_datetime" not in reps.columns:
            reps["session_datetime"] = derived_datetime
        else:
            reps["session_datetime"] = pd.to_datetime(
                reps["session_datetime"],
                errors="coerce",
            )
            reps["session_datetime"] = reps["session_datetime"].fillna(
                derived_datetime
            )

        return reps

    def _load_synthetic_repetitions(self) -> pd.DataFrame:
        if (
            self.synthetic_dir is None
            or not self.synthetic_dir.exists()
            or not self.synthetic_dir.is_dir()
        ):
            return pd.DataFrame()

        dataframes: list[pd.DataFrame] = []

        for csv_path in sorted(self.synthetic_dir.glob("*_reps.csv")):
            filename_match = _REPS_FILENAME_PATTERN.match(csv_path.name)
            if not filename_match:
                continue

            reps = _read_csv(csv_path)
            if reps.empty:
                continue

            reps = reps.copy()
            patient_id = filename_match.group("patient_id")
            date_value = filename_match.group("date")
            time_value = filename_match.group("time")
            session_id = f"{date_value}_{time_value}"
            session_datetime = _session_datetime_from_parts(date_value, time_value)

            if "patient_id" not in reps.columns:
                reps.insert(0, "patient_id", patient_id)
            else:
                missing_patient = reps["patient_id"].isna() | (
                    reps["patient_id"].astype(str).str.strip() == ""
                )
                reps.loc[missing_patient, "patient_id"] = patient_id

            if "session_id" not in reps.columns:
                reps["session_id"] = session_id
            else:
                missing_session = reps["session_id"].isna() | (
                    reps["session_id"].astype(str).str.strip() == ""
                )
                reps.loc[missing_session, "session_id"] = session_id

            reps["session_date"] = date_value
            reps["session_time"] = time_value
            reps["session_datetime"] = session_datetime
            reps["data_source"] = "synthetic"
            reps["source_file"] = csv_path.name
            dataframes.append(reps)

        if not dataframes:
            return pd.DataFrame()

        return pd.concat(dataframes, ignore_index=True, sort=False)

    def _load_all_repetitions(self) -> pd.DataFrame:
        datasets = [
            dataframe
            for dataframe in (
                self._load_real_repetitions(),
                self._load_synthetic_repetitions(),
            )
            if not dataframe.empty
        ]

        if not datasets:
            return pd.DataFrame()

        combined = pd.concat(datasets, ignore_index=True, sort=False)

        if "session_datetime" in combined.columns:
            combined["session_datetime"] = pd.to_datetime(
                combined["session_datetime"],
                errors="coerce",
            )
            combined = combined.sort_values(
                by=["session_datetime", "rep_id"]
                if "rep_id" in combined.columns
                else ["session_datetime"],
                kind="stable",
                na_position="first",
            )

        return combined.reset_index(drop=True)

    def _select_patient_rows(
        self,
        reps: pd.DataFrame,
        *,
        patient_id: str,
    ) -> pd.DataFrame:
        if reps.empty or "patient_id" not in reps.columns:
            return pd.DataFrame()

        requested_patient = _normalise_identifier(patient_id)
        patient_values = reps["patient_id"].map(_normalise_identifier)

        # "*" is used only for explicitly injected repetition CSVs so legacy
        # isolated tests and adapters remain patient-agnostic.
        return reps[
            (patient_values == requested_patient) | (patient_values == "*")
        ].copy()

    def _select_session_rows(
        self,
        patient_reps: pd.DataFrame,
        *,
        session_id: str | None,
        latest_if_missing: bool = True,
    ) -> pd.DataFrame:
        if patient_reps.empty:
            return patient_reps

        selected = patient_reps.copy()

        if session_id is not None and "session_id" in selected.columns:
            requested_session = _normalise_identifier(session_id)
            session_values = selected["session_id"].map(_normalise_identifier)
            return selected[session_values == requested_session].copy()

        if not latest_if_missing or "session_id" not in selected.columns:
            return selected

        valid_sessions = selected[
            selected["session_id"].notna()
            & (selected["session_id"].astype(str).str.strip() != "")
        ].copy()

        if valid_sessions.empty:
            return selected

        if (
            "session_datetime" in valid_sessions.columns
            and valid_sessions["session_datetime"].notna().any()
        ):
            latest_datetime = valid_sessions["session_datetime"].max()
            latest_rows = valid_sessions[
                valid_sessions["session_datetime"] == latest_datetime
            ]
            if not latest_rows.empty:
                latest_session_id = str(latest_rows["session_id"].iloc[-1])
            else:
                latest_session_id = str(valid_sessions["session_id"].iloc[-1])
        else:
            latest_session_id = sorted(
                valid_sessions["session_id"].astype(str).unique()
            )[-1]

        return valid_sessions[
            valid_sessions["session_id"].astype(str) == latest_session_id
        ].copy()

    def _available_patient_ids(self, reps: pd.DataFrame) -> list[str]:
        if reps.empty or "patient_id" not in reps.columns:
            return []

        return sorted(
            {
                str(value).strip()
                for value in reps["patient_id"].dropna()
                if str(value).strip()
            }
        )

    def _filter_by_date_range(
        self,
        reps: pd.DataFrame,
        *,
        session_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Filter patient-scoped rows by one date or an inclusive date range.

        Real and synthetic rows are treated as one mobility history. The
        ``data_source`` field remains traceability metadata and is never used
        as a filtering condition.
        """

        if reps.empty:
            return reps

        selected = reps.copy()

        if "session_datetime" not in selected.columns:
            return pd.DataFrame()

        selected["session_datetime"] = pd.to_datetime(
            selected["session_datetime"],
            errors="coerce",
        )
        selected = selected[selected["session_datetime"].notna()].copy()

        if selected.empty:
            return selected

        if session_date:
            parsed_date = pd.to_datetime(
                session_date,
                errors="coerce",
            )
            if pd.isna(parsed_date):
                return pd.DataFrame()

            target_date = parsed_date.date()
            return selected[
                selected["session_datetime"].dt.date == target_date
            ].copy()

        parsed_start = pd.to_datetime(start_date, errors="coerce") if start_date else None
        parsed_end = pd.to_datetime(end_date, errors="coerce") if end_date else None

        if parsed_start is not None and pd.isna(parsed_start):
            return pd.DataFrame()
        if parsed_end is not None and pd.isna(parsed_end):
            return pd.DataFrame()

        if parsed_start is not None:
            selected = selected[
                selected["session_datetime"].dt.date >= parsed_start.date()
            ]

        if parsed_end is not None:
            selected = selected[
                selected["session_datetime"].dt.date <= parsed_end.date()
            ]

        return selected.copy()

    def get_sessions_by_date_range(
        self,
        *,
        patient_id: str,
        session_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Return sessions for one date or an inclusive date range.

        The result combines real and synthetic sessions for the selected
        patient. Source type is retained only as internal traceability
        metadata.
        """

        all_reps = self._load_all_repetitions()
        patient_reps = self._select_patient_rows(
            all_reps,
            patient_id=patient_id,
        )
        matching_reps = self._filter_by_date_range(
            patient_reps,
            session_date=session_date,
            start_date=start_date,
            end_date=end_date,
        )

        if matching_reps.empty:
            return {
                "available": True,
                "patient_id": patient_id,
                "session_date": session_date,
                "start_date": start_date,
                "end_date": end_date,
                "session_count": 0,
                "total_repetitions": 0,
                "sessions": [],
                "message": "No mobility sessions were found for the requested date range.",
            }

        sessions: list[dict[str, Any]] = []

        for grouped_session_id, session_rows in matching_reps.groupby(
            "session_id",
            sort=False,
            dropna=False,
        ):
            record: dict[str, Any] = {
                "session_id": _to_python(grouped_session_id),
                "total_repetitions": int(len(session_rows)),
            }

            if "session_datetime" in session_rows.columns:
                datetimes = pd.to_datetime(
                    session_rows["session_datetime"],
                    errors="coerce",
                ).dropna()
                if not datetimes.empty:
                    record["session_datetime"] = datetimes.iloc[0].isoformat()

            if "data_source" in session_rows.columns:
                sources = sorted(
                    {
                        str(value)
                        for value in session_rows["data_source"].dropna()
                    }
                )
                record["data_sources"] = sources

            if "flagged" in session_rows.columns:
                flags = _to_bool(session_rows["flagged"])
            else:
                flags = pd.Series(False, index=session_rows.index)
                for flag_column in ("flag_hold", "flag_angle", "flag_jerk"):
                    if flag_column in session_rows.columns:
                        flags = flags | _to_bool(session_rows[flag_column])

            record["flagged_repetitions"] = int(flags.sum())

            metric_means: dict[str, float] = {}
            for column in (
                "duration_s",
                "hold_duration_s",
                "shin_lift_angle_deg",
                "peak_angular_velocity",
                "jerk_score",
                "tof_symmetry",
            ):
                if column not in session_rows.columns:
                    continue
                values = pd.to_numeric(
                    session_rows[column],
                    errors="coerce",
                ).dropna()
                if not values.empty:
                    metric_means[column] = round(float(values.mean()), 3)

            record["metric_means"] = metric_means
            sessions.append(record)

        sessions.sort(
            key=lambda item: item.get("session_datetime", "")
        )

        if limit > 0:
            sessions = sessions[:limit]

        return {
            "available": True,
            "patient_id": patient_id,
            "session_date": session_date,
            "start_date": start_date,
            "end_date": end_date,
            "session_count": len(sessions),
            "total_repetitions": int(
                sum(item["total_repetitions"] for item in sessions)
            ),
            "sessions": sessions,
        }

    def get_session_summary(
        self,
        *,
        patient_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        all_reps = self._load_all_repetitions()
        patient_reps = self._select_patient_rows(
            all_reps,
            patient_id=patient_id,
        )
        reps = self._select_session_rows(
            patient_reps,
            session_id=session_id,
            latest_if_missing=True,
        )

        if reps.empty:
            return {
                "available": False,
                "patient_id": patient_id,
                "session_id": session_id,
                "message": (
                    "No mobility session data was found for this patient"
                    + (
                        f" and session {session_id}."
                        if session_id is not None
                        else "."
                    )
                ),
                "available_patient_ids": self._available_patient_ids(all_reps),
                "checked_paths": [
                    str(self.reps_csv),
                    str(self.synthetic_dir) if self.synthetic_dir else "disabled",
                    str(self.session_json),
                ],
            }

        resolved_session_id = (
            _to_python(reps["session_id"].iloc[0])
            if "session_id" in reps.columns
            else session_id
        )
        data_source = (
            str(reps["data_source"].iloc[0])
            if "data_source" in reps.columns
            else "unknown"
        )

        result: dict[str, Any] = {
            "available": True,
            "patient_id": patient_id,
            "session_id": resolved_session_id,
            "data_source": data_source,
            "total_repetitions": int(len(reps)),
        }

        if "source_file" in reps.columns:
            result["source_file"] = str(reps["source_file"].iloc[0])

        if "session_datetime" in reps.columns:
            session_datetimes = pd.to_datetime(
                reps["session_datetime"],
                errors="coerce",
            ).dropna()
            if not session_datetimes.empty:
                result["session_datetime"] = session_datetimes.iloc[0].isoformat()

        # The raw JSON describes the current real session only. Do not attach it
        # to a synthetic session.
        if data_source == "real":
            raw_session = _read_json(self.session_json)
            if raw_session:
                result["start_time"] = raw_session.get("start_time")
                result["sample_rate_hz"] = raw_session.get("sample_rate_hz")
                result["total_frames"] = raw_session.get("total_frames") or len(
                    raw_session.get("frames", [])
                )

        if "flagged" in reps.columns:
            flags = _to_bool(reps["flagged"])
            result["flagged_repetitions"] = int(flags.sum())
            result["unflagged_repetitions"] = int((~flags).sum())
        else:
            fallback_flags = pd.Series(False, index=reps.index)
            for flag_column in ("flag_hold", "flag_angle", "flag_jerk"):
                if flag_column in reps.columns:
                    fallback_flags = fallback_flags | _to_bool(reps[flag_column])
            result["flagged_repetitions"] = int(fallback_flags.sum())
            result["unflagged_repetitions"] = int((~fallback_flags).sum())

        if "leg" in reps.columns:
            result["repetitions_by_leg"] = {
                str(key): int(value)
                for key, value in reps["leg"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .items()
            }

        metrics: dict[str, dict[str, float]] = {}
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
        all_reps = self._load_all_repetitions()
        patient_reps = self._select_patient_rows(
            all_reps,
            patient_id=patient_id,
        )
        reps = self._select_session_rows(
            patient_reps,
            session_id=session_id,
            latest_if_missing=True,
        )

        if reps.empty:
            return {
                "available": False,
                "patient_id": patient_id,
                "session_id": session_id,
                "flagged_repetitions": [],
                "message": "No repetition data was found for this patient/session.",
                "available_patient_ids": self._available_patient_ids(all_reps),
            }

        if "flagged" in reps.columns:
            mask = _to_bool(reps["flagged"])
        else:
            mask = pd.Series(False, index=reps.index)
            for flag_column in ("flag_hold", "flag_angle", "flag_jerk"):
                if flag_column in reps.columns:
                    mask = mask | _to_bool(reps[flag_column])

        flagged = reps[mask].head(max(int(limit), 0))
        records: list[dict[str, Any]] = []

        for _, row in flagged.iterrows():
            reasons: list[str] = []
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
                    "y",
                }:
                    reasons.append(label)

            records.append(
                {
                    "patient_id": _to_python(row.get("patient_id")),
                    "session_id": _to_python(row.get("session_id")),
                    "rep_id": _to_python(row.get("rep_id")),
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
                    "data_source": _to_python(row.get("data_source")),
                    "source_file": _to_python(row.get("source_file")),
                }
            )

        return {
            "available": True,
            "patient_id": patient_id,
            "session_id": (
                _to_python(reps["session_id"].iloc[0])
                if "session_id" in reps.columns
                else session_id
            ),
            "data_source": (
                _to_python(reps["data_source"].iloc[0])
                if "data_source" in reps.columns
                else None
            ),
            "total_flagged": int(mask.sum()),
            "returned_flagged": len(records),
            "flagged_repetitions": records,
        }

    def get_repetition(
        self,
        *,
        patient_id: str,
        rep_id: int,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        all_reps = self._load_all_repetitions()
        patient_reps = self._select_patient_rows(
            all_reps,
            patient_id=patient_id,
        )
        reps = self._select_session_rows(
            patient_reps,
            session_id=session_id,
            latest_if_missing=True,
        )

        if reps.empty or "rep_id" not in reps.columns:
            return {
                "available": False,
                "patient_id": patient_id,
                "session_id": session_id,
                "rep_id": rep_id,
                "message": "Repetition data was not found for this patient/session.",
            }

        rep_ids = pd.to_numeric(reps["rep_id"], errors="coerce")
        selected = reps[rep_ids == int(rep_id)]

        if selected.empty:
            return {
                "available": False,
                "patient_id": patient_id,
                "session_id": (
                    _to_python(reps["session_id"].iloc[0])
                    if "session_id" in reps.columns
                    else session_id
                ),
                "rep_id": rep_id,
                "message": f"Repetition {rep_id} was not found.",
            }

        row = selected.iloc[0]
        return {
            "available": True,
            "patient_id": patient_id,
            "session_id": _to_python(row.get("session_id")),
            "data_source": _to_python(row.get("data_source")),
            "repetition": {
                str(column): _to_python(value)
                for column, value in row.items()
                if column != "session_datetime"
            },
        }

    def get_trend_analysis(
        self,
        *,
        patient_id: str,
        session_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        all_reps = self._load_all_repetitions()
        patient_reps = self._select_patient_rows(
            all_reps,
            patient_id=patient_id,
        )
        patient_reps = self._filter_by_date_range(
            patient_reps,
            session_date=session_date,
            start_date=start_date,
            end_date=end_date,
        )

        if patient_reps.empty or "session_id" not in patient_reps.columns:
            return {
                "available": False,
                "patient_id": patient_id,
                "message": "No dated mobility sessions were found for this patient.",
                "available_patient_ids": self._available_patient_ids(all_reps),
            }

        session_records: list[dict[str, Any]] = []

        grouped = patient_reps.groupby("session_id", sort=False, dropna=False)
        for grouped_session_id, session_rows in grouped:
            record: dict[str, Any] = {
                "session_id": _to_python(grouped_session_id),
                "total_repetitions": int(len(session_rows)),
                "data_source": (
                    _to_python(session_rows["data_source"].iloc[0])
                    if "data_source" in session_rows.columns
                    else None
                ),
                "source_file": (
                    _to_python(session_rows["source_file"].iloc[0])
                    if "source_file" in session_rows.columns
                    else None
                ),
            }

            if "session_datetime" in session_rows.columns:
                datetimes = pd.to_datetime(
                    session_rows["session_datetime"],
                    errors="coerce",
                ).dropna()
                if not datetimes.empty:
                    record["session_datetime"] = datetimes.iloc[0].isoformat()
                    record["_sort_datetime"] = datetimes.iloc[0]

            if "flagged" in session_rows.columns:
                record["flagged_repetitions"] = int(
                    _to_bool(session_rows["flagged"]).sum()
                )
            else:
                fallback_flags = pd.Series(False, index=session_rows.index)
                for flag_column in ("flag_hold", "flag_angle", "flag_jerk"):
                    if flag_column in session_rows.columns:
                        fallback_flags = fallback_flags | _to_bool(
                            session_rows[flag_column]
                        )
                record["flagged_repetitions"] = int(fallback_flags.sum())

            metric_means: dict[str, float] = {}
            for column in (
                "duration_s",
                "hold_duration_s",
                "shin_lift_angle_deg",
                "peak_angular_velocity",
                "jerk_score",
                "tof_symmetry",
            ):
                values = pd.to_numeric(
                    session_rows.get(column),
                    errors="coerce",
                ).dropna() if column in session_rows.columns else pd.Series(dtype=float)

                if not values.empty:
                    metric_means[column] = round(float(values.mean()), 3)

            record["metric_means"] = metric_means
            session_records.append(record)

        session_records.sort(
            key=lambda item: item.get("_sort_datetime", pd.Timestamp.min)
        )

        for record in session_records:
            record.pop("_sort_datetime", None)

        if limit > 0:
            session_records = session_records[-limit:]

        return {
            "available": len(session_records) >= 2,
            "patient_id": patient_id,
            "session_count": len(session_records),
            "message": (
                "Trend data is available."
                if len(session_records) >= 2
                else "At least two sessions are required for longitudinal trends."
            ),
            "sessions": session_records,
        }

    def _load_feedback_records(self) -> pd.DataFrame:
        """
        Load doctor and caregiver feedback.

        Supported JSON formats:
        - a list of feedback records;
        - {"feedback": [...]};
        - {"doctor_feedback": [...], "caregiver_feedback": [...]}.

        Supported CSV columns are normalized from common aliases such as
        patient_id, type/feedback_type, author/author_name, date/feedback_date,
        and message/feedback/comment/text.
        """

        records: list[dict[str, Any]] = []

        raw_json = _read_json(self.feedback_json)
        if raw_json:
            if isinstance(raw_json.get("feedback"), list):
                records.extend(
                    item
                    for item in raw_json["feedback"]
                    if isinstance(item, dict)
                )

            for key, feedback_type in (
                ("doctor_feedback", "doctor"),
                ("caregiver_feedback", "caregiver"),
            ):
                values = raw_json.get(key)
                if not isinstance(values, list):
                    continue

                for item in values:
                    if not isinstance(item, dict):
                        continue
                    normalized_item = dict(item)
                    normalized_item.setdefault(
                        "feedback_type",
                        feedback_type,
                    )
                    records.append(normalized_item)

        json_frame = pd.DataFrame(records)
        csv_frame = _read_csv(self.feedback_csv)

        frames = [
            frame
            for frame in (json_frame, csv_frame)
            if not frame.empty
        ]

        if frames:
            feedback = pd.concat(
                frames,
                ignore_index=True,
                sort=False,
            )
        elif self.enable_demo_feedback:
            feedback = pd.DataFrame(_DEMO_FEEDBACK)
        else:
            return pd.DataFrame()

        alias_groups = {
            "patient_id": (
                "patient_id",
                "patient",
                "patientid",
            ),
            "feedback_type": (
                "feedback_type",
                "type",
                "category",
                "source_type",
                "role",
            ),
            "author_name": (
                "author_name",
                "author",
                "provider_name",
                "doctor_name",
                "caregiver_name",
                "name",
            ),
            "feedback_date": (
                "feedback_date",
                "date",
                "created_at",
                "timestamp",
            ),
            "message": (
                "message",
                "feedback",
                "comment",
                "text",
                "note",
            ),
        }

        normalized_columns = {
            str(column).strip().casefold().replace(" ", "_"): column
            for column in feedback.columns
        }

        selected = pd.DataFrame(index=feedback.index)

        for target, aliases in alias_groups.items():
            source_column = next(
                (
                    normalized_columns[alias]
                    for alias in aliases
                    if alias in normalized_columns
                ),
                None,
            )
            if source_column is not None:
                selected[target] = feedback[source_column]

        required = {
            "patient_id",
            "feedback_type",
            "message",
        }
        if not required.issubset(selected.columns):
            return pd.DataFrame()

        if "author_name" not in selected.columns:
            selected["author_name"] = "Care team"
        if "feedback_date" not in selected.columns:
            selected["feedback_date"] = None

        selected["feedback_type"] = (
            selected["feedback_type"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .replace(
                {
                    "clinician": "doctor",
                    "physician": "doctor",
                    "provider": "doctor",
                    "care giver": "caregiver",
                }
            )
        )
        selected["feedback_date"] = pd.to_datetime(
            selected["feedback_date"],
            errors="coerce",
        )
        selected = selected[
            selected["message"].notna()
            & (selected["message"].astype(str).str.strip() != "")
        ].copy()

        return selected

    def get_feedback(
        self,
        *,
        patient_id: str,
        feedback_type: str = "doctor",
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Return feedback for the selected patient.

        feedback_type may be "doctor", "caregiver", or "all".
        """

        feedback = self._load_feedback_records()
        normalized_type = str(feedback_type).strip().casefold()

        if feedback.empty:
            return {
                "available": False,
                "patient_id": patient_id,
                "feedback_type": normalized_type,
                "feedback": [],
                "message": "No care-team feedback data was found.",
            }

        selected = feedback[
            feedback["patient_id"].map(_normalise_identifier)
            == _normalise_identifier(patient_id)
        ].copy()

        if normalized_type != "all":
            selected = selected[
                selected["feedback_type"] == normalized_type
            ].copy()

        if selected.empty:
            label = (
                "doctor"
                if normalized_type == "doctor"
                else "caregiver"
                if normalized_type == "caregiver"
                else "care-team"
            )
            return {
                "available": True,
                "patient_id": patient_id,
                "feedback_type": normalized_type,
                "feedback_count": 0,
                "feedback": [],
                "message": f"No {label} feedback was found for this patient.",
            }

        selected = selected.sort_values(
            "feedback_date",
            ascending=False,
            na_position="last",
            kind="stable",
        )

        if limit > 0:
            selected = selected.head(int(limit))

        records: list[dict[str, Any]] = []
        for _, row in selected.iterrows():
            feedback_date = row.get("feedback_date")
            records.append(
                {
                    "feedback_type": _to_python(
                        row.get("feedback_type")
                    ),
                    "author_name": _to_python(
                        row.get("author_name")
                    ),
                    "feedback_date": (
                        feedback_date.strftime("%Y-%m-%d")
                        if pd.notna(feedback_date)
                        else None
                    ),
                    "message": _to_python(row.get("message")),
                }
            )

        return {
            "available": True,
            "patient_id": patient_id,
            "feedback_type": normalized_type,
            "feedback_count": len(records),
            "feedback": records,
            "data_source": (
                "feedback_file"
                if self.feedback_json.exists() or self.feedback_csv.exists()
                else "demo_feedback"
            ),
        }

    def get_pose_summary(self) -> dict[str, Any]:
        pose = _read_csv(self.pose_csv)

        if pose.empty:
            return {"available": False, "message": "No pose data was found."}

        metrics: dict[str, dict[str, float]] = {}
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
            "data_source": "real",
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
                tremor[patient_column].map(_normalise_identifier)
                == _normalise_identifier(patient_id)
            ]
            if selected.empty:
                return {
                    "available": False,
                    "patient_id": patient_id,
                    "message": "No tremor data was found for this patient.",
                }
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
        session_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        has_date_filter = bool(session_date or start_date or end_date)

        if intent == "doctor_feedback":
            return self.get_feedback(
                patient_id=patient_id,
                feedback_type="doctor",
            )

        if intent == "caregiver_feedback":
            return self.get_feedback(
                patient_id=patient_id,
                feedback_type="caregiver",
            )

        if intent == "care_team_feedback":
            return self.get_feedback(
                patient_id=patient_id,
                feedback_type="all",
            )

        if intent in {
            "sessions_by_date",
            "sessions_by_date_range",
            "session_count",
        }:
            return self.get_sessions_by_date_range(
                patient_id=patient_id,
                session_date=session_date,
                start_date=start_date,
                end_date=end_date,
            )

        if intent == "session_summary":
            if has_date_filter:
                return self.get_sessions_by_date_range(
                    patient_id=patient_id,
                    session_date=session_date,
                    start_date=start_date,
                    end_date=end_date,
                )
            return self.get_session_summary(
                patient_id=patient_id,
                session_id=session_id,
            )

        if intent == "flag_explanation":
            if has_date_filter:
                ranged = self.get_sessions_by_date_range(
                    patient_id=patient_id,
                    session_date=session_date,
                    start_date=start_date,
                    end_date=end_date,
                )
                return ranged

            if rep_id is not None:
                return self.get_repetition(
                    patient_id=patient_id,
                    session_id=session_id,
                    rep_id=rep_id,
                )
            return self.get_flagged_repetitions(
                patient_id=patient_id,
                session_id=session_id,
            )

        if intent == "trend_analysis":
            return self.get_trend_analysis(
                patient_id=patient_id,
                session_date=session_date,
                start_date=start_date,
                end_date=end_date,
            )

        if intent == "metric_explanation":
            if has_date_filter:
                session_data = self.get_sessions_by_date_range(
                    patient_id=patient_id,
                    session_date=session_date,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                session_data = self.get_session_summary(
                    patient_id=patient_id,
                    session_id=session_id,
                )

            return {
                "available": True,
                "patient_id": patient_id,
                "session": session_data,
                "pose": self.get_pose_summary(),
                "tremor": self.get_tremor_summary(patient_id=patient_id),
            }

        return {
            "available": False,
            "patient_id": patient_id,
            "message": "No structured mobility retrieval was required.",
        }