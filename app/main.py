import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

from chatbot_ui import render_chatbot

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Healthcare Mobility Monitoring", layout="wide")

# --- ELDER-FRIENDLY STYLING + REAL DATA BADGE ---
st.markdown("""
<style>
    /* ============================================
       WCAG 2.1 Level AA Compliant Styling
       - Contrast ratios >= 4.5:1 (normal text)
       - Contrast ratios >= 3:1 (large text 18px+ / 14px+ bold)
       - Min font size 16px for body
       - Line height >= 1.5x font size
       - Focus indicators for keyboard navigation
       - Touch targets >= 44x44px
       ============================================ */

    /* Larger base font for readability (WCAG 1.4.4) */
    html, body, [class*="st-"] {
        font-size: 18px !important;
    }

    /* Headings — explicit line-height for readability (WCAG 1.4.12) */
    h1 { font-size: 2.2rem !important; line-height: 1.3 !important; }
    h2 { font-size: 1.8rem !important; line-height: 1.35 !important; }
    h3 { font-size: 1.5rem !important; line-height: 1.4 !important; }
    h4 { font-size: 1.3rem !important; line-height: 1.4 !important; }

    /* Body text — 1.5x+ line height, adequate spacing (WCAG 1.4.12) */
    p, span, label, .stMarkdown, .stCaption p {
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        letter-spacing: 0.01em !important;
        word-spacing: 0.05em !important;
    }

    /* Captions — enforce minimum contrast on dark/light themes */
    .stCaption p {
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }

    /* Metric values — large and bold (WCAG 1.4.4) */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        font-size: 1.05rem !important;
    }

    /* Buttons — min 44px touch target (WCAG 2.5.5), visible focus (WCAG 2.4.7) */
    .stButton > button {
        font-size: 1.1rem !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        min-height: 44px !important;
        min-width: 44px !important;
    }
    .stButton > button:focus-visible {
        outline: 3px solid #4A90D9 !important;
        outline-offset: 2px !important;
    }

    /* Checkboxes — accessible label size + focus */
    .stCheckbox label {
        font-size: 1.1rem !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
    }
    .stCheckbox input:focus-visible + label {
        outline: 3px solid #4A90D9 !important;
        outline-offset: 2px !important;
    }

    /* Inputs — focus indicator (WCAG 2.4.7) */
    input:focus-visible, textarea:focus-visible, select:focus-visible,
    [data-testid="stTextInput"] input:focus-visible {
        outline: 3px solid #4A90D9 !important;
        outline-offset: 1px !important;
    }

    /* Dataframe text — bold headers for differentiation */
    .stDataFrame td {
        font-size: 1rem !important;
    }
    .stDataFrame th {
        font-size: 1rem !important;
        font-weight: 700 !important;
    }

    /* Links — visible underline + focus (WCAG 1.4.1, 2.4.7) */
    a {
        text-decoration: underline !important;
    }
    a:focus-visible {
        outline: 3px solid #4A90D9 !important;
        outline-offset: 2px !important;
    }

    /* Real Data badge — WCAG AA contrast compliant
       #0b6e2f on white: 7.5:1 contrast ratio (passes AA + AAA)
       white on #0b6e2f: 7.5:1 contrast ratio (passes AA + AAA) */
    .real-data-badge {
        background: #0b6e2f;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.85rem !important;
        font-weight: 700;
        margin-left: 8px;
        vertical-align: middle;
        letter-spacing: 0.5px;
    }
    .real-data-section {
        border-left: 4px solid #0b6e2f;
        padding-left: 12px;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    /* Dark theme override for badge contrast (WCAG 1.4.3) */
    @media (prefers-color-scheme: dark) {
        .real-data-badge {
            background: #2ecc71;
            color: #000000;
        }
        .real-data-section {
            border-left-color: #2ecc71;
        }
    }

    /* Tabs — accessible focus ring */
    [data-testid="stTab"]:focus-visible {
        outline: 3px solid #4A90D9 !important;
        outline-offset: 2px !important;
    }

    /* Expanders — accessible target and focus */
    [data-testid="stExpander"] summary {
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stExpander"] summary:focus-visible {
        outline: 3px solid #4A90D9 !important;
        outline-offset: 2px !important;
    }
</style>
""", unsafe_allow_html=True)

REAL_DATA_BADGE = '<span class="real-data-badge">Real Data</span>'

# =====================================================================
# DATA SOURCE CONFIGURATION
# =====================================================================

try:
    import config as cfg
    DATA_SOURCE = cfg.DATA_SOURCE
except ImportError:
    DATA_SOURCE = os.environ.get("DATA_SOURCE", "mock")
    cfg = None

if DATA_SOURCE == "aws":
    try:
        import boto3
        from decimal import Decimal
    except ImportError:
        st.sidebar.warning("AWS dependencies not available. Using mock data.")
        DATA_SOURCE = "mock"

# =====================================================================
# LOAD REAL DATA FILES
# =====================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHAIR_REPS_PATH = os.path.join(
    BASE_DIR, "Hongpeng Data Files", "Raw Data attributes",
    "session_20260314_131518_reps.csv")

GLOVE_TREMOR_PATH = os.path.join(
    BASE_DIR, "An Data Files", "tremor_validation_master.csv")


@st.cache_data
def load_chair_reps():
    """Load Hongpeng's chair session reps CSV."""
    if not os.path.exists(CHAIR_REPS_PATH):
        return None
    df = pd.read_csv(CHAIR_REPS_PATH)
    return df


@st.cache_data
def load_glove_tremor():
    """Load An's tremor validation CSV."""
    if not os.path.exists(GLOVE_TREMOR_PATH):
        return None
    df = pd.read_csv(GLOVE_TREMOR_PATH)
    return df


def band_power_to_severity(band_power: float) -> float:
    """Map tremor band_power (4-6 Hz) to MDS-UPDRS 0-4 severity scale.

    Mapping derived from validation data:
      - Rest baseline band_power: ~12-70 -> score 0
      - Light tremor: ~600-2000 -> score 1
      - Moderate tremor: ~2000-6000 -> score 2
      - Moderate-severe: ~6000-15000 -> score 3
      - Severe (exaggerated): ~15000-26000 -> score 4

    Per An's data contract: raw DSP values are NOT shown to users.
    Only this derived severity score is displayed.
    """
    if band_power < 100:
        return 0.0
    elif band_power < 1500:
        return 0.5 + (band_power - 100) / (1500 - 100) * 0.5
    elif band_power < 5000:
        return 1.0 + (band_power - 1500) / (5000 - 1500) * 1.0
    elif band_power < 12000:
        return 2.0 + (band_power - 5000) / (12000 - 5000) * 1.0
    elif band_power < 22000:
        return 3.0 + (band_power - 12000) / (22000 - 12000) * 1.0
    else:
        return 4.0


FINGER_NAMES = {0: "Thumb", 1: "Index", 2: "Middle", 3: "Ring"}


def get_glove_session_summaries(person_id="person_1"):
    """Process glove tremor data into per-session severity summaries."""
    df = load_glove_tremor()
    if df is None:
        return None
    pdf = df[df["person_id"] == person_id].copy()
    if pdf.empty:
        return None

    sessions = []
    for test_name in pdf["test_name"].unique():
        tdf = pdf[pdf["test_name"] == test_name]
        tremor_rows = tdf[tdf["condition"] == "tremor"]
        rest_rows = tdf[tdf["condition"] == "rest"]

        if tremor_rows.empty:
            continue

        avg_bp = tremor_rows["band_power"].mean()
        avg_severity = band_power_to_severity(avg_bp)

        per_finger = {}
        for _, row in tremor_rows.iterrows():
            ch = int(row["channel"])
            fname = FINGER_NAMES.get(ch, f"Ch{ch}")
            per_finger[fname] = {
                "severity": round(band_power_to_severity(row["band_power"]), 2),
                "freq_hz": round(row["dominant_freq_hz"], 1),
            }

        rest_avg_bp = rest_rows["band_power"].mean() if not rest_rows.empty else 0
        rest_severity = band_power_to_severity(rest_avg_bp)

        timestamp = tremor_rows["timestamp"].iloc[0]
        notes = tremor_rows["notes"].iloc[0] if "notes" in tremor_rows.columns else ""

        sessions.append({
            "test_name": test_name,
            "timestamp": timestamp,
            "avg_severity": round(avg_severity, 2),
            "rest_severity": round(rest_severity, 2),
            "per_finger": per_finger,
            "avg_band_power": round(avg_bp, 1),
            "sampling_hz": tremor_rows["sampling_hz"].iloc[0],
            "notes": notes if pd.notna(notes) else "",
        })

    return sessions


def get_chair_session_summary():
    """Process chair reps into a session summary."""
    df = load_chair_reps()
    if df is None:
        return None

    left = df[df["leg"] == "left"]
    right = df[df["leg"] == "right"]

    return {
        "total_reps": len(df),
        "left_reps": len(left),
        "right_reps": len(right),
        "avg_shin_angle": round(df["shin_lift_angle_deg"].mean(), 1),
        "avg_duration": round(df["duration_s"].mean(), 1),
        "avg_jerk": round(df["jerk_score"].mean(), 3),
        "avg_angular_vel": round(df["peak_angular_velocity"].mean(), 1),
        "avg_symmetry": round(df.loc[df["tof_symmetry"] > 0, "tof_symmetry"].mean(), 3)
                        if (df["tof_symmetry"] > 0).any() else 0.0,
        "flagged_count": int(df["flagged"].sum()),
        "flag_hold_count": int(df["flag_hold"].sum()),
        "flag_angle_count": int(df["flag_angle"].sum()),
        "flag_jerk_count": int(df["flag_jerk"].sum()),
        "left_avg_angle": round(left["shin_lift_angle_deg"].mean(), 1) if not left.empty else 0,
        "right_avg_angle": round(right["shin_lift_angle_deg"].mean(), 1) if not right.empty else 0,
        "left_avg_jerk": round(left["jerk_score"].mean(), 3) if not left.empty else 0,
        "right_avg_jerk": round(right["jerk_score"].mean(), 3) if not right.empty else 0,
        "session_id": df["session_id"].iloc[0],
    }


# =====================================================================
# MOCK DATA - USERS & RBAC
# =====================================================================

USERS = {
    "P1001": {"password": "pass123", "role": "Patient", "name": "John Smith",
              "caregiver": "C2001", "clinician": "D3001"},
    "P1002": {"password": "pass123", "role": "Patient", "name": "Mary Johnson",
              "caregiver": "C2001", "clinician": "D3001"},
    "P1003": {"password": "pass123", "role": "Patient", "name": "Robert Davis",
              "caregiver": "C2002", "clinician": "D3001"},
    "P1004": {"password": "pass123", "role": "Patient", "name": "Linda Wilson",
              "caregiver": "C2002", "clinician": "D3002"},
    "P1005": {"password": "pass123", "role": "Patient", "name": "James Brown",
              "caregiver": "C2003", "clinician": "D3002"},
    "C2001": {"password": "pass123", "role": "Caregiver", "name": "Alice Martin",
              "clinician": "D3001", "patients": ["P1001", "P1002"]},
    "C2002": {"password": "pass123", "role": "Caregiver", "name": "Bob Taylor",
              "clinician": "D3001", "patients": ["P1003", "P1004"]},
    "C2003": {"password": "pass123", "role": "Caregiver", "name": "Carol White",
              "clinician": "D3002", "patients": ["P1005"]},
    "D3001": {"password": "pass123", "role": "Clinician", "name": "Dr. Sarah Lee",
              "caregivers": ["C2001", "C2002"]},
    "D3002": {"password": "pass123", "role": "Clinician", "name": "Dr. Michael Chen",
              "caregivers": ["C2003"]},
}


# =====================================================================
# EXERCISE DEFINITIONS
# =====================================================================

def get_exercises(patient_id: str):
    """Return exercises for a patient -- mock + real data exercises."""
    all_mock = [
        {"name": "Finger Tapping Drill", "category": "Upper Limb Exercises",
         "duration": "5 min", "frequency": "Daily", "device": "Glove",
         "device_id": "Glove_001", "exercise_code": "finger_tapping", "mds_updrs": "3.4",
         "metric": "Tap Rate (taps/s)", "target": 3.0, "lower_is_better": False,
         "youtube": "https://www.youtube.com/watch?v=OKpGKiEGbFY",
         "description": "Tap each finger to thumb rapidly for 1 minute per hand. "
                        "Glove IMU tracks movement patterns for MDS-UPDRS 3.4 scoring.",
         "is_real_data": False},
        {"name": "Hand Open-Close Repetitions", "category": "Upper Limb Exercises",
         "duration": "5 min", "frequency": "Daily", "device": "Glove",
         "device_id": "Glove_001", "exercise_code": "hand_open_close", "mds_updrs": "3.5",
         "metric": "Grip Force (N)", "target": 15.0, "lower_is_better": False,
         "youtube": "https://www.youtube.com/watch?v=QFpLkgD3R_8",
         "description": "Fully open and close each hand 20 times. "
                        "Glove measures grip force and release speed (MDS-UPDRS 3.5).",
         "is_real_data": False},
        {"name": "Resting Tremor Assessment", "category": "Upper Limb Exercises",
         "duration": "3 min", "frequency": "Daily", "device": "Glove",
         "device_id": "Glove_001", "exercise_code": "rest_tremor", "mds_updrs": "3.17-3.18",
         "metric": "Tremor Severity (0-4)", "target": 1.5, "lower_is_better": True,
         "youtube": "https://www.youtube.com/watch?v=4SZsPGa-yBQ",
         "description": "Rest hands on lap for 3 minutes while glove records "
                        "resting tremor intensity (MDS-UPDRS 3.17-3.18).",
         "is_real_data": False},
        {"name": "Sit-to-Stand Practice", "category": "Lower Limb Exercises",
         "duration": "10 min", "frequency": "Daily", "device": "Chair",
         "device_id": "Chair_001", "exercise_code": "sit_to_stand", "mds_updrs": None,
         "metric": "Rep Duration (s)", "target": 4.0, "lower_is_better": True,
         "youtube": "https://www.youtube.com/watch?v=RjnNMRsmNqU",
         "description": "Rise from the sensor chair without using hands, 10 reps.",
         "is_real_data": False},
        {"name": "Seated Marching", "category": "Lower Limb Exercises",
         "duration": "5 min", "frequency": "Daily", "device": "Chair",
         "device_id": "Chair_001", "exercise_code": "seated_marching", "mds_updrs": None,
         "metric": "Peak Angular Velocity (deg/s)", "target": 100.0,
         "lower_is_better": False,
         "youtube": "https://www.youtube.com/watch?v=J5_QC1ZZqvg",
         "description": "Alternate lifting knees in a marching motion while seated.",
         "is_real_data": False},
    ]

    real_exercises = []

    chair_reps = load_chair_reps()
    if chair_reps is not None:
        real_exercises.append({
            "name": "Leg Raise Session (2026-03-14)",
            "category": "Lower Limb Exercises",
            "duration": "1 min", "frequency": "Session", "device": "Chair",
            "device_id": "Chair_001", "exercise_code": "leg_raises_real", "mds_updrs": None,
            "metric": "Jerk Score", "target": 0.35, "lower_is_better": True,
            "youtube": "https://www.youtube.com/watch?v=YoGa_PoFfEo",
            "description": "Perform seated leg raises, alternating legs. "
                           "Chair sensors will track your shin angle, movement smoothness, and balance.",
            "is_real_data": True, "real_data_type": "chair_reps"})

    glove_sessions = get_glove_session_summaries("person_1")
    if glove_sessions:
        real_exercises.append({
            "name": "Tremor Assessment (Validation)",
            "category": "Upper Limb Exercises",
            "duration": "30 sec", "frequency": "Session", "device": "Glove",
            "device_id": "Glove_001", "exercise_code": "rest_tremor_real",
            "mds_updrs": "3.17-3.18",
            "metric": "Tremor Severity (0-4)", "target": 1.5, "lower_is_better": True,
            "youtube": "https://www.youtube.com/watch?v=4SZsPGa-yBQ",
            "description": "Rest hands on lap while the glove records tremor activity. "
                           "Severity is assessed per finger across the 4-6 Hz tremor band.",
            "is_real_data": True, "real_data_type": "glove_tremor"})

    np.random.seed(hash(patient_id + "exercises") % 2**31)
    count = np.random.randint(3, len(all_mock) + 1)
    indices = np.random.choice(len(all_mock), size=count, replace=False)
    selected = [all_mock[i] for i in sorted(indices)]

    return selected + real_exercises


# =====================================================================
# DATA GENERATORS
# =====================================================================

METRIC_RANGES = {
    "Tap Rate (taps/s)":              (1.0, 4.5),
    "Grip Force (N)":                 (8.0, 22.0),
    "Rotation Angle (deg)":           (80.0, 180.0),
    "Tremor Severity (0-4)":          (0.3, 3.5),
    "Range of Motion (deg)":          (35.0, 85.0),
    "Rep Duration (s)":               (3.0, 7.5),
    "Shin Lift Angle (deg)":          (40.0, 135.0),
    "Peak Angular Velocity (deg/s)":  (55.0, 190.0),
    "Symmetry Score":                 (0.0, 1.0),
    "Jerk Score":                     (0.15, 0.55),
}


def generate_exercise_history(patient_id: str, exercise: dict):
    """Generate history data for an exercise -- mock or real."""

    if exercise.get("real_data_type") == "chair_reps":
        df = load_chair_reps()
        if df is not None:
            labels = [f"Rep {r} ({'L' if l == 'left' else 'R'})"
                      for r, l in zip(df["rep_id"], df["leg"])]
            return pd.DataFrame({
                "Day": labels,
                "Duration (min)": (df["duration_s"] / 60.0).round(2).values,
                exercise["metric"]: df["jerk_score"].round(3).values,
            })

    if exercise.get("real_data_type") == "glove_tremor":
        sessions = get_glove_session_summaries("person_1")
        if sessions:
            labels = [f"Session {i+1}" for i in range(len(sessions))]
            severities = [s["avg_severity"] for s in sessions]
            durations = [0.5] * len(sessions)
            return pd.DataFrame({
                "Day": labels,
                "Duration (min)": durations,
                exercise["metric"]: severities,
            })

    seed = hash(patient_id + exercise["name"]) % 2**31
    np.random.seed(seed)
    today = datetime.date.today()
    dates = [today - datetime.timedelta(days=i) for i in range(6, -1, -1)]
    day_labels = [d.strftime("%a %d") for d in dates]

    base_dur = float(exercise["duration"].split()[0])
    durations = np.random.uniform(max(1, base_dur - 3), base_dur + 2, 7).round(1)

    metric_name = exercise["metric"]
    lo, hi = METRIC_RANGES.get(metric_name, (0.0, 10.0))
    metrics = np.random.uniform(lo, hi, 7).round(2)

    return pd.DataFrame({
        "Day": day_labels,
        "Duration (min)": durations,
        metric_name: metrics,
    })


# =====================================================================
# REAL DATA DETAIL RENDERERS
# =====================================================================

def render_chair_real_detail():
    """Render detailed real data section for Chair session.

    Shows 4 uniform bar charts: Duration, Jerk Score, Hold Duration, Symmetry.
    Flagged reps shown in red for Jerk Score and Hold Duration charts.
    """
    import plotly.graph_objects as go

    df = load_chair_reps()
    if df is None:
        return
    summary = get_chair_session_summary()
    if summary is None:
        return

    st.markdown(f'<div class="real-data-section">', unsafe_allow_html=True)
    st.markdown(f"**Session:** `{summary['session_id']}` -- "
                f"{summary['total_reps']} reps "
                f"({summary['left_reps']} left, {summary['right_reps']} right)")

    # Left vs Right summary
    col_l, col_r = st.columns(2)
    with col_l:
        st.metric("Left Leg Avg Jerk", f"{summary['left_avg_jerk']}")
    with col_r:
        st.metric("Right Leg Avg Jerk", f"{summary['right_avg_jerk']}")

    # Flag summary
    if summary["flagged_count"] > 0:
        st.warning(
            f"**{summary['flagged_count']}/{summary['total_reps']} reps flagged** -- "
            f"Hold: {summary['flag_hold_count']}, "
            f"Angle: {summary['flag_angle_count']}, "
            f"Jerk: {summary['flag_jerk_count']}"
        )
    else:
        st.success("No reps flagged in this session.")

    rep_labels = [f"R{r} ({'L' if l == 'left' else 'R'})"
                  for r, l in zip(df["rep_id"], df["leg"])]
    flagged_mask = df["flagged"].values

    def _chair_bar(y_vals, y_label, sym_mode=False):
        """Build a Plotly bar chart with red bars + black border for flagged reps."""
        if sym_mode:
            colors = ["#CCCCCC" if v == 0.0 else ("#EF553B" if f else "#636EFA")
                      for v, f in zip(y_vals, flagged_mask)]
        else:
            colors = ["#EF553B" if f else "#636EFA" for f in flagged_mask]
        border_widths = [2 if f else 0 for f in flagged_mask]
        fig = go.Figure(go.Bar(
            x=rep_labels, y=y_vals, marker_color=colors,
            marker_line_width=border_widths, marker_line_color="black",
        ))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=40),
                          xaxis_title="", yaxis_title=y_label)
        return fig

    # Legend — single mention
    st.caption("🔴 Red bars = flagged repetitions  |  ⬜ Grey bars = sensor out of range")

    # --- Row 1: Duration + Jerk Score ---
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.markdown(f"**Duration per Rep (s)** {REAL_DATA_BADGE}", unsafe_allow_html=True)
        st.plotly_chart(_chair_bar(df["duration_s"].values, "Duration (s)"),
                        use_container_width=True)
    with row1_c2:
        st.markdown(f"**Jerk Score per Rep** {REAL_DATA_BADGE}", unsafe_allow_html=True)
        st.plotly_chart(_chair_bar(df["jerk_score"].values, "Jerk Score"),
                        use_container_width=True)

    # --- Row 2: Hold Duration + Symmetry ---
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.markdown(f"**Hold Duration per Rep (s)** {REAL_DATA_BADGE}", unsafe_allow_html=True)
        st.plotly_chart(_chair_bar(df["hold_duration_s"].values, "Hold (s)"),
                        use_container_width=True)
    with row2_c2:
        st.markdown(f"**Symmetry Score per Rep** {REAL_DATA_BADGE}", unsafe_allow_html=True)
        st.plotly_chart(_chair_bar(df["tof_symmetry"].values, "Symmetry", sym_mode=True),
                        use_container_width=True)

    # Per-rep detail table
    with st.expander("Per-Rep Detail Table"):
        display_df = df[["rep_id", "leg", "duration_s", "hold_duration_s",
                         "jerk_score", "tof_symmetry", "flagged",
                         "flag_hold", "flag_jerk"]].copy()
        display_df.columns = ["Rep", "Leg", "Duration (s)", "Hold (s)",
                              "Jerk", "Symmetry", "Flagged",
                              "Flag Hold", "Flag Jerk"]
        st.dataframe(display_df, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_glove_real_detail():
    """Render detailed real data section for Glove tremor."""
    sessions = get_glove_session_summaries("person_1")
    if not sessions:
        return

    st.markdown(f'<div class="real-data-section">', unsafe_allow_html=True)
    st.markdown(f"**Source:** PD-Glove hardware validation -- "
                f"{len(sessions)} sessions, 4 finger channels per session")

    latest = sessions[-1]
    st.markdown(f"**Per-Finger Tremor Severity (Latest: {latest['test_name']})** "
                f"{REAL_DATA_BADGE}", unsafe_allow_html=True)

    finger_cols = st.columns(4)
    for i, (fname, fdata) in enumerate(latest["per_finger"].items()):
        with finger_cols[i]:
            sev = fdata["severity"]
            if sev < 1.0:
                color, label = "🟢", "Normal"
            elif sev < 2.0:
                color, label = "🟡", "Mild"
            elif sev < 3.0:
                color, label = "🟠", "Moderate"
            else:
                color, label = "🔴", "Severe"
            st.metric(f"{color} {fname} ({label})", f"{sev}/4")
            st.caption(f"Freq: {fdata['freq_hz']} Hz")

    # Rest baseline summary
    rest_vals = [s["rest_severity"] for s in sessions]
    max_rest = max(rest_vals)
    if max_rest > 0:
        st.warning(f"⚠️ Rest baseline elevated ({max_rest}/4) — tremor detected even at rest.")
    else:
        st.caption("Rest baseline: normal (0/4 across all sessions) — no tremor at rest.")

    noted = [s for s in sessions if s["notes"]]
    if noted:
        with st.expander("Session Notes"):
            for s in noted:
                st.caption(f"**{s['test_name']}**: {s['notes']}")

    with st.expander("Session Detail Table"):
        rows = []
        for s in sessions:
            row = {
                "Session": s["test_name"],
                "Avg Severity": s["avg_severity"],
                "Rest Baseline": s["rest_severity"],
                "Sampling (Hz)": s["sampling_hz"],
            }
            for fname, fdata in s["per_finger"].items():
                row[f"{fname} Sev"] = fdata["severity"]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# SCORING, ALERTS, AND FEEDBACK
# =====================================================================

def get_exercise_remark(exercise: dict, latest_value: float):
    target = exercise["target"]
    if exercise["lower_is_better"]:
        ratio = target / latest_value if latest_value > 0 else 1.0
    else:
        ratio = latest_value / target if target > 0 else 1.0

    if ratio >= 1.0:
        return "😊", "On Track"
    elif ratio >= 0.75:
        return "😐", "Needs Attention"
    else:
        return "😟", "Below Target"


def get_alerts(patient_id: str):
    alerts = [
        {"time": "10:45 AM", "type": "Fall Risk",
         "message": "Near-fall detected during sit-to-stand transition.",
         "severity": "high", "source": "Chair"},
        {"time": "02:15 PM", "type": "Inactivity",
         "message": "Seated for over 2.5 hours -- encourage movement.",
         "severity": "medium", "source": "Chair"},
    ]

    summary = get_chair_session_summary()
    if summary and summary["flagged_count"] > 0:
        alerts.append({
            "time": "Session", "type": "Rep Quality Flags",
            "message": (f"{summary['flagged_count']}/{summary['total_reps']} reps flagged "
                        f"in session {summary['session_id']} -- "
                        f"Hold: {summary['flag_hold_count']}, "
                        f"Jerk: {summary['flag_jerk_count']}"),
            "severity": "medium", "source": "Chair - Real Data",
        })

    sessions = get_glove_session_summaries("person_1")
    if sessions:
        latest_sev = sessions[-1]["avg_severity"]
        if latest_sev >= 2.5:
            alerts.append({
                "time": "Session", "type": "Tremor Spike",
                "message": (f"Latest tremor severity {latest_sev}/4 "
                            f"(session: {sessions[-1]['test_name']}). "
                            f"Consider medication review."),
                "severity": "high", "source": "Glove - Real Data",
            })
        elif latest_sev >= 1.5:
            alerts.append({
                "time": "Session", "type": "Tremor Elevated",
                "message": f"Tremor severity {latest_sev}/4 -- monitor trend.",
                "severity": "medium", "source": "Glove - Real Data",
            })

    return alerts


def get_exercise_feedbacks(patient_id: str):
    user_info = USERS[patient_id]
    clinician_id = user_info.get("clinician", "")
    caregiver_id = user_info.get("caregiver", "")
    clinician_name = USERS.get(clinician_id, {}).get("name", "Clinician")
    caregiver_name = USERS.get(caregiver_id, {}).get("name", "Caregiver")

    return {
        "Finger Tapping Drill": {
            "from": clinician_name, "role": "Clinician", "date": "2026-03-14",
            "message": "Tap rate is improving. Keep consistent with daily practice."},
        "Hand Open-Close Repetitions": {
            "from": caregiver_name, "role": "Caregiver", "date": "2026-03-15",
            "message": "Grip strength looks better this week. Great effort!"},
        "Resting Tremor Assessment": {
            "from": clinician_name, "role": "Clinician", "date": "2026-03-15",
            "message": "Tremor severity score trending down -- medication adjustment is working."},
        "Sit-to-Stand Practice": {
            "from": clinician_name, "role": "Clinician", "date": "2026-03-14",
            "message": "Rep duration is decreasing nicely. Watch balance during the rise."},
        "Seated Marching": {
            "from": clinician_name, "role": "Clinician", "date": "2026-03-13",
            "message": "Good angular velocity. Increase pace gradually next week."},
        "Leg Raise Session (2026-03-14)": {
            "from": clinician_name, "role": "Clinician", "date": "2026-03-14",
            "message": "Good shin lift angles overall. Some reps flagged for hold duration -- "
                       "try holding at the top for at least 1 second."},
        "Tremor Assessment (Validation)": {
            "from": clinician_name, "role": "Clinician", "date": "2026-03-25",
            "message": "Tremor severity varies across sessions. Session 5 shows improvement. "
                       "Continue monitoring trend."},
    }


def get_todays_exercises(patient_id: str):
    exercises = get_exercises(patient_id)
    np.random.seed(hash(patient_id + str(datetime.date.today())) % 2**31)
    todays = []
    for ex in exercises:
        if ex.get("is_real_data"):
            todays.append(ex)
        elif ex["frequency"] == "Daily":
            todays.append(ex)
        elif np.random.random() > 0.5:
            todays.append(ex)
    return todays


def get_feedbacks(patient_id: str):
    return [
        {"from": "Dr. Sarah Lee", "date": "2026-02-24", "source": "Doctor",
         "message": "Tremor severity levels look stable this week. Keep up the exercises."},
        {"from": "Dr. Michael Chen", "date": "2026-02-22", "source": "Doctor",
         "message": "Consider adjusting sitting breaks -- prolonged sitting detected by chair sensors."},
        {"from": "Alice Martin", "date": "2026-02-23", "source": "Caregiver",
         "message": "Your tap rate has improved -- keep it up!"},
        {"from": "Bob Taylor", "date": "2026-02-21", "source": "Caregiver",
         "message": "Remember to do your stretches after long sitting periods."},
    ]


# =====================================================================
# SESSION STATE HELPERS
# =====================================================================

def init_session():
    for key in ["logged_in", "user_id", "role", "name", "view", "selected_patient",
                "selected_caregiver"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.logged_in is None:
        st.session_state.logged_in = False


def do_login(user_id: str, password: str, role: str):
    user_id = user_id.strip().upper()
    if user_id in USERS and USERS[user_id]["password"] == password and USERS[user_id]["role"] == role:
        st.session_state.logged_in = True
        st.session_state.user_id = user_id
        st.session_state.role = role
        st.session_state.name = USERS[user_id]["name"]
        st.session_state.view = "home"
        st.session_state.selected_patient = None
        st.session_state.selected_caregiver = None
        return True
    return False


def do_logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.logged_in = False


# =====================================================================
# SHARED COMPONENTS
# =====================================================================

def render_reports(patient_id: str):
    exercises = get_exercises(patient_id)
    ex_feedbacks = get_exercise_feedbacks(patient_id)

    mock_exercises = [ex for ex in exercises if not ex.get("is_real_data")]
    real_exercises = [ex for ex in exercises if ex.get("is_real_data")]

    tabs = ["📋 Daily Report", "📊 Weekly Report"]
    if real_exercises:
        tabs.append("🟢 Real Data Reports")
    tabs.append("📥 Download")
    tab_list = st.tabs(tabs)

    # --- DAILY REPORT TAB (mock exercises only) ---
    with tab_list[0]:
        st.subheader("📋 Today's Exercise Report")

        on_track = 0
        glove_scores = []
        chair_scores = []

        for ex in mock_exercises:
            history = generate_exercise_history(patient_id, ex)
            latest_val = history[ex["metric"]].iloc[-1]
            emoji, remark_text = get_exercise_remark(ex, latest_val)
            target = ex["target"]
            if ex["lower_is_better"]:
                ratio = target / latest_val if latest_val > 0 else 1.0
            else:
                ratio = latest_val / target if target > 0 else 1.0
            ratio = min(ratio, 1.5)
            if remark_text == "On Track":
                on_track += 1
            if ex["category"] == "Upper Limb Exercises":
                glove_scores.append(ratio)
            else:
                chair_scores.append(ratio)

        total_ex = len(mock_exercises)
        glove_avg = round(sum(glove_scores) / len(glove_scores) * 100) if glove_scores else 0
        chair_avg = round(sum(chair_scores) / len(chair_scores) * 100) if chair_scores else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Exercises On Track", f"{on_track}/{total_ex}")
        c2.metric("Upper-Limb Score", f"{glove_avg}%")
        c3.metric("Lower-Limb Score", f"{chair_avg}%")
        st.divider()

        categories = {}
        for ex in mock_exercises:
            categories.setdefault(ex["category"], []).append(ex)

        device_icons = {"Upper Limb Exercises": "🧤", "Lower Limb Exercises": "🪑"}
        for cat, exs in categories.items():
            icon = device_icons.get(cat, "")
            st.markdown(f"### {icon} {cat}")
            for ex in exs:
                history = generate_exercise_history(patient_id, ex)
                latest_val = history[ex["metric"]].iloc[-1]
                latest_dur = history["Duration (min)"].iloc[-1]
                emoji, remark_text = get_exercise_remark(ex, latest_val)
                target_label = "below" if ex["lower_is_better"] else "above"
                st.markdown(f"**{ex['name']}**")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Duration", f"{latest_dur} min")
                mc2.metric(ex["metric"], f"{latest_val}")
                mc2.caption(f"Target: {target_label} {ex['target']}")
                mc3.metric("Status", remark_text)
                fb = ex_feedbacks.get(ex["name"])
                if fb:
                    role_icon = "🩺" if fb["role"] == "Clinician" else "🤝"
                    st.info(f"{role_icon} **{fb['from']}** ({fb['role']}) -- _{fb['date']}_: {fb['message']}")
                st.divider()

    # --- WEEKLY REPORT TAB (mock exercises only) ---
    with tab_list[1]:
        st.subheader("📊 Past 7 Days -- Exercise Trends")
        categories = {}
        for ex in mock_exercises:
            categories.setdefault(ex["category"], []).append(ex)
        device_icons = {"Upper Limb Exercises": "🧤", "Lower Limb Exercises": "🪑"}
        for cat, exs in categories.items():
            icon = device_icons.get(cat, "")
            st.markdown(f"### {icon} {cat}")
            for ex in exs:
                history = generate_exercise_history(patient_id, ex)
                st.markdown(f"**{ex['name']}** -- _{ex['duration']}, {ex['frequency']}_")
                col_dur, col_metric = st.columns(2)
                with col_dur:
                    st.markdown("**Duration (min)**")
                    st.line_chart(history.set_index("Day")[["Duration (min)"]], height=180)
                with col_metric:
                    st.markdown(f"**{ex['metric']}**")
                    st.bar_chart(history.set_index("Day")[[ex["metric"]]], height=180)
                    target_label = "below" if ex["lower_is_better"] else "above"
                    st.caption(f"Target: {target_label} {ex['target']}")
                st.divider()

    # --- REAL DATA REPORTS TAB ---
    if real_exercises:
        with tab_list[2]:
            st.markdown(f"## 🟢 Real Data Reports {REAL_DATA_BADGE}", unsafe_allow_html=True)
            st.caption("These reports are generated from actual hardware sensor data.")
            st.divider()

            for ex in real_exercises:
                history = generate_exercise_history(patient_id, ex)
                latest_val = history[ex["metric"]].iloc[-1]
                emoji, remark_text = get_exercise_remark(ex, latest_val)

                st.markdown(f"### {ex['name']} {REAL_DATA_BADGE}", unsafe_allow_html=True)
                st.caption(f"Device: {ex['device']} | Exercise: {ex.get('exercise_code', '')} | "
                           f"MDS-UPDRS: {ex.get('mds_updrs') or 'N/A'}")

                # Summary metrics
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Duration", f"{ex['duration']}")
                mc2.metric(ex["metric"], f"{latest_val}")
                target_label = "below" if ex["lower_is_better"] else "above"
                mc2.caption(f"Target: {target_label} {ex['target']}")
                mc3.metric("Status", remark_text)

                # Chair real data skips primary charts — all shown uniformly in detail section
                if ex.get("real_data_type") != "chair_reps":
                    col_a, col_b = st.columns(2)
                    with col_a:
                        dur_label = "per Session"
                        st.markdown(f"**Duration ({dur_label})** {REAL_DATA_BADGE}",
                                    unsafe_allow_html=True)
                        st.line_chart(history.set_index("Day")[["Duration (min)"]], height=200)
                    with col_b:
                        st.markdown(f"**{ex['metric']}** {REAL_DATA_BADGE}",
                                    unsafe_allow_html=True)
                        st.bar_chart(history.set_index("Day")[[ex["metric"]]], height=200)
                        target_label = "below" if ex["lower_is_better"] else "above"
                        st.caption(f"Target: {target_label} {ex['target']}")

                # Feedback
                fb = ex_feedbacks.get(ex["name"])
                if fb:
                    role_icon = "🩺" if fb["role"] == "Clinician" else "🤝"
                    st.info(f"{role_icon} **{fb['from']}** ({fb['role']}) -- _{fb['date']}_: {fb['message']}")

                # Detailed real data sections
                if ex.get("real_data_type") == "chair_reps":
                    render_chair_real_detail()
                elif ex.get("real_data_type") == "glove_tremor":
                    render_glove_real_detail()

                st.divider()

    # --- DOWNLOAD TAB ---
    download_idx = 3 if real_exercises else 2
    with tab_list[download_idx]:
        st.subheader("📥 Download Report")
        render_download_button(patient_id)


def render_alerts(patient_id: str):
    alerts = get_alerts(patient_id)
    st.subheader("🚨 Alerts")
    for a in alerts:
        source_tag = f" [{a.get('source', '')}]" if a.get("source") else ""
        if a["severity"] == "high":
            st.error(f"**{a['time']} -- {a['type']}{source_tag}**: {a['message']}")
        elif a["severity"] == "medium":
            st.warning(f"**{a['time']} -- {a['type']}{source_tag}**: {a['message']}")
        else:
            st.info(f"**{a['time']} -- {a['type']}{source_tag}**: {a['message']}")


def render_download_button(patient_id: str):
    exercises = get_exercises(patient_id)
    rows = []
    for ex in exercises:
        history = generate_exercise_history(patient_id, ex)
        for _, row in history.iterrows():
            rows.append({
                "Exercise": ex["name"],
                "Device": ex["device"],
                "Data Source": "Real" if ex.get("is_real_data") else "Mock",
                "Day": row["Day"],
                "Duration (min)": row["Duration (min)"],
                "Metric": ex["metric"],
                "Value": row[ex["metric"]],
                "Target": ex["target"],
            })
    csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Patient Report (CSV)",
        data=csv,
        file_name=f"patient_{patient_id}_report.csv",
        mime="text/csv",
    )


def render_send_feedback(sender_id: str, target_id: str, label: str = "patient"):
    with st.expander(f"Send Feedback / Alert to {label} {target_id}"):
        fb_type = st.selectbox("Type", ["Feedback", "Alert"], key=f"fb_type_{target_id}")
        fb_msg = st.text_area("Message", key=f"fb_msg_{target_id}")
        if st.button("Send", key=f"fb_send_{target_id}"):
            if fb_msg.strip():
                st.success(f"{fb_type} sent to {target_id} successfully!")
            else:
                st.warning("Please enter a message.")


# =====================================================================
# LOGIN PAGE
# =====================================================================

def login_page():
    st.markdown("<h1 style='text-align:center;'>Healthcare Mobility Monitoring System</h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Login to continue</h2>",
                unsafe_allow_html=True)
    st.divider()
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        role = st.selectbox("I am a:", ["Patient", "Caregiver", "Clinician"])
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if do_login(user_id, password, role):
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
        st.divider()
        st.markdown("**Demo credentials** (password for all: `pass123`)")
        demo = {
            "Patient": "P1001, P1002, P1003, P1004, P1005",
            "Caregiver": "C2001, C2002, C2003",
            "Clinician": "D3001, D3002",
        }
        for r, ids in demo.items():
            st.markdown(f"- **{r}**: {ids}")


# =====================================================================
# PATIENT FRONTEND
# =====================================================================

def patient_dashboard():
    uid = st.session_state.user_id
    name = st.session_state.name

    st.sidebar.title(f"👤 {name}")
    st.sidebar.caption(f"Patient ID: {uid}")
    if st.sidebar.button("Logout"):
        do_logout()
        st.rerun()

    menu = st.sidebar.radio("Navigation",
                            ["Home", "Daily Goals", "Reports", "Doctor Feedback", "AI Assistant"])

    if menu == "Home":
        hdr_left, hdr_right = st.columns([5, 1])
        with hdr_left:
            st.markdown(f"<h1 style='text-align:center;'>Welcome, {name}!</h1>",
                        unsafe_allow_html=True)
            st.markdown("<h3 style='text-align:center;'>Your daily mobility summary</h3>",
                        unsafe_allow_html=True)
        with hdr_right:
            alerts = get_alerts(uid)
            bell_label = f"🔔 {len(alerts)} alerts" if alerts else "🔔 No alerts"
            with st.popover(bell_label):
                st.markdown("### 🚨 Alerts")
                if alerts:
                    for a in alerts:
                        source_tag = f" [{a.get('source', '')}]" if a.get("source") else ""
                        if a["severity"] == "high":
                            st.error(f"**{a['time']} -- {a['type']}{source_tag}**: {a['message']}")
                        elif a["severity"] == "medium":
                            st.warning(f"**{a['time']} -- {a['type']}{source_tag}**: {a['message']}")
                        else:
                            st.info(f"**{a['time']} -- {a['type']}{source_tag}**: {a['message']}")
                else:
                    st.caption("No alerts right now.")
        st.divider()

        exercises = get_exercises(uid)
        ex_feedbacks = get_exercise_feedbacks(uid)

        on_track = 0
        needs_attention = 0
        below_target = 0
        glove_scores = []
        chair_scores = []

        for ex in exercises:
            history = generate_exercise_history(uid, ex)
            latest_val = history[ex["metric"]].iloc[-1]
            emoji, remark_text = get_exercise_remark(ex, latest_val)
            if remark_text == "On Track":
                on_track += 1
            elif remark_text == "Needs Attention":
                needs_attention += 1
            else:
                below_target += 1
            target = ex["target"]
            if ex["lower_is_better"]:
                ratio = target / latest_val if latest_val > 0 else 1.0
            else:
                ratio = latest_val / target if target > 0 else 1.0
            ratio = min(ratio, 1.5)
            if ex["category"] == "Upper Limb Exercises":
                glove_scores.append(ratio)
            else:
                chair_scores.append(ratio)

        total_ex = len(exercises)
        overall_pct = round(on_track / total_ex * 100) if total_ex else 0
        glove_avg = round(sum(glove_scores) / len(glove_scores) * 100) if glove_scores else 0
        chair_avg = round(sum(chair_scores) / len(chair_scores) * 100) if chair_scores else 0

        if overall_pct >= 75:
            health_label = "Good"
        elif overall_pct >= 50:
            health_label = "Fair"
        else:
            health_label = "Needs Improvement"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exercises On Track", f"{on_track}/{total_ex}")
        c2.metric("Upper-Limb Score", f"{glove_avg}%")
        c3.metric("Lower-Limb Score", f"{chair_avg}%")
        c4.metric("Overall Health", health_label)

        st.divider()
        if below_target > 0:
            st.warning(f"⚠️ {below_target} exercise(s) are below target today. Check the summaries below for details.")
        if needs_attention > 0:
            st.info(f"💡 {needs_attention} exercise(s) need attention -- you're close to target, keep pushing!")
        if on_track == total_ex:
            st.success("🏆 All exercises are on track today! Excellent work!")

        st.divider()
        st.subheader("🏋️ Assigned Exercises -- Summary Charts")

        categories = {}
        for ex in exercises:
            categories.setdefault(ex["category"], []).append(ex)

        device_icons = {"Upper Limb Exercises": "🧤", "Lower Limb Exercises": "🪑"}

        for cat, exs in categories.items():
            icon = device_icons.get(cat, "")
            st.markdown(f"### {icon} {cat}")

            for ex in exs:
                history = generate_exercise_history(uid, ex)
                latest_val = history[ex["metric"]].iloc[-1]
                emoji, remark_text = get_exercise_remark(ex, latest_val)
                badge = REAL_DATA_BADGE if ex.get("is_real_data") else ""
                st.markdown(f"#### {ex['name']}  --  _{ex['duration']}, {ex['frequency']}_ {badge}",
                            unsafe_allow_html=True)

                # Chair real data skips the primary chart section — all charts
                # are rendered uniformly in render_chair_real_detail() below.
                if ex.get("real_data_type") != "chair_reps":
                    col_time, col_metric, col_remark = st.columns([2, 2, 1])

                    with col_time:
                        if ex.get("is_real_data"):
                            dur_label = "per Session"
                            st.markdown(f"**Duration ({dur_label})** {REAL_DATA_BADGE}",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown("**Duration (past 7 days)**")
                        st.line_chart(history.set_index("Day")[["Duration (min)"]], height=180)
                        st.caption(f"Latest: **{history['Duration (min)'].iloc[-1]} min**")

                    with col_metric:
                        if ex.get("is_real_data"):
                            st.markdown(f"**{ex['metric']}** {REAL_DATA_BADGE}",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{ex['metric']}**")
                        st.bar_chart(history.set_index("Day")[[ex["metric"]]], height=180)
                        target_label = "below" if ex["lower_is_better"] else "above"
                        st.caption(f"Latest: **{latest_val}** -- Target: {target_label} {ex['target']}")

                    with col_remark:
                        st.markdown("**Remarks**")
                        st.markdown(f"<div style='text-align:center;font-size:3rem;'>{emoji}</div>",
                                    unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center;font-weight:bold;'>{remark_text}</div>",
                                    unsafe_allow_html=True)

                fb = ex_feedbacks.get(ex["name"])
                if fb:
                    role_icon = "🩺" if fb["role"] == "Clinician" else "🤝"
                    st.info(f"{role_icon} **{fb['from']}** ({fb['role']}) -- _{fb['date']}_: {fb['message']}")

                if ex.get("real_data_type") == "chair_reps":
                    render_chair_real_detail()
                elif ex.get("real_data_type") == "glove_tremor":
                    render_glove_real_detail()

                st.divider()

    elif menu == "Daily Goals":
        st.title("🎯 Daily Goals")
        todays_ex = get_todays_exercises(uid)
        if todays_ex:
            total = len(todays_ex)
            progress_placeholder = st.empty()
            completion_placeholder = st.empty()
            st.divider()
            today_cats = {}
            for ex in todays_ex:
                today_cats.setdefault(ex["category"], []).append(ex)
            device_icons = {"Upper Limb Exercises": "🧤", "Lower Limb Exercises": "🪑"}
            for cat, exs in today_cats.items():
                icon = device_icons.get(cat, "")
                st.markdown(f"### {icon} {cat}")
                for ex in exs:
                    chk_key = f"chk_{uid}_{ex['name']}"
                    done = st.checkbox(f"**{ex['name']}** -- {ex['duration']}",
                                       key=chk_key, value=True if ex.get("is_real_data") else False)
                    if ex.get("is_real_data"):
                        st.markdown(REAL_DATA_BADGE, unsafe_allow_html=True)
                    if done:
                        st.caption("✅ Completed")
                    else:
                        st.caption(f"📝 {ex['description']}")
                    st.markdown(f"[▶ Watch exercise video]({ex['youtube']})")
            completed_count = sum(1 for ex in todays_ex
                                  if st.session_state.get(f"chk_{uid}_{ex['name']}", False))
            progress_placeholder.progress(completed_count / total,
                                          text=f"{completed_count}/{total} exercises completed")
            if completed_count == total:
                st.balloons()
                completion_placeholder.success("🎉 All exercises completed for today! Great work!")
        else:
            st.info("No exercises scheduled for today. Enjoy your rest day!")

    elif menu == "Reports":
        st.title("📊 Reports")
        render_reports(uid)

    elif menu == "Doctor Feedback":
        st.title("💬 Feedback from Your Care Team")
        feedbacks = get_feedbacks(uid)
        doctor_fb = [fb for fb in feedbacks if fb["source"] == "Doctor"]
        caregiver_fb = [fb for fb in feedbacks if fb["source"] == "Caregiver"]
        st.markdown("### 🩺 Doctor Feedback")
        if doctor_fb:
            for fb in doctor_fb:
                with st.container():
                    st.markdown(f"**{fb['from']}** -- _{fb['date']}_")
                    st.info(fb["message"])
        else:
            st.caption("No doctor feedback yet.")
        st.divider()
        st.markdown("### 🤝 Caregiver Feedback")
        if caregiver_fb:
            for fb in caregiver_fb:
                with st.container():
                    st.markdown(f"**{fb['from']}** -- _{fb['date']}_")
                    st.success(fb["message"])
        else:
            st.caption("No caregiver feedback yet.")


    elif menu == "AI Assistant":
        render_chatbot(
            user_id=uid,
            user_name=name,
            role="Patient",
            patient_id=uid,
            authorized_patient_ids=[uid],
            current_page=menu,
            active_alert={"alerts": get_alerts(uid)},
        )


# =====================================================================
# CAREGIVER FRONTEND
# =====================================================================

def caregiver_dashboard():
    uid = st.session_state.user_id
    name = st.session_state.name
    user_info = USERS[uid]
    patient_ids = user_info.get("patients", [])

    st.sidebar.title(f"👤 {name}")
    st.sidebar.caption(f"Caregiver ID: {uid}")
    if st.sidebar.button("Logout"):
        do_logout()
        st.rerun()

    if st.session_state.selected_patient:
        pid = st.session_state.selected_patient
        pname = USERS[pid]["name"]
        if st.sidebar.button("Back to Patient List"):
            st.session_state.selected_patient = None
            st.rerun()
        st.title(f"Patient: {pname} ({pid})")
        tab1, tab2, tab3, tab4 = st.tabs(["Reports", "Alerts", "Feedback", "AI Assistant"])
        with tab1:
            render_reports(pid)
        with tab2:
            render_alerts(pid)
        with tab3:
            render_send_feedback(uid, pid, label="patient")
        with tab4:
            render_chatbot(
                user_id=uid,
                user_name=name,
                role="Caregiver",
                patient_id=pid,
                authorized_patient_ids=patient_ids,
                current_page="Patient Details / AI Assistant",
                active_alert={"alerts": get_alerts(pid)},
            )
        return

    st.title("My Patients")
    st.divider()
    for pid in patient_ids:
        pname = USERS[pid]["name"]
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"### {pname}")
        col1.caption(f"Patient ID: {pid}")
        if col2.button("View Details", key=f"view_{pid}"):
            st.session_state.selected_patient = pid
            st.rerun()
        st.divider()

    st.subheader("🚨 Patient Alerts Overview")
    for pid in patient_ids:
        pname = USERS[pid]["name"]
        alerts = get_alerts(pid)
        high_alerts = [a for a in alerts if a["severity"] == "high"]
        if high_alerts:
            for a in high_alerts:
                source_tag = f" [{a.get('source', '')}]" if a.get("source") else ""
                st.error(f"**{pname} ({pid})** -- {a['time']}{source_tag}: {a['message']}")


# =====================================================================
# CLINICIAN FRONTEND
# =====================================================================

def clinician_dashboard():
    uid = st.session_state.user_id
    name = st.session_state.name
    user_info = USERS[uid]
    caregiver_ids = user_info.get("caregivers", [])

    st.sidebar.title(f"👤 {name}")
    st.sidebar.caption(f"Clinician ID: {uid}")
    if st.sidebar.button("Logout"):
        do_logout()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("🔍 Search Patient")
    search_pid = st.sidebar.text_input("Enter Patient ID").strip().upper()
    if st.sidebar.button("Search"):
        if search_pid in USERS and USERS[search_pid]["role"] == "Patient":
            st.session_state.selected_patient = search_pid
            st.session_state.selected_caregiver = None
            st.rerun()
        elif search_pid:
            st.sidebar.error("Patient not found.")

    if st.session_state.selected_patient:
        pid = st.session_state.selected_patient
        pname = USERS[pid]["name"]
        if st.sidebar.button("Back"):
            st.session_state.selected_patient = None
            st.session_state.selected_caregiver = None
            st.rerun()
        st.title(f"Patient: {pname} ({pid})")
        tab1, tab2, tab3, tab4 = st.tabs(["Reports", "Alerts", "Feedback", "AI Assistant"])
        with tab1:
            render_reports(pid)
        with tab2:
            render_alerts(pid)
        with tab3:
            render_send_feedback(uid, pid, label="patient")
            cg_id = USERS[pid].get("caregiver")
            if cg_id:
                render_send_feedback(uid, cg_id, label="caregiver")
        with tab4:
            authorized_patients = []
            for caregiver_id in caregiver_ids:
                authorized_patients.extend(
                    USERS.get(caregiver_id, {}).get("patients", [])
                )
            render_chatbot(
                user_id=uid,
                user_name=name,
                role="Clinician",
                patient_id=pid,
                authorized_patient_ids=authorized_patients,
                current_page="Patient Details / AI Assistant",
                active_alert={"alerts": get_alerts(pid)},
            )
        return

    if st.session_state.selected_caregiver:
        cg_id = st.session_state.selected_caregiver
        cg_name = USERS[cg_id]["name"]
        cg_patients = USERS[cg_id].get("patients", [])
        if st.sidebar.button("Back to Caregivers"):
            st.session_state.selected_caregiver = None
            st.rerun()
        st.title(f"Caregiver: {cg_name} ({cg_id})")
        render_send_feedback(uid, cg_id, label="caregiver")
        st.divider()
        st.subheader("Patients under this caregiver")
        for pid in cg_patients:
            pname = USERS[pid]["name"]
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"### {pname}")
            col1.caption(f"Patient ID: {pid}")
            if col2.button("View Patient", key=f"cview_{pid}"):
                st.session_state.selected_patient = pid
                st.rerun()
            st.divider()
        return

    st.title("My Caregivers")
    st.divider()
    for cg_id in caregiver_ids:
        cg_name = USERS[cg_id]["name"]
        cg_patients = USERS[cg_id].get("patients", [])
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"### {cg_name}")
        col1.caption(f"Caregiver ID: {cg_id} -- Patients: {len(cg_patients)}")
        if col2.button("View Caregiver", key=f"cg_{cg_id}"):
            st.session_state.selected_caregiver = cg_id
            st.rerun()
        st.divider()


# =====================================================================
# MAIN
# =====================================================================

init_session()

if not st.session_state.logged_in:
    login_page()
else:
    role = st.session_state.role
    if role == "Patient":
        patient_dashboard()
    elif role == "Caregiver":
        caregiver_dashboard()
    elif role == "Clinician":
        clinician_dashboard()
