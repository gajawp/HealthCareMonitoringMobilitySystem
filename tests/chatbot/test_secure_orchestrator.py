from unittest.mock import Mock

from app.security.privacy_filter import PrivacyFilter


def test_real_chatbot_context_contains_no_identifiers():
    privacy_filter = PrivacyFilter()

    raw_mobility_data = {
        "patient_id": "P1001",
        "patient_key": "opaque-patient-key",
        "filename": "session_20260101_P1001_reps.csv",
        "jerk_score": {
            "mean": 0.36,
            "latest": 0.43,
        },
    }

    safe_context = privacy_filter.sanitize_context(
        {
            "role": "patient",
            "patient_reference": "current_patient",
            "mobility_data": raw_mobility_data,
            "clinical_knowledge": [],
        }
    )

    captured_prompt = repr(safe_context)

    assert "P1001" not in captured_prompt
    assert "opaque-patient-key" not in captured_prompt
    assert "patient_id" not in captured_prompt
    assert "patient_key" not in captured_prompt
    assert "filename" not in captured_prompt

    assert "0.36" in captured_prompt
    assert "0.43" in captured_prompt
    assert "current_patient" in captured_prompt