import json

from app.security.patient_identity import PatientIdentityService
from app.security.privacy_filter import PrivacyFilter


def test_patient_identifiers_do_not_reach_llm_prompt(
    monkeypatch,
):
    monkeypatch.setenv(
        "PATIENT_TOKEN_SECRET",
        "test-secret-value-with-more-than-32-characters",
    )

    identity_service = PatientIdentityService()
    privacy_filter = PrivacyFilter()

    patient_id = "P1001"
    patient_key = identity_service.create_patient_key(
        patient_id
    )

    raw_context = {
        "patient_id": patient_id,
        "patient_key": patient_key,
        "user_id": "C2001",
        "authorized_patient_ids": [
            "P1001",
            "P1002",
        ],
        "mobility_data": {
            "jerk_score": 0.36,
            "filename": (
                "session_20260101_P1001_reps.csv"
            ),
        },
    }

    safe_context = privacy_filter.sanitize_context(
        raw_context
    )

    prompt = json.dumps(safe_context)

    assert "P1001" not in prompt
    assert "P1002" not in prompt
    assert "C2001" not in prompt
    assert patient_key not in prompt

    assert "patient_id" not in prompt
    assert "patient_key" not in prompt
    assert "user_id" not in prompt
    assert "authorized_patient_ids" not in prompt
    assert "filename" not in prompt

    assert "0.36" in prompt
    assert "current_patient" in prompt