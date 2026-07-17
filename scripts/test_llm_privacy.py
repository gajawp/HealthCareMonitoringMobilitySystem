from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.security.patient_identity import PatientIdentityService
from app.security.privacy_filter import PrivacyFilter


class FakeLLMClient:
    """
    Fake LLM client used to inspect exactly what would be sent
    to an external language model.
    """

    def __init__(self) -> None:
        self.captured_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.captured_prompt = prompt
        return (
            "The current patient's mean jerk score is 0.36. "
            "This metric describes movement smoothness."
        )


def build_llm_prompt(
    question: str,
    context: dict,
) -> str:
    return (
        "Answer using only the supplied de-identified mobility data.\n\n"
        f"Question:\n{question}\n\n"
        "De-identified context:\n"
        f"{json.dumps(context, indent=2)}"
    )


def main() -> None:
    identity_service = PatientIdentityService()
    privacy_filter = PrivacyFilter()
    fake_llm = FakeLLMClient()

    raw_patient_id = "P1001"

    patient_key = identity_service.create_patient_key(
        raw_patient_id
    )

    # This represents data received internally before sanitization.
    raw_context = {
        "patient_id": raw_patient_id,
        "patient_key": patient_key,
        "user_id": "P1001",
        "authorized_patient_ids": [
            "P1001",
            "P1002",
        ],
        "mobility_data": {
            "jerk_score": {
                "mean": 0.36,
                "latest": 0.43,
                "minimum": 0.28,
            },
            "filename": (
                "session_20260101_100000_P1001_reps.csv"
            ),
        },
    }

    print("\n1. Raw backend context")
    print("----------------------")
    print(json.dumps(raw_context, indent=2))

    # Remove private identifiers before building the LLM prompt.
    safe_context = privacy_filter.sanitize_context(
        raw_context
    )

    privacy_filter.assert_safe_context(
        safe_context,
        prohibited_values=[
            raw_patient_id,
            patient_key,
            "P1002",
        ],
    )

    print("\n2. Sanitized LLM context")
    print("-------------------------")
    print(json.dumps(safe_context, indent=2))

    prompt = build_llm_prompt(
        question="What does my jerk score mean?",
        context=safe_context,
    )

    response = fake_llm.generate(prompt)

    captured_prompt = fake_llm.captured_prompt

    if captured_prompt is None:
        raise AssertionError(
            "The fake LLM did not capture a prompt."
        )

    print("\n3. Exact prompt sent to the fake LLM")
    print("-------------------------------------")
    print(captured_prompt)

    # Critical privacy assertions.
    assert raw_patient_id not in captured_prompt
    assert "P1002" not in captured_prompt
    assert patient_key not in captured_prompt

    assert "patient_id" not in captured_prompt
    assert "patient_key" not in captured_prompt
    assert "user_id" not in captured_prompt
    assert "authorized_patient_ids" not in captured_prompt
    assert "filename" not in captured_prompt

    # Required mobility values must remain.
    assert "0.36" in captured_prompt
    assert "0.43" in captured_prompt

    safe_response = privacy_filter.sanitize_answer(
        response,
        prohibited_values=[
            raw_patient_id,
            patient_key,
            "P1002",
        ],
    )

    privacy_filter.assert_safe_answer(
        safe_response,
        prohibited_values=[
            raw_patient_id,
            patient_key,
            "P1002",
        ],
    )

    print("\n4. Fake LLM response")
    print("---------------------")
    print(safe_response)

    print("\nLLM PRIVACY TEST: PASSED")
    print("The LLM received mobility measurements.")
    print("The LLM did not receive the raw patient ID.")
    print("The LLM did not receive the patient key.")
    print("The LLM did not receive authorization data.")
    print("The LLM did not receive the source filename.")


if __name__ == "__main__":
    main()