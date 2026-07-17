from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

# Important: load environment variables before importing app.config.
loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)

if not loaded:
    raise RuntimeError(
        f"Could not load the environment file: {ENV_PATH}"
    )

# Import configuration only after .env has been loaded.
from app.config import load_app_settings
from app.security.encryption import PatientEncryptionService
from app.security.patient_identity import PatientIdentityService
from app.security.privacy_filter import PrivacyFilter


def main() -> None:
    token_secret_exists = bool(
        os.getenv("PATIENT_TOKEN_SECRET")
    )
    encryption_key_exists = bool(
        os.getenv("PATIENT_ENCRYPTION_KEY")
    )

    print("Environment file:", ENV_PATH)
    print("PATIENT_TOKEN_SECRET loaded:", token_secret_exists)
    print("PATIENT_ENCRYPTION_KEY loaded:", encryption_key_exists)

    assert token_secret_exists, (
        "PATIENT_TOKEN_SECRET was not loaded from .env"
    )

    assert encryption_key_exists, (
        "PATIENT_ENCRYPTION_KEY was not loaded from .env"
    )

    settings = load_app_settings()

    identity_service = PatientIdentityService(
        settings.security.patient_token_secret
    )

    encryption_service = PatientEncryptionService(
        settings.security.patient_encryption_key
    )

    privacy_filter = PrivacyFilter()

    patient_key = identity_service.create_patient_key(
        "P1001"
    )

    same_patient_key = (
        identity_service.create_patient_key("P1001")
    )

    different_patient_key = (
        identity_service.create_patient_key("P1002")
    )

    assert patient_key == same_patient_key
    assert patient_key != different_patient_key
    assert "P1001" not in patient_key
    assert len(patient_key) == 64

    encrypted_patient_id = (
        encryption_service.encrypt_patient_id(
            "P1001"
        )
    )

    decrypted_patient_id = (
        encryption_service.decrypt_patient_id(
            encrypted_patient_id
        )
    )

    assert encrypted_patient_id != "P1001"
    assert decrypted_patient_id == "P1001"

    raw_context = {
        "patient_id": "P1001",
        "patient_key": patient_key,
        "user_id": "C2001",
        "mobility_data": {
            "jerk_score": 0.36,
            "filename": (
                "session_20260101_100000_P1001_reps.csv"
            ),
        },
    }

    safe_context = privacy_filter.sanitize_context(
        raw_context
    )

    serialized = repr(safe_context)

    assert "P1001" not in serialized
    assert "C2001" not in serialized
    assert patient_key not in serialized
    assert "patient_id" not in serialized
    assert "patient_key" not in serialized
    assert "user_id" not in serialized
    assert "filename" not in serialized
    assert (
        safe_context["mobility_data"]["jerk_score"]
        == 0.36
    )

    print("Patient key generation: PASSED")
    print("Patient encryption/decryption: PASSED")
    print("Privacy filtering: PASSED")
    print("All security configuration checks passed.")


if __name__ == "__main__":
    main()