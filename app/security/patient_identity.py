from __future__ import annotations

import hashlib
import hmac
import os

from app.security.exceptions import SecurityConfigurationError


class PatientIdentityService:
    """Creates stable pseudonymous lookup keys for patient identifiers."""

    def __init__(self, secret: str | None = None) -> None:
        configured_secret = secret or os.getenv("PATIENT_TOKEN_SECRET", "")

        if not configured_secret:
            raise SecurityConfigurationError(
                "PATIENT_TOKEN_SECRET must be configured."
            )

        if len(configured_secret) < 32:
            raise SecurityConfigurationError(
                "PATIENT_TOKEN_SECRET must contain at least 32 characters."
            )

        self._secret = configured_secret.encode("utf-8")

    @staticmethod
    def normalize_patient_id(patient_id: str) -> str:
        if not patient_id or not patient_id.strip():
            raise ValueError("patient_id cannot be empty.")

        return patient_id.strip().upper()

    def create_patient_key(self, patient_id: str) -> str:
        normalized_patient_id = self.normalize_patient_id(patient_id)

        return hmac.new(
            self._secret,
            normalized_patient_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify_patient_key(
        self,
        patient_id: str,
        patient_key: str,
    ) -> bool:
        expected_key = self.create_patient_key(patient_id)

        return hmac.compare_digest(
            expected_key,
            patient_key,
        )