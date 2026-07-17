from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken

from app.security.exceptions import SecurityConfigurationError


class PatientEncryptionService:
    """Encrypts patient identifiers for secure administrative storage."""

    def __init__(self, encryption_key: str | None = None) -> None:
        configured_key = (
            encryption_key
            or os.getenv("PATIENT_ENCRYPTION_KEY", "")
        )

        if not configured_key:
            raise SecurityConfigurationError(
                "PATIENT_ENCRYPTION_KEY must be configured."
            )

        try:
            self._fernet = Fernet(configured_key.encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise SecurityConfigurationError(
                "PATIENT_ENCRYPTION_KEY is invalid."
            ) from exc

    def encrypt_patient_id(self, patient_id: str) -> str:
        if not patient_id or not patient_id.strip():
            raise ValueError("patient_id cannot be empty.")

        normalized_patient_id = patient_id.strip().upper()

        return self._fernet.encrypt(
            normalized_patient_id.encode("utf-8")
        ).decode("utf-8")

    def decrypt_patient_id(self, encrypted_patient_id: str) -> str:
        if not encrypted_patient_id:
            raise ValueError(
                "encrypted_patient_id cannot be empty."
            )

        try:
            return self._fernet.decrypt(
                encrypted_patient_id.encode("utf-8")
            ).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError(
                "The encrypted patient identifier is invalid."
            ) from exc