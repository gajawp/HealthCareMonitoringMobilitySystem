from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import (
    AUDIT_LOG_PATH,
    ENABLE_AUDIT_LOGGING,
    PATIENT_KEY_PREFIX_LENGTH,
)


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    action: str
    user_reference: str
    role: str
    patient_reference: str | None
    allowed: bool
    request_id: str
    timestamp: str
    details: dict[str, Any]


class AuditLogger:
    """
    Writes privacy-safe JSON Lines audit records.

    The logger must not receive passwords, raw patient IDs, API keys,
    full prompts, full LLM contexts, or raw patient records.
    """

    BLOCKED_DETAIL_KEYS = {
        "password",
        "password_hash",
        "patient_id",
        "patientid",
        "patient_key",
        "patientkey",
        "user_id",
        "userid",
        "openai_api_key",
        "api_key",
        "patient_token_secret",
        "patient_encryption_key",
        "prompt",
        "raw_prompt",
        "context",
        "llm_context",
        "patient_data",
        "medical_notes",
    }

    def __init__(
        self,
        log_path: str | Path | None = None,
        *,
        enabled: bool = ENABLE_AUDIT_LOGGING,
        patient_reference_length: int = (
            PATIENT_KEY_PREFIX_LENGTH
        ),
    ) -> None:
        self._enabled = enabled
        self._patient_reference_length = (
            patient_reference_length
        )

        self._log_path = Path(
            log_path or AUDIT_LOG_PATH
        )

        self._write_lock = threading.Lock()

        if self._enabled:
            self._log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._logger = logging.getLogger(
            "healthcare_audit"
        )

    @staticmethod
    def create_request_id() -> str:
        return str(uuid.uuid4())

    def record(
        self,
        *,
        event_type: str,
        action: str,
        user_key: str,
        role: str,
        allowed: bool,
        request_id: str | None = None,
        patient_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        """
        Record one audit event and return its request ID.

        user_key and patient_key should already be opaque HMAC values.
        """

        final_request_id = (
            request_id
            or self.create_request_id()
        )

        if not self._enabled:
            return final_request_id

        safe_details = self._sanitize_details(
            details or {}
        )

        event = AuditEvent(
            event_type=event_type,
            action=action,
            user_reference=self._reference_from_key(
                user_key
            ),
            role=role,
            patient_reference=(
                self._reference_from_key(
                    patient_key
                )
                if patient_key
                else None
            ),
            allowed=allowed,
            request_id=final_request_id,
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            details=safe_details,
        )

        serialized_event = json.dumps(
            asdict(event),
            sort_keys=True,
            default=str,
        )

        with self._write_lock:
            with self._log_path.open(
                "a",
                encoding="utf-8",
            ) as audit_file:
                audit_file.write(
                    serialized_event + "\n"
                )

        return final_request_id

    def record_login(
        self,
        *,
        username: str,
        success: bool,
        role: str = "unknown",
        reason_code: str,
    ) -> str:
        """
        Record a login attempt without storing the raw username.

        A one-way SHA-256 reference is sufficient here because this value is
        for log correlation only, not patient-data retrieval.
        """

        username_reference = hashlib.sha256(
            username.strip().upper().encode(
                "utf-8"
            )
        ).hexdigest()

        return self.record(
            event_type="authentication",
            action="login",
            user_key=username_reference,
            role=role,
            allowed=success,
            details={
                "reason_code": reason_code,
            },
        )

    def _sanitize_details(
        self,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        safe_details: dict[str, Any] = {}

        for key, value in details.items():
            normalized_key = (
                str(key)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized_key in self.BLOCKED_DETAIL_KEYS:
                continue

            safe_details[str(key)] = (
                self._sanitize_detail_value(
                    value
                )
            )

        return safe_details

    def _sanitize_detail_value(
        self,
        value: Any,
    ) -> Any:
        if isinstance(value, dict):
            return self._sanitize_details(value)

        if isinstance(value, list):
            return [
                self._sanitize_detail_value(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self._sanitize_detail_value(item)
                for item in value
            ]

        if isinstance(value, str):
            sensitive_values = {
                os.getenv(
                    "OPENAI_API_KEY",
                    "",
                ),
                os.getenv(
                    "PATIENT_TOKEN_SECRET",
                    "",
                ),
                os.getenv(
                    "PATIENT_ENCRYPTION_KEY",
                    "",
                ),
            }

            safe_value = value

            for sensitive_value in sensitive_values:
                if sensitive_value:
                    safe_value = safe_value.replace(
                        sensitive_value,
                        "[REDACTED]",
                    )

            return safe_value

        return value

    def _reference_from_key(
        self,
        opaque_key: str,
    ) -> str:
        if not opaque_key:
            return "unknown"

        return opaque_key[
            : self._patient_reference_length
        ]