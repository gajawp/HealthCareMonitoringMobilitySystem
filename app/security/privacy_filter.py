from __future__ import annotations

import copy
import re
from typing import Any

from app.security.exceptions import (
    PrivacyViolationError,
    UnsafeLLMContextError,
)


class PrivacyFilter:
    BLOCKED_KEYS = {
        "patient_id",
        "patientid",
        "patient_key",
        "patientkey",
        "user_id",
        "userid",
        "user_key",
        "authorized_patient_ids",
        "authorized_patient_keys",
        "patient_name",
        "full_name",
        "email",
        "phone",
        "address",
        "date_of_birth",
        "dob",
        "filename",
        "file_path",
        "source_path",
        "user_name",
        "display_name",
        "patient_name",
        "first_name",
        "last_name",
    }

    PATIENT_ID_PATTERN = re.compile(
        r"\bP\d{3,}\b",
        flags=re.IGNORECASE,
    )

    def sanitize_context(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        sanitized = self._sanitize_value(
            copy.deepcopy(context)
        )

        sanitized.setdefault(
            "patient_reference",
            "current_patient",
        )

        return sanitized

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized_dict: dict[str, Any] = {}

            for key, nested_value in value.items():
                normalized_key = key.lower().replace(
                    "-",
                    "_",
                )

                if normalized_key in self.BLOCKED_KEYS:
                    continue

                sanitized_dict[key] = self._sanitize_value(
                    nested_value
                )

            return sanitized_dict

        if isinstance(value, list):
            return [
                self._sanitize_value(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._sanitize_value(item)
                for item in value
            )

        if isinstance(value, str):
            return self.PATIENT_ID_PATTERN.sub(
                "[PATIENT_REFERENCE]",
                value,
            )

        return value

    def assert_safe_context(
        self,
        context: dict[str, Any],
        prohibited_values: list[str] | None = None,
    ) -> None:
        serialized_context = repr(context)

        if self.PATIENT_ID_PATTERN.search(
            serialized_context
        ):
            raise UnsafeLLMContextError(
                "A patient identifier was found in LLM context."
            )

        for value in prohibited_values or []:
            if value and value in serialized_context:
                raise UnsafeLLMContextError(
                    "A prohibited identifier was found "
                    "in LLM context."
                )

    def sanitize_answer(
        self,
        answer: str,
        prohibited_values: list[str] | None = None,
    ) -> str:
        sanitized_answer = self.PATIENT_ID_PATTERN.sub(
            "the selected patient",
            answer,
        )

        for value in prohibited_values or []:
            if value:
                sanitized_answer = sanitized_answer.replace(
                    value,
                    "[REDACTED]",
                )

        return sanitized_answer

    def assert_safe_answer(
        self,
        answer: str,
        prohibited_values: list[str] | None = None,
    ) -> None:
        if self.PATIENT_ID_PATTERN.search(answer):
            raise PrivacyViolationError(
                "The generated response contains "
                "a patient identifier."
            )

        for value in prohibited_values or []:
            if value and value in answer:
                raise PrivacyViolationError(
                    "The generated response contains "
                    "a prohibited identifier."
                )