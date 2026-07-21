from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.auth.models import AuthenticatedUser, UserRole
from app.security.patient_identity import PatientIdentityService


@dataclass(frozen=True)
class PatientAccessReference:
    """
    Privacy-safe patient option returned to the application and UI.

    patient_key:
        Opaque HMAC key used internally for repository access.

    display_alias:
        Non-identifying label displayed in the UI.
    """

    patient_key: str
    display_alias: str


@dataclass(frozen=True)
class AccessDecision:
    """Result of a patient-access authorization check."""

    allowed: bool
    patient_key: str | None
    display_alias: str | None
    reason: str


DEFAULT_DEMO_ACCESS_MAPPING: dict[str, tuple[str, ...]] = {
    "P1001": ("P1001",),
    "C2001": ("P1001", "P1002"),
    "D3001": ("P1001", "P1002", "P1003"),
}


class AuthorizationService:
    """
    Resolves and verifies user-to-patient access.

    Raw patient IDs are used only inside the trusted authorization layer to
    generate opaque patient keys. They must not be sent to the LLM or shown
    directly in the UI.
    """

    def __init__(
        self,
        identity_service: PatientIdentityService,
        access_mapping: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._identity_service = identity_service

        configured_mapping = (
            access_mapping
            if access_mapping is not None
            else DEFAULT_DEMO_ACCESS_MAPPING
        )

        self._access_mapping = self._normalize_access_mapping(
            configured_mapping
        )

    def list_authorized_patients(
        self,
        user: AuthenticatedUser,
    ) -> list[PatientAccessReference]:
        """
        Return privacy-safe patient references available to the user.

        The returned list contains opaque patient keys and display aliases,
        but does not expose raw patient IDs.
        """

        normalized_username = self._normalize_username(
            user.username
        )

        patient_ids = self._access_mapping.get(
            normalized_username,
            (),
        )

        references: list[PatientAccessReference] = []

        for index, patient_id in enumerate(
            patient_ids,
            start=1,
        ):
            patient_key = (
                self._identity_service.create_patient_key(
                    patient_id
                )
            )

            references.append(
                PatientAccessReference(
                    patient_key=patient_key,
                    display_alias=self._create_display_alias(
                        index=index,
                        role=user.role,
                    ),
                )
            )

        return references

    def authorize_patient_access(
        self,
        user: AuthenticatedUser,
        requested_patient_key: str,
    ) -> AccessDecision:
        """
        Verify that the requested opaque patient key is assigned to the user.

        Authorization is denied when:
        - the requested key is empty;
        - the user has no assigned patients;
        - the requested key does not match an assigned patient.
        """

        normalized_requested_key = (
            requested_patient_key.strip()
            if requested_patient_key
            else ""
        )

        if not normalized_requested_key:
            return AccessDecision(
                allowed=False,
                patient_key=None,
                display_alias=None,
                reason="No patient record was selected.",
            )

        authorized_patients = self.list_authorized_patients(
            user
        )

        if not authorized_patients:
            return AccessDecision(
                allowed=False,
                patient_key=None,
                display_alias=None,
                reason=(
                    "No patient records are assigned to this user."
                ),
            )

        for patient in authorized_patients:
            if hmac.compare_digest(
                patient.patient_key,
                normalized_requested_key,
            ):
                return AccessDecision(
                    allowed=True,
                    patient_key=patient.patient_key,
                    display_alias=patient.display_alias,
                    reason="Access permitted.",
                )

        return AccessDecision(
            allowed=False,
            patient_key=None,
            display_alias=None,
            reason=(
                "The selected patient is not assigned to this user."
            ),
        )

    def get_default_patient(
        self,
        user: AuthenticatedUser,
    ) -> PatientAccessReference | None:
        """
        Return the user's default authorized patient.

        For a patient, this represents their own record.
        For caregivers and clinicians, this is the first assigned patient.
        """

        authorized_patients = self.list_authorized_patients(
            user
        )

        if not authorized_patients:
            return None

        return authorized_patients[0]

    def user_has_patient_access(
        self,
        user: AuthenticatedUser,
    ) -> bool:
        """Return True when the user has at least one assigned patient."""

        return bool(
            self.list_authorized_patients(user)
        )

    @staticmethod
    def _normalize_username(username: str) -> str:
        if not username or not username.strip():
            raise ValueError("username cannot be empty.")

        return username.strip().upper()

    @staticmethod
    def _normalize_patient_id(patient_id: str) -> str:
        if not patient_id or not patient_id.strip():
            raise ValueError("patient_id cannot be empty.")

        return patient_id.strip().upper()

    @classmethod
    def _normalize_access_mapping(
        cls,
        access_mapping: Mapping[str, Sequence[str]],
    ) -> dict[str, tuple[str, ...]]:
        """
        Normalize usernames and patient IDs and remove duplicate assignments.
        """

        normalized_mapping: dict[str, tuple[str, ...]] = {}

        for username, patient_ids in access_mapping.items():
            normalized_username = cls._normalize_username(
                username
            )

            normalized_patient_ids: list[str] = []
            seen_patient_ids: set[str] = set()

            for patient_id in patient_ids:
                normalized_patient_id = (
                    cls._normalize_patient_id(
                        patient_id
                    )
                )

                if normalized_patient_id in seen_patient_ids:
                    continue

                seen_patient_ids.add(
                    normalized_patient_id
                )
                normalized_patient_ids.append(
                    normalized_patient_id
                )

            normalized_mapping[normalized_username] = tuple(
                normalized_patient_ids
            )

        return normalized_mapping

    @staticmethod
    def _create_display_alias(
        index: int,
        role: UserRole,
    ) -> str:
        if role == UserRole.PATIENT:
            return "My record"

        if role == UserRole.CAREGIVER:
            return f"Assigned patient {index}"

        if role == UserRole.CLINICIAN:
            return f"Clinical patient {index}"

        return f"Patient {index}"