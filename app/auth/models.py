from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UserRole(str, Enum):
    PATIENT = "patient"
    CAREGIVER = "caregiver"
    CLINICIAN = "clinician"


@dataclass(frozen=True)
class AuthenticatedUser:
    """
    Represents a successfully authenticated application user.

    Passwords, patient IDs, encryption keys, and API keys must never be stored
    in this object.
    """

    user_key: str
    username: str
    display_name: str
    role: UserRole