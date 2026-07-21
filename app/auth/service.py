from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass

import bcrypt

from app.auth.demo_users import DEMO_USERS
from app.auth.models import AuthenticatedUser, UserRole


@dataclass(frozen=True)
class AuthenticationResult:
    success: bool
    user: AuthenticatedUser | None
    reason: str


class AuthenticationService:
    """
    Authenticates users using bcrypt password hashes.

    The service intentionally returns the same public failure message for
    unknown users and invalid passwords to reduce username enumeration.
    """

    INVALID_CREDENTIALS_MESSAGE = "Invalid username or password."

    # Valid bcrypt hash for a known dummy password. It prevents unknown-user
    # requests from returning substantially faster than known-user requests.
    _DUMMY_PASSWORD_HASH = (
        b"$2b$12$C6UzMDM.H6dfI/f/IKcEe."
        b"uQxWMYQjF5uAZQp7B0n0Zpy2wK9hG"
    )

    def __init__(
        self,
        users: Mapping[str, Mapping[str, str]] | None = None,
        *,
        user_key_secret: str | None = None,
    ) -> None:
        self._users = {
            username.strip().upper(): dict(record)
            for username, record in (
                users or DEMO_USERS
            ).items()
        }

        configured_secret = (
            user_key_secret
            or os.getenv("PATIENT_TOKEN_SECRET", "")
        )

        if not configured_secret:
            raise RuntimeError(
                "PATIENT_TOKEN_SECRET is required to create user keys."
            )

        self._user_key_secret = configured_secret.encode("utf-8")

    def authenticate(
        self,
        username: str,
        password: str,
    ) -> AuthenticationResult:
        normalized_username = self._normalize_username(
            username
        )

        if not password:
            self._perform_dummy_password_check("")
            return AuthenticationResult(
                success=False,
                user=None,
                reason=self.INVALID_CREDENTIALS_MESSAGE,
            )

        user_record = self._users.get(
            normalized_username
        )

        if user_record is None:
            self._perform_dummy_password_check(
                password
            )

            return AuthenticationResult(
                success=False,
                user=None,
                reason=self.INVALID_CREDENTIALS_MESSAGE,
            )

        password_hash = user_record.get(
            "password_hash",
            "",
        ).encode("utf-8")

        try:
            password_matches = bcrypt.checkpw(
                password.encode("utf-8"),
                password_hash,
            )
        except ValueError:
            password_matches = False

        if not password_matches:
            return AuthenticationResult(
                success=False,
                user=None,
                reason=self.INVALID_CREDENTIALS_MESSAGE,
            )

        try:
            role = UserRole(
                user_record["role"].strip().lower()
            )
        except (
            KeyError,
            ValueError,
            AttributeError,
        ):
            return AuthenticationResult(
                success=False,
                user=None,
                reason=(
                    "The account configuration is invalid."
                ),
            )

        user = AuthenticatedUser(
            user_key=self._create_user_key(
                normalized_username
            ),
            username=normalized_username,
            display_name=user_record.get(
                "display_name",
                "Authenticated User",
            ),
            role=role,
        )

        return AuthenticationResult(
            success=True,
            user=user,
            reason="Authentication successful.",
        )

    def _create_user_key(
        self,
        username: str,
    ) -> str:
        """
        Create a stable HMAC-based user key.

        This is preferable to plain SHA-256 because the result depends on a
        secret unavailable to an attacker.
        """

        return hmac.new(
            self._user_key_secret,
            f"user:{username}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @classmethod
    def _perform_dummy_password_check(
        cls,
        password: str,
    ) -> None:
        try:
            bcrypt.checkpw(
                password.encode("utf-8"),
                cls._DUMMY_PASSWORD_HASH,
            )
        except ValueError:
            # Keep failure behavior uniform if the dummy hash is unsupported
            # by a particular bcrypt version.
            time.sleep(0.05)

    @staticmethod
    def _normalize_username(
        username: str,
    ) -> str:
        return username.strip().upper()