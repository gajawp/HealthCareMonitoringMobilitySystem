from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import streamlit as st

from app.auth.session import (
    LOGIN_ATTEMPTS_KEY,
    LOGIN_LOCKED_UNTIL_KEY,
)
from app.config import (
    LOGIN_LOCKOUT_MINUTES,
    MAX_LOGIN_ATTEMPTS,
)


@dataclass(frozen=True)
class LoginAttemptDecision:
    allowed: bool
    remaining_attempts: int
    locked_until: datetime | None
    reason: str


def check_login_allowed() -> LoginAttemptDecision:
    """Check whether the current session is locked out."""

    locked_until = st.session_state.get(
        LOGIN_LOCKED_UNTIL_KEY
    )

    if isinstance(locked_until, datetime):
        now = datetime.now(timezone.utc)

        if now < locked_until:
            return LoginAttemptDecision(
                allowed=False,
                remaining_attempts=0,
                locked_until=locked_until,
                reason=(
                    "Too many failed login attempts. "
                    "Please try again later."
                ),
            )

        st.session_state[LOGIN_LOCKED_UNTIL_KEY] = None
        st.session_state[LOGIN_ATTEMPTS_KEY] = 0

    attempts = int(
        st.session_state.get(
            LOGIN_ATTEMPTS_KEY,
            0,
        )
    )

    return LoginAttemptDecision(
        allowed=True,
        remaining_attempts=max(
            MAX_LOGIN_ATTEMPTS - attempts,
            0,
        ),
        locked_until=None,
        reason="Login permitted.",
    )


def record_failed_login() -> LoginAttemptDecision:
    """Increment failed attempts and lock the session when needed."""

    attempts = int(
        st.session_state.get(
            LOGIN_ATTEMPTS_KEY,
            0,
        )
    ) + 1

    st.session_state[LOGIN_ATTEMPTS_KEY] = attempts

    remaining = max(
        MAX_LOGIN_ATTEMPTS - attempts,
        0,
    )

    if attempts >= MAX_LOGIN_ATTEMPTS:
        locked_until = (
            datetime.now(timezone.utc)
            + timedelta(
                minutes=LOGIN_LOCKOUT_MINUTES
            )
        )

        st.session_state[
            LOGIN_LOCKED_UNTIL_KEY
        ] = locked_until

        return LoginAttemptDecision(
            allowed=False,
            remaining_attempts=0,
            locked_until=locked_until,
            reason=(
                "Too many failed login attempts. "
                "Please try again later."
            ),
        )

    return LoginAttemptDecision(
        allowed=True,
        remaining_attempts=remaining,
        locked_until=None,
        reason="Invalid username or password.",
    )


def record_successful_login() -> None:
    """Reset failed-login state after successful authentication."""

    st.session_state[LOGIN_ATTEMPTS_KEY] = 0
    st.session_state[LOGIN_LOCKED_UNTIL_KEY] = None