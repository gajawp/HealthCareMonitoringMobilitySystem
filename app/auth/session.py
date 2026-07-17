from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

from app.auth.models import AuthenticatedUser, UserRole
from app.config import SESSION_TIMEOUT_MINUTES


SESSION_USER_KEY = "authenticated_user"
SESSION_EXPIRY_KEY = "session_expires_at"
SESSION_LAST_ACTIVITY_KEY = "session_last_activity"
ACTIVE_PATIENT_KEY = "active_patient_key"
LOGIN_ATTEMPTS_KEY = "login_attempts"
LOGIN_LOCKED_UNTIL_KEY = "login_locked_until"


def initialize_session_state() -> None:
    """Initialize security-related Streamlit session values."""

    defaults: dict[str, Any] = {
        SESSION_USER_KEY: None,
        SESSION_EXPIRY_KEY: None,
        SESSION_LAST_ACTIVITY_KEY: None,
        ACTIVE_PATIENT_KEY: None,
        LOGIN_ATTEMPTS_KEY: 0,
        LOGIN_LOCKED_UNTIL_KEY: None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_authenticated_user(
    user: AuthenticatedUser,
    *,
    timeout_minutes: int = SESSION_TIMEOUT_MINUTES,
) -> None:
    """
    Store only minimum identity information in Streamlit session state.

    Passwords, secrets, patient IDs, authorization mappings, and complete
    patient records are deliberately excluded.
    """

    now = datetime.now(timezone.utc)

    st.session_state[SESSION_USER_KEY] = {
        "user_key": user.user_key,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role.value,
    }

    st.session_state[SESSION_LAST_ACTIVITY_KEY] = now
    st.session_state[SESSION_EXPIRY_KEY] = (
        now + timedelta(minutes=timeout_minutes)
    )

    st.session_state[LOGIN_ATTEMPTS_KEY] = 0
    st.session_state[LOGIN_LOCKED_UNTIL_KEY] = None


def get_authenticated_user() -> AuthenticatedUser | None:
    """
    Return the authenticated user when the session remains valid.

    Expired or malformed sessions are cleared automatically.
    """

    user_data = st.session_state.get(
        SESSION_USER_KEY
    )

    expiry = st.session_state.get(
        SESSION_EXPIRY_KEY
    )

    if not isinstance(user_data, dict):
        return None

    if not isinstance(expiry, datetime):
        logout()
        return None

    if datetime.now(timezone.utc) >= expiry:
        logout()
        return None

    try:
        return AuthenticatedUser(
            user_key=str(user_data["user_key"]),
            username=str(user_data["username"]),
            display_name=str(
                user_data["display_name"]
            ),
            role=UserRole(
                str(user_data["role"])
            ),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        logout()
        return None


def is_authenticated() -> bool:
    """Return True when a valid authenticated session exists."""

    return get_authenticated_user() is not None


def refresh_session(
    *,
    timeout_minutes: int = SESSION_TIMEOUT_MINUTES,
) -> bool:
    """
    Extend the authenticated session after valid user activity.

    Returns False if no valid authenticated session exists.
    """

    user = get_authenticated_user()

    if user is None:
        return False

    now = datetime.now(timezone.utc)

    st.session_state[SESSION_LAST_ACTIVITY_KEY] = now
    st.session_state[SESSION_EXPIRY_KEY] = (
        now + timedelta(minutes=timeout_minutes)
    )

    return True


def set_active_patient_key(
    patient_key: str | None,
) -> None:
    """
    Store the selected opaque patient key.

    The real patient ID must not be stored here.
    """

    if patient_key is None:
        st.session_state[ACTIVE_PATIENT_KEY] = None
        return

    normalized_key = patient_key.strip()

    if not normalized_key:
        raise ValueError(
            "patient_key cannot be empty."
        )

    st.session_state[ACTIVE_PATIENT_KEY] = (
        normalized_key
    )


def get_active_patient_key() -> str | None:
    """Return the selected opaque patient key."""

    value = st.session_state.get(
        ACTIVE_PATIENT_KEY
    )

    if not isinstance(value, str):
        return None

    normalized_value = value.strip()

    return normalized_value or None


def logout() -> None:
    """
    Clear all authenticated and patient-selection session values.

    Chat history may also be cleared here when it could contain sensitive
    mobility information.
    """

    keys_to_clear = {
        SESSION_USER_KEY,
        SESSION_EXPIRY_KEY,
        SESSION_LAST_ACTIVITY_KEY,
        ACTIVE_PATIENT_KEY,
        "chat_messages",
        "chat_history",
        "selected_patient",
        "selected_patient_id",
        "authorized_patient_ids",
    }

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state[SESSION_USER_KEY] = None
    st.session_state[SESSION_EXPIRY_KEY] = None
    st.session_state[SESSION_LAST_ACTIVITY_KEY] = None
    st.session_state[ACTIVE_PATIENT_KEY] = None