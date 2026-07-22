"""Coordinate in-app message persistence with email notification.

``MessagingService.send_message`` records the message in the message store and
sends an email notification to the recipient. The message is always persisted,
even if the email notification fails, so the in-app inbox stays reliable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .email_service import EmailResult, EmailService
from .message_store import MessageStore


APP_NAME = "Healthcare Mobility Monitoring System"


@dataclass
class SendOutcome:
    """Combined result of persisting a message and notifying by email."""

    record: dict[str, Any]
    email: EmailResult


class MessagingService:
    """Send and retrieve in-app messages with email notifications."""

    def __init__(
        self,
        *,
        email_service: EmailService | None = None,
        message_store: MessageStore | None = None,
        app_name: str = APP_NAME,
    ) -> None:
        self.email_service = email_service or EmailService()
        self.message_store = message_store or MessageStore()
        self.app_name = app_name

    def _build_email_body(
        self,
        *,
        sender_name: str,
        sender_role: str,
        recipient_name: str,
        message_type: str,
        body: str,
    ) -> str:
        kind = message_type.strip().lower() or "message"
        return (
            f"Hello {recipient_name},\n\n"
            f"You have received a new {kind} from {sender_name} "
            f"({sender_role}) in the {self.app_name}.\n\n"
            "-------------------------------------------\n"
            f"{body}\n"
            "-------------------------------------------\n\n"
            "Log in to the dashboard to read the message and reply.\n"
        )

    def send_message(
        self,
        *,
        sender_id: str,
        sender_name: str,
        sender_role: str,
        recipient_id: str,
        recipient_name: str,
        recipient_role: str,
        recipient_email: str | None,
        body: str,
        message_type: str = "Message",
    ) -> SendOutcome:
        subject = (
            f"[{self.app_name}] New {message_type.lower()} from "
            f"{sender_name} ({sender_role})"
        )
        email_body = self._build_email_body(
            sender_name=sender_name,
            sender_role=sender_role,
            recipient_name=recipient_name,
            message_type=message_type,
            body=body,
        )

        email_result = self.email_service.send(
            to_email=recipient_email,
            subject=subject,
            body=email_body,
        )

        record = self.message_store.add(
            {
                "sender_id": sender_id,
                "sender_name": sender_name,
                "sender_role": sender_role,
                "recipient_id": recipient_id,
                "recipient_name": recipient_name,
                "recipient_role": recipient_role,
                "recipient_email": recipient_email,
                "message_type": message_type,
                "body": body,
                "email_backend": email_result.backend,
                "email_sent": email_result.success,
                "email_detail": email_result.detail,
            }
        )

        return SendOutcome(record=record, email=email_result)

    def inbox(self, user_id: str) -> list[dict[str, Any]]:
        return self.message_store.inbox(user_id)

    def sent(self, user_id: str) -> list[dict[str, Any]]:
        return self.message_store.sent(user_id)

    def thread(self, user_a: str, user_b: str) -> list[dict[str, Any]]:
        return self.message_store.thread(user_a, user_b)
