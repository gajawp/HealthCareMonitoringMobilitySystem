"""In-app messaging with email notifications."""
from __future__ import annotations

from .email_service import EmailResult, EmailService
from .message_store import MessageStore
from .service import MessagingService, SendOutcome

__all__ = [
    "EmailResult",
    "EmailService",
    "MessageStore",
    "MessagingService",
    "SendOutcome",
]
