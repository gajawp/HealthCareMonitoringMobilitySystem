"""Configurable email-notification backend for in-app messages.

The backend is selected by the ``EMAIL_BACKEND`` environment variable:

- ``console`` (default): no network access. The fully formatted email is
  written to ``logs/emails/`` and returned as a preview. Safe for demos and
  automated tests, and requires no credentials.
- ``smtp``: send through an SMTP server using ``SMTP_HOST``, ``SMTP_PORT``,
  ``SMTP_USERNAME``, ``SMTP_PASSWORD`` and ``SMTP_FROM``.
- ``ses``: send through AWS SES using ``SES_FROM`` and ``AWS_REGION`` with the
  standard AWS credential chain.

Switching to real delivery is therefore a configuration change only; no code
in the application has to change.
"""
from __future__ import annotations

import os
import re
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMAIL_LOG_DIR = PROJECT_ROOT / "logs" / "emails"


@dataclass(frozen=True)
class EmailResult:
    """Outcome of a single email-send attempt."""

    success: bool
    backend: str
    detail: str
    preview: str | None = None


class EmailService:
    """Send notification emails through a configurable backend."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        log_dir: str | Path | None = None,
        default_from: str | None = None,
    ) -> None:
        self.backend = (
            backend or os.getenv("EMAIL_BACKEND", "console")
        ).strip().lower()
        self.log_dir = Path(
            log_dir or os.getenv("EMAIL_LOG_DIR", str(DEFAULT_EMAIL_LOG_DIR))
        )
        self.default_from = (
            default_from
            or os.getenv("SMTP_FROM")
            or os.getenv("SES_FROM")
            or os.getenv("EMAIL_FROM")
            or "no-reply@healthcare-mobility.local"
        )

    def send(
        self,
        *,
        to_email: str | None,
        subject: str,
        body: str,
        from_email: str | None = None,
    ) -> EmailResult:
        """Send one email, never raising. Failures are returned as results."""
        sender = from_email or self.default_from

        if not to_email:
            return EmailResult(
                success=False,
                backend=self.backend,
                detail="No recipient email address is on file.",
            )

        try:
            if self.backend == "smtp":
                return self._send_smtp(sender, to_email, subject, body)
            if self.backend == "ses":
                return self._send_ses(sender, to_email, subject, body)
            return self._send_console(sender, to_email, subject, body)
        except Exception as exc:  # pragma: no cover - defensive
            # A notification failure must never break the in-app message flow.
            return EmailResult(
                success=False,
                backend=self.backend,
                detail=f"{type(exc).__name__}: {exc}",
            )

    def _compose(
        self,
        sender: str,
        to_email: str,
        subject: str,
        body: str,
    ) -> str:
        stamp = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
        return (
            f"From: {sender}\n"
            f"To: {to_email}\n"
            f"Subject: {subject}\n"
            f"Date: {stamp}\n"
            f"\n"
            f"{body}\n"
        )

    def _send_console(
        self,
        sender: str,
        to_email: str,
        subject: str,
        body: str,
    ) -> EmailResult:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        composed = self._compose(sender, to_email, subject, body)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        safe_recipient = re.sub(r"[^A-Za-z0-9_.-]", "_", to_email)
        path = self.log_dir / f"{stamp}_{safe_recipient}.txt"
        path.write_text(composed, encoding="utf-8")
        return EmailResult(
            success=True,
            backend="console",
            detail=str(path),
            preview=composed,
        )

    def _send_smtp(
        self,
        sender: str,
        to_email: str,
        subject: str,
        body: str,
    ) -> EmailResult:
        host = os.getenv("SMTP_HOST")
        if not host:
            return EmailResult(
                success=False,
                backend="smtp",
                detail="SMTP_HOST is not configured.",
            )

        port = int(os.getenv("SMTP_PORT", "587"))
        username = os.getenv("SMTP_USERNAME")
        password = os.getenv("SMTP_PASSWORD")

        message = EmailMessage()
        message["From"] = sender
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(body)

        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(message)

        return EmailResult(
            success=True,
            backend="smtp",
            detail=f"Sent via {host}:{port} to {to_email}",
        )

    def _send_ses(
        self,
        sender: str,
        to_email: str,
        subject: str,
        body: str,
    ) -> EmailResult:
        import boto3  # imported lazily so the app runs without AWS libs

        region = os.getenv("AWS_REGION", "us-west-2")
        client = boto3.client("ses", region_name=region)
        response = client.send_email(
            Source=sender,
            Destination={"ToAddresses": [to_email]},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Text": {"Data": body}},
            },
        )
        return EmailResult(
            success=True,
            backend="ses",
            detail=f"SES MessageId: {response.get('MessageId')}",
        )
