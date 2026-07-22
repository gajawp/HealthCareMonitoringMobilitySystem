from __future__ import annotations

from pathlib import Path

from app.messaging import EmailService, MessageStore, MessagingService


def _messaging(tmp_path: Path) -> MessagingService:
    return MessagingService(
        email_service=EmailService(
            backend="console",
            log_dir=tmp_path / "emails",
            default_from="no-reply@test.local",
        ),
        message_store=MessageStore(path=tmp_path / "messages.jsonl"),
    )


def test_email_console_backend_writes_file_and_preview(tmp_path: Path):
    service = EmailService(backend="console", log_dir=tmp_path / "emails")

    result = service.send(
        to_email="d3001@example.com",
        subject="Test subject",
        body="Hello there",
    )

    assert result.success is True
    assert result.backend == "console"
    assert Path(result.detail).exists()
    assert "Test subject" in result.preview
    assert "Hello there" in result.preview


def test_email_missing_recipient_is_reported_not_raised(tmp_path: Path):
    service = EmailService(backend="console", log_dir=tmp_path / "emails")
    result = service.send(to_email=None, subject="x", body="y")
    assert result.success is False
    assert "No recipient" in result.detail


def test_message_store_inbox_sent_and_thread(tmp_path: Path):
    store = MessageStore(path=tmp_path / "messages.jsonl")

    store.add({"sender_id": "P1001", "recipient_id": "D3001", "body": "hi doc"})
    store.add({"sender_id": "D3001", "recipient_id": "P1001", "body": "hi patient"})
    store.add({"sender_id": "P1002", "recipient_id": "D3001", "body": "unrelated"})

    assert len(store.inbox("D3001")) == 2
    assert len(store.inbox("P1001")) == 1
    assert len(store.sent("P1001")) == 1

    thread = store.thread("P1001", "D3001")
    assert [m["body"] for m in thread] == ["hi doc", "hi patient"]


def test_send_message_persists_and_notifies(tmp_path: Path):
    service = _messaging(tmp_path)

    outcome = service.send_message(
        sender_id="P1001",
        sender_name="John Smith",
        sender_role="Patient",
        recipient_id="D3001",
        recipient_name="Dr. Sarah Lee",
        recipient_role="Clinician",
        recipient_email="d3001@example.com",
        body="Is my knee angle improving?",
        message_type="Question",
    )

    # Email notification delivered (console) ...
    assert outcome.email.success is True
    assert "John Smith" in outcome.email.preview

    # ... and the message is retrievable in the recipient's inbox.
    inbox = service.inbox("D3001")
    assert len(inbox) == 1
    assert inbox[0]["body"] == "Is my knee angle improving?"
    assert inbox[0]["email_sent"] is True


def test_send_message_saved_even_when_email_fails(tmp_path: Path):
    service = _messaging(tmp_path)

    outcome = service.send_message(
        sender_id="D3001",
        sender_name="Dr. Sarah Lee",
        sender_role="Clinician",
        recipient_id="P1001",
        recipient_name="John Smith",
        recipient_role="Patient",
        recipient_email=None,  # no address on file -> email fails
        body="Please keep up the exercises.",
        message_type="Feedback",
    )

    assert outcome.email.success is False
    # The in-app message is still stored so the inbox stays reliable.
    assert len(service.inbox("P1001")) == 1
