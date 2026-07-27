"""Append-only JSON-lines store for in-app messages.

Each message is one JSON object per line in ``data/messages/messages.jsonl``.
The store powers the in-app inbox and conversation threads so recipients can
read their message history inside the application, independently of email.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESSAGES_PATH = PROJECT_ROOT / "data" / "messages" / "messages.jsonl"


def _key(value: Any) -> str:
    return str(value or "").strip().casefold()


class MessageStore:
    """Persist and query in-app messages."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(
            path or os.getenv("MESSAGES_PATH", str(DEFAULT_MESSAGES_PATH))
        )

    def add(self, record: dict[str, Any]) -> dict[str, Any]:
        """Append a message record, filling in id and timestamp if absent."""
        stored = dict(record)
        stored.setdefault("id", uuid.uuid4().hex)
        stored.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(stored, ensure_ascii=False) + "\n")

        return stored

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _sorted(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(records, key=lambda item: str(item.get("timestamp", "")))

    def inbox(self, user_id: str) -> list[dict[str, Any]]:
        """Messages received by ``user_id`` (oldest first)."""
        target = _key(user_id)
        return self._sorted(
            [m for m in self.all() if _key(m.get("recipient_id")) == target]
        )

    def sent(self, user_id: str) -> list[dict[str, Any]]:
        """Messages sent by ``user_id`` (oldest first)."""
        source = _key(user_id)
        return self._sorted(
            [m for m in self.all() if _key(m.get("sender_id")) == source]
        )

    def thread(self, user_a: str, user_b: str) -> list[dict[str, Any]]:
        """The full two-way conversation between two users (oldest first)."""
        pair = {_key(user_a), _key(user_b)}
        return self._sorted(
            [
                m
                for m in self.all()
                if {_key(m.get("sender_id")), _key(m.get("recipient_id"))} == pair
            ]
        )
