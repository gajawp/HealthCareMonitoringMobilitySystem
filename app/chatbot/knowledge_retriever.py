from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = (
    PROJECT_ROOT / "knowledge_base" / "generated" / "knowledge_chunks.json"
)


def tokenize(text: str) -> list[str]:
    """Return normalized tokens for lightweight local retrieval."""
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 2
    ]


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    """Calculate cosine similarity between token-frequency vectors."""
    if not left or not right:
        return 0.0

    dot_product = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


class ClinicalKnowledgeRetriever:
    """Search locally generated knowledge chunks without an external vector DB."""

    def __init__(self, chunks_path: str | Path = DEFAULT_CHUNKS_PATH) -> None:
        self.chunks_path = Path(chunks_path)
        self.chunks = self._load_chunks()

    def _load_chunks(self) -> list[dict[str, Any]]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Knowledge chunks were not found at {self.chunks_path}. "
                "Run: python scripts/build_knowledge_base.py"
            )

        with self.chunks_path.open("r", encoding="utf-8") as file:
            content = json.load(file)

        if not isinstance(content, list):
            raise ValueError("knowledge_chunks.json must contain a JSON list.")

        return content

    def search(
        self,
        query: str,
        top_k: int = 5,
        exercise_id: str | None = None,
        approved_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return top matching chunks.

        Args:
            query: Natural-language search text.
            top_k: Maximum number of results.
            exercise_id: Optional exercise filter.
            approved_only: When True, exclude records not marked approved.
        """
        if not query.strip():
            return []

        query_vector = Counter(tokenize(query))
        scored: list[tuple[float, dict[str, Any]]] = []

        for chunk in self.chunks:
            if exercise_id and chunk.get("exercise_id") != exercise_id:
                continue

            if approved_only and chunk.get("clinical_review_status") != "approved":
                continue

            candidate = " ".join(
                str(chunk.get(field, ""))
                for field in ("title", "text", "exercise_id", "metric_id", "sensor_id")
            )
            score = cosine_similarity(query_vector, Counter(tokenize(candidate)))

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            {**chunk, "retrieval_score": round(score, 4)}
            for score, chunk in scored[:max(top_k, 0)]
        ]

    def get_exercise(self, exercise_id: str) -> list[dict[str, Any]]:
        """Return all chunks for one exercise."""
        return [
            chunk for chunk in self.chunks
            if chunk.get("exercise_id") == exercise_id
        ]
