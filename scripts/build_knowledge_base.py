from __future__ import annotations

import json
from pathlib import Path
from typing import Any


APPLICATION_DIR = Path(__file__).resolve().parent.parent
KB_DIR = APPLICATION_DIR / "knowledge_base"
OUTPUT_PATH = KB_DIR / "generated" / "knowledge_chunks.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_chunk(
    *,
    chunk_id: str,
    document_type: str,
    title: str,
    text: str,
    sources: list[str] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_type": document_type,
        "title": title,
        "text": text.strip(),
        "sources": sources or [],
        **metadata,
    }


def exercise_to_chunks(exercise: dict[str, Any]) -> list[dict[str, Any]]:
    exercise_id = exercise["exercise_id"]
    name = exercise["exercise_name"]
    sources = exercise.get("evidence_sources", [])
    review_status = exercise.get("clinical_review_status", "pending")

    chunks = [
        make_chunk(
            chunk_id=f"{exercise_id}_overview",
            document_type="exercise_overview",
            exercise_id=exercise_id,
            title=f"{name}: Overview",
            text=(
                f"Exercise: {name}. "
                f"Category: {exercise.get('category', '')}. "
                f"Purpose: {'; '.join(exercise.get('purpose', []))}. "
                f"Starting position: {'; '.join(exercise.get('starting_position', []))}."
            ),
            sources=sources,
            clinical_review_status=review_status,
        ),
        make_chunk(
            chunk_id=f"{exercise_id}_phases",
            document_type="movement_phases",
            exercise_id=exercise_id,
            title=f"{name}: Movement phases",
            text="; ".join(
                f"{phase['phase']}: {phase['description']}"
                for phase in exercise.get("movement_phases", [])
            ),
            sources=sources,
            clinical_review_status=review_status,
        ),
        make_chunk(
            chunk_id=f"{exercise_id}_guidance",
            document_type="patient_guidance",
            exercise_id=exercise_id,
            title=f"{name}: General guidance",
            text="; ".join(exercise.get("patient_friendly_guidance", [])),
            sources=sources,
            clinical_review_status=review_status,
        ),
        make_chunk(
            chunk_id=f"{exercise_id}_safety",
            document_type="exercise_safety",
            exercise_id=exercise_id,
            title=f"{name}: Safety limits",
            text="; ".join(exercise.get("safety_notes", [])),
            sources=sources,
            clinical_review_status=review_status,
        ),
    ]

    for index, error in enumerate(exercise.get("common_execution_errors", []), start=1):
        chunks.append(
            make_chunk(
                chunk_id=f"{exercise_id}_error_{index}",
                document_type="execution_error",
                exercise_id=exercise_id,
                title=f"{name}: {error['error']}",
                text=(
                    f"Possible execution issue: {error['error']}. "
                    f"Observable signals: {'; '.join(error.get('observable_signals', []))}. "
                    f"Safe explanation: {error.get('safe_explanation', '')}"
                ),
                sources=sources,
                clinical_review_status=review_status,
            )
        )

    return chunks


def records_to_chunks(
    records: list[dict[str, Any]],
    *,
    identifier_field: str,
    document_type: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for record in records:
        identifier = str(record[identifier_field])
        title = str(
            record.get("display_name")
            or record.get("title")
            or identifier
        )
        text_parts = []

        for key, value in record.items():
            if key == identifier_field or value is None:
                continue
            if isinstance(value, list):
                rendered = "; ".join(str(item) for item in value)
            elif isinstance(value, dict):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)
            text_parts.append(f"{key.replace('_', ' ')}: {rendered}")

        chunks.append(
            make_chunk(
                chunk_id=f"{document_type}_{identifier}",
                document_type=document_type,
                title=title,
                text=". ".join(text_parts),
                sources=record.get("evidence_sources", []),
                **{identifier_field: identifier},
            )
        )

    return chunks


def build_knowledge_base() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for path in sorted((KB_DIR / "exercises").glob("*.json")):
        chunks.extend(exercise_to_chunks(load_json(path)))

    mobility_metrics = load_json(KB_DIR / "metrics" / "mobility_metrics.json")
    chunks.extend(
        records_to_chunks(
            mobility_metrics,
            identifier_field="metric_id",
            document_type="metric_definition",
        )
    )

    sensor_metrics = load_json(KB_DIR / "metrics" / "sensor_metrics.json")
    chunks.extend(
        records_to_chunks(
            sensor_metrics,
            identifier_field="sensor_id",
            document_type="sensor_definition",
        )
    )

    rom = load_json(KB_DIR / "metrics" / "range_of_motion.json")
    chunks.extend(
        records_to_chunks(
            [rom],
            identifier_field="concept_id",
            document_type="clinical_concept",
        )
    )

    boundaries = load_json(KB_DIR / "safety" / "chatbot_boundaries.json")
    chunks.append(
        make_chunk(
            chunk_id="safety_chatbot_boundaries",
            document_type="safety_policy",
            title="Chatbot boundaries",
            text=json.dumps(boundaries, ensure_ascii=False),
        )
    )

    escalation = load_json(KB_DIR / "safety" / "escalation_rules.json")
    chunks.append(
        make_chunk(
            chunk_id="safety_escalation_rules",
            document_type="safety_policy",
            title="Escalation rules",
            text=json.dumps(escalation, ensure_ascii=False),
        )
    )

    sources = load_json(KB_DIR / "sources" / "source_registry.json")
    chunks.extend(
        records_to_chunks(
            sources,
            identifier_field="source_id",
            document_type="source_metadata",
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    return chunks


if __name__ == "__main__":
    generated = build_knowledge_base()
    print(f"Created {len(generated)} knowledge chunks.")
    print(f"Saved to: {OUTPUT_PATH}")
