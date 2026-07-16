from __future__ import annotations

import json
from pathlib import Path
from typing import Any


APPLICATION_DIR = Path(__file__).resolve().parent.parent
KB_DIR = APPLICATION_DIR / "knowledge_base"


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_exercise(path: Path) -> list[str]:
    errors: list[str] = []
    record = load(path)

    required = {
        "exercise_id",
        "exercise_name",
        "category",
        "starting_position",
        "movement_phases",
        "observable_metrics",
        "common_execution_errors",
        "safety_notes",
        "evidence_sources",
        "clinical_review_status",
    }

    missing = required - set(record)
    if missing:
        errors.append(f"{path.name}: missing fields {sorted(missing)}")

    if record.get("clinical_review_status") not in {"pending", "approved", "rejected"}:
        errors.append(f"{path.name}: invalid clinical_review_status")

    return errors


def main() -> None:
    errors: list[str] = []

    for path in sorted((KB_DIR / "exercises").glob("*.json")):
        errors.extend(validate_exercise(path))

    for path in KB_DIR.rglob("*.json"):
        try:
            load(path)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: invalid JSON: {exc}")

    if errors:
        print("Knowledge-base validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Knowledge-base validation passed.")


if __name__ == "__main__":
    main()
