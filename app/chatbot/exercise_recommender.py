from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXERCISE_FILE = (
    PROJECT_ROOT
    / "data"
    / "exercise_recommendations.json"
)


class ExerciseRecommendationService:
    def __init__(
        self,
        exercise_file: Path = DEFAULT_EXERCISE_FILE,
    ) -> None:
        self.exercise_file = exercise_file
        self.data = self._load_data()

    def _load_data(self) -> dict[str, Any]:
        if not self.exercise_file.exists():
            raise FileNotFoundError(
                f"Exercise recommendation file not found: "
                f"{self.exercise_file}"
            )

        with self.exercise_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Exercise recommendation JSON must contain an object."
            )

        return data

    @staticmethod
    def normalize_condition(
        condition: str | None,
    ) -> str:
        if not condition:
            return ""

        return (
            condition.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    def _resolve_profile_key(
        self,
        condition: str | None,
    ) -> str:
        profiles = self.data.get(
            "condition_profiles",
            {},
        )

        fallback_profile = self.data.get(
            "fallback_profile",
            "general_lower_limb_mobility",
        )

        normalized_condition = self.normalize_condition(
            condition
        )

        if normalized_condition in profiles:
            return normalized_condition

        aliases = {
            "knee_pain": "knee_rehabilitation",
            "knee_injury": "knee_rehabilitation",
            "knee_rehab": "knee_rehabilitation",
            "knee_osteoarthritis": "knee_rehabilitation",
            "lower_limb_weakness":
                "lower_limb_strength_and_control",
            "leg_weakness":
                "lower_limb_strength_and_control",
            "balance_issue": "balance_and_stepping",
            "balance_issues": "balance_and_stepping",
            "fall_risk": "balance_and_stepping",
            "lateral_mobility_issue": "lateral_mobility",
            "squat_training": "squat_movement_training",
        }

        mapped_profile = aliases.get(
            normalized_condition
        )

        if mapped_profile in profiles:
            return mapped_profile

        return fallback_profile

    @staticmethod
    def _resolve_image_path(
        image_path: str | None,
    ) -> str | None:
        if not image_path:
            return None

        path = Path(image_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            return None

        return str(path)

    def get_recommendations(
        self,
        condition: str | None,
    ) -> dict[str, Any]:
        profiles = self.data.get(
            "condition_profiles",
            {},
        )

        exercise_library = self.data.get(
            "exercise_library",
            {},
        )

        profile_key = self._resolve_profile_key(
            condition
        )

        profile = profiles.get(
            profile_key,
            {},
        )

        recommendations = []

        for exercise_id in profile.get(
            "exercise_ids",
            [],
        ):
            exercise = exercise_library.get(
                exercise_id
            )

            if not exercise:
                continue

            if not exercise.get(
                "enabled",
                False,
            ):
                continue

            prepared_exercise = dict(exercise)
            prepared_media = dict(
                exercise.get("media", {})
            )

            prepared_images = []

            for image in prepared_media.get(
                "images",
                [],
            ):
                prepared_image = dict(image)

                prepared_image[
                    "resolved_image_path"
                ] = self._resolve_image_path(
                    image.get("image_path")
                )

                prepared_images.append(
                    prepared_image
                )

            prepared_images.sort(
                key=lambda item: item.get(
                    "display_order",
                    0,
                )
            )

            prepared_media["images"] = (
                prepared_images
            )

            prepared_exercise["media"] = (
                prepared_media
            )

            recommendations.append(
                prepared_exercise
            )

        return {
            "profile_key": profile_key,
            "profile_name": profile.get(
                "display_name",
                "General Lower-Limb Mobility",
            ),
            "profile_description": profile.get(
                "description",
                "",
            ),
            "clinical_review_status": profile.get(
                "clinical_review_status",
                "pending",
            ),
            "global_safety_message": self.data.get(
                "global_safety_message",
                "",
            ),
            "exercises": recommendations,
        }