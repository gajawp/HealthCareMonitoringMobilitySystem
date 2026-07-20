from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class LanguageInfo:
    code: str
    name: str


class LanguageService:
    """
    Detects the user's language and translates chatbot content.

    The chatbot continues processing internally in English. Patient-style
    references such as P1001 are masked before text is sent to the translation
    model and restored afterward.
    """

    SUPPORTED_LANGUAGES: dict[str, str] = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "hi": "Hindi",
        "te": "Telugu",
        "ta": "Tamil",
        "zh": "Chinese",
        "ar": "Arabic",
    }

    PATIENT_REFERENCE_PATTERN = re.compile(
        r"\bP\d{3,10}\b",
        flags=re.IGNORECASE,
    )

    def __init__(self) -> None:
        # Load variables from the project-root .env file.
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        )

        self.client = (
            OpenAI(api_key=api_key)
            if api_key
            else None
        )

    @classmethod
    def language_name(
        cls,
        language_code: str,
    ) -> str:
        """
        Return the display name for a supported language code.
        """

        return cls.SUPPORTED_LANGUAGES.get(
            language_code,
            "English",
        )

    def is_available(self) -> bool:
        """
        Return True when translation through OpenAI is available.
        """

        return self.client is not None

    def detect_language(
        self,
        text: str,
    ) -> str:
        """
        Detect the language used in the supplied text.

        Returns one of:
        en, es, fr, de, hi, te, ta, zh, ar

        Falls back to English if detection cannot be completed.
        """

        clean_text = text.strip()

        if not clean_text:
            return "en"

        if self.client is None:
            return "en"

        masked_text, _ = self._mask_patient_references(
            clean_text
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={
                    "type": "json_object"
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Detect the language used in the user message. "
                            "Return JSON only with the key language_code. "
                            "Choose exactly one code from: "
                            "en, es, fr, de, hi, te, ta, zh, ar. "
                            'Example: {"language_code": "te"}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": masked_text,
                    },
                ],
            )

            content = (
                response.choices[0].message.content
                or "{}"
            )

            result: dict[str, Any] = json.loads(
                content
            )

            language_code = str(
                result.get(
                    "language_code",
                    "en",
                )
            ).strip().lower()

            if (
                language_code
                in self.SUPPORTED_LANGUAGES
            ):
                return language_code

        except Exception:
            pass

        return "en"

    def translate_to_english(
        self,
        text: str,
        source_language: str,
    ) -> str:
        """
        Translate user input into English for internal chatbot processing.
        """

        if not text.strip():
            return text

        if source_language == "en":
            return text

        return self.translate(
            text=text,
            target_language_code="en",
        )

    def translate_from_english(
        self,
        text: str,
        target_language: str,
    ) -> str:
        """
        Translate an English chatbot response into the user's language.
        """

        if not text.strip():
            return text

        if target_language == "en":
            return text

        return self.translate(
            text=text,
            target_language_code=target_language,
        )

    def translate(
        self,
        *,
        text: str,
        target_language_code: str,
    ) -> str:
        """
        Translate text into the requested supported language.

        If translation is unavailable or fails, return the original text.
        """

        if not text.strip():
            return text

        if (
            target_language_code
            not in self.SUPPORTED_LANGUAGES
        ):
            return text

        if self.client is None:
            return text

        target_language_name = self.language_name(
            target_language_code
        )

        masked_text, replacements = (
            self._mask_patient_references(text)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate the supplied healthcare chatbot text "
                            f"into {target_language_name}. "
                            "Preserve medical meaning, dates, numbers, metric "
                            "values, Markdown formatting, bullet points, URLs, "
                            "exercise terminology, and opaque placeholders such "
                            "as __PRIVATE_REFERENCE_1__. "
                            "Do not add, remove, reinterpret, or modify medical "
                            "advice. Return only the translated text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": masked_text,
                    },
                ],
            )

            translated_text = (
                response.choices[0].message.content
                or masked_text
            ).strip()

            return self._restore_patient_references(
                translated_text,
                replacements,
            )

        except Exception:
            return text

    def translate_list(
        self,
        values: list[str],
        target_language_code: str,
    ) -> list[str]:
        """
        Translate every string in a list.
        """

        return [
            self.translate(
                text=value,
                target_language_code=(
                    target_language_code
                ),
            )
            for value in values
        ]

    def translate_exercise_data(
        self,
        exercise_data: dict[str, Any] | None,
        target_language_code: str,
    ) -> dict[str, Any] | None:
        """
        Translate only user-visible exercise fields.

        URLs, IDs, image paths, filenames, flags, and internal metadata remain
        unchanged.
        """

        if not exercise_data:
            return exercise_data

        if target_language_code == "en":
            return exercise_data

        if (
            target_language_code
            not in self.SUPPORTED_LANGUAGES
        ):
            return exercise_data

        translated = dict(exercise_data)

        for field in (
            "profile_name",
            "profile_description",
            "global_safety_message",
        ):
            value = translated.get(field)

            if (
                isinstance(value, str)
                and value.strip()
            ):
                translated[field] = self.translate(
                    text=value,
                    target_language_code=(
                        target_language_code
                    ),
                )

        translated_exercises: list[
            dict[str, Any]
        ] = []

        exercises = translated.get(
            "exercises",
            [],
        )

        if not isinstance(exercises, list):
            exercises = []

        for exercise in exercises:
            if not isinstance(exercise, dict):
                continue

            translated_exercise = dict(
                exercise
            )

            for field in (
                "exercise_name",
                "description",
            ):
                value = translated_exercise.get(
                    field
                )

                if (
                    isinstance(value, str)
                    and value.strip()
                ):
                    translated_exercise[
                        field
                    ] = self.translate(
                        text=value,
                        target_language_code=(
                            target_language_code
                        ),
                    )

            for field in (
                "starting_position",
                "patient_friendly_guidance",
                "safety_notes",
            ):
                values = translated_exercise.get(
                    field
                )

                if isinstance(values, list):
                    translated_exercise[
                        field
                    ] = self.translate_list(
                        [
                            str(item)
                            for item in values
                        ],
                        target_language_code,
                    )

            media = translated_exercise.get(
                "media"
            )

            if isinstance(media, dict):
                translated_media = dict(media)

                translated_images: list[
                    dict[str, Any]
                ] = []

                images = translated_media.get(
                    "images",
                    [],
                )

                if not isinstance(images, list):
                    images = []

                for image in images:
                    if not isinstance(
                        image,
                        dict,
                    ):
                        continue

                    translated_image = dict(
                        image
                    )

                    for field in (
                        "caption",
                        "alt_text",
                    ):
                        value = (
                            translated_image.get(
                                field
                            )
                        )

                        if (
                            isinstance(value, str)
                            and value.strip()
                        ):
                            translated_image[
                                field
                            ] = self.translate(
                                text=value,
                                target_language_code=(
                                    target_language_code
                                ),
                            )

                    translated_images.append(
                        translated_image
                    )

                translated_media[
                    "images"
                ] = translated_images

                translated_exercise[
                    "media"
                ] = translated_media

            translated_exercises.append(
                translated_exercise
            )

        translated[
            "exercises"
        ] = translated_exercises

        return translated

    @classmethod
    def _mask_patient_references(
        cls,
        text: str,
    ) -> tuple[str, dict[str, str]]:
        """
        Replace patient-style identifiers with opaque placeholders.
        """

        replacements: dict[str, str] = {}

        def replace(
            match: re.Match[str],
        ) -> str:
            placeholder = (
                "__PRIVATE_REFERENCE_"
                f"{len(replacements) + 1}__"
            )

            replacements[
                placeholder
            ] = match.group(0)

            return placeholder

        masked_text = (
            cls.PATIENT_REFERENCE_PATTERN.sub(
                replace,
                text,
            )
        )

        return masked_text, replacements

    @staticmethod
    def _restore_patient_references(
        text: str,
        replacements: dict[str, str],
    ) -> str:
        """
        Restore masked patient references after translation.
        """

        restored_text = text

        for (
            placeholder,
            original_value,
        ) in replacements.items():
            restored_text = (
                restored_text.replace(
                    placeholder,
                    original_value,
                )
            )

        return restored_text