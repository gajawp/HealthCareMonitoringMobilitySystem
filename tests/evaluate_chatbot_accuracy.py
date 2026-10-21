from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("PATIENT_TOKEN_SECRET"):
    raise RuntimeError(
        "PATIENT_TOKEN_SECRET is not loaded. "
        "Check the project-root .env file."
    )

from app.chatbot.language_service import LanguageService
from app.chatbot.orchestrator import HealthcareChatbotOrchestrator


DENIAL_TERMS = (
    "access denied",
    "not authorized",
    "unauthorized",
    "permission denied",
    "do not have access",
    "don't have access",
    "cannot access",
    "can't access",
    "not permitted",
    "not allowed",
)


@dataclass
class EvalResult:
    test_id: str
    category: str
    language: str
    question: str
    english_question: str
    expected_answer: str
    expected_authorization: str
    bot_answer: str
    displayed_answer: str
    detected_language: str
    translation_available: bool
    factual_correct: Optional[bool]
    authorization_correct: Optional[bool]
    latency_seconds: float
    error: str = ""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def extract_numbers(text: str) -> list[float]:
    output = []
    for match in re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text or ""):
        try:
            output.append(float(match.replace(",", "")))
        except ValueError:
            pass
    return output


def score_factual(expected: str, actual: str, tolerance: float) -> Optional[bool]:
    if not expected or normalize_text(expected) in {"allow", "deny"}:
        return None

    items = [x.strip() for x in expected.split(";") if x.strip()]
    actual_n = normalize_text(actual)

    for item in items:
        value = item.split("=", 1)[1].strip() if "=" in item else item

        expected_nums = extract_numbers(value)
        if expected_nums:
            actual_nums = extract_numbers(actual)
            if not all(
                any(abs(e - a) <= tolerance for a in actual_nums)
                for e in expected_nums
            ):
                return False
        elif normalize_text(value) not in actual_n:
            return False

    return True


def score_authorization(expected: str, answer: str) -> Optional[bool]:
    expected = normalize_text(expected)
    if not expected:
        return None

    denied = any(term in normalize_text(answer) for term in DENIAL_TERMS)

    if expected == "deny":
        return denied
    if expected == "allow":
        return not denied
    return None


def unpack_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(
            response.get("answer")
            or response.get("response")
            or response.get("message")
            or response
        )
    return str(getattr(response, "answer", response))


def evaluate(input_path: Path, output_path: Path) -> list[EvalResult]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        tests = list(csv.DictReader(f))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chatbot = HealthcareChatbotOrchestrator()
    language_service = LanguageService()

    results: list[EvalResult] = []

    print(f"\nLoaded {len(tests)} chatbot accuracy tests.")
    print(
        "Translation service available:",
        language_service.is_available(),
        "\n",
    )

    for index, row in enumerate(tests, start=1):
        test_id = row.get("test_id", f"T{index:03d}")
        question = row.get("question", "")
        language = row.get("language", "en").strip().lower() or "en"
        expected = row.get("expected_answer", "")
        expected_auth = row.get("expected_authorization", "")
        tolerance = float(row.get("numeric_tolerance", "0") or 0)

        print(f"[{index}/{len(tests)}] {test_id}: {question}")

        started = time.perf_counter()

        try:
            # The CSV tells us the intended language. Detection is also run
            # and recorded as a separate multilingual diagnostic.
            detected_language = language_service.detect_language(question)

            english_question = language_service.translate_to_english(
                question,
                source_language=language,
            )

            response = chatbot.answer(
                question=english_question,
                user_id=row.get("user_id", ""),
                user_name=row.get("user_name", "").strip()
                or row.get("user_id", ""),
                role=row.get("role", ""),
                patient_id=row.get("patient_id", ""),
            )

            bot_answer = unpack_response(response)

            displayed_answer = language_service.translate_from_english(
                bot_answer,
                target_language=language,
            )

            latency = time.perf_counter() - started

            # Factual scoring is performed on the English internal answer.
            # Numbers and deterministic facts should remain the same after
            # translation, while this avoids language-specific text matching.
            factual = score_factual(expected, bot_answer, tolerance)
            auth = score_authorization(expected_auth, bot_answer)

            print(
                f"    factual={'PASS' if factual is True else 'FAIL' if factual is False else 'N/A'} "
                f"auth={'PASS' if auth is True else 'FAIL' if auth is False else 'N/A'} "
                f"detected={detected_language} latency={latency:.2f}s"
            )

            if factual is False:
                print(f"    EXPECTED: {expected}")
                print(f"    ENGLISH : {english_question}")
                print(f"    BOT     : {bot_answer}")

            results.append(
                EvalResult(
                    test_id=test_id,
                    category=row.get("category", ""),
                    language=language,
                    question=question,
                    english_question=english_question,
                    expected_answer=expected,
                    expected_authorization=expected_auth,
                    bot_answer=bot_answer,
                    displayed_answer=displayed_answer,
                    detected_language=detected_language,
                    translation_available=language_service.is_available(),
                    factual_correct=factual,
                    authorization_correct=auth,
                    latency_seconds=latency,
                )
            )

        except Exception as exc:
            latency = time.perf_counter() - started
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            results.append(
                EvalResult(
                    test_id=test_id,
                    category=row.get("category", ""),
                    language=language,
                    question=question,
                    english_question="",
                    expected_answer=expected,
                    expected_authorization=expected_auth,
                    bot_answer="",
                    displayed_answer="",
                    detected_language="",
                    translation_available=language_service.is_available(),
                    factual_correct=False if expected and normalize_text(expected) not in {"allow", "deny"} else None,
                    authorization_correct=False if expected_auth else None,
                    latency_seconds=latency,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    fields = [
        "test_id","category","language","question","english_question",
        "expected_answer","expected_authorization","bot_answer",
        "displayed_answer","detected_language","translation_available",
        "factual_correct","authorization_correct","latency_seconds","error",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "test_id": r.test_id,
                "category": r.category,
                "language": r.language,
                "question": r.question,
                "english_question": r.english_question,
                "expected_answer": r.expected_answer,
                "expected_authorization": r.expected_authorization,
                "bot_answer": r.bot_answer,
                "displayed_answer": r.displayed_answer,
                "detected_language": r.detected_language,
                "translation_available": r.translation_available,
                "factual_correct": "" if r.factual_correct is None else ("PASS" if r.factual_correct else "FAIL"),
                "authorization_correct": "" if r.authorization_correct is None else ("PASS" if r.authorization_correct else "FAIL"),
                "latency_seconds": f"{r.latency_seconds:.4f}",
                "error": r.error,
            })

    return results


def metric(values: list[Optional[bool]]) -> tuple[int, int, float]:
    values = [v for v in values if v is not None]
    if not values:
        return 0, 0, 0.0
    passed = sum(v is True for v in values)
    return passed, len(values), passed / len(values) * 100.0


def print_summary(results: list[EvalResult]) -> None:
    fp, ft, fpc = metric([r.factual_correct for r in results])
    ap, at, apc = metric([r.authorization_correct for r in results])

    print("\n" + "=" * 72)
    print("HEALTHBRIDGE AI - CHATBOT ACCURACY EVALUATION")
    print("=" * 72)
    print(f"Total test cases       : {len(results)}")
    print(f"Execution errors       : {sum(bool(r.error) for r in results)}")
    print(f"Factual accuracy       : {fp}/{ft} ({fpc:.2f}%)")
    print(f"Authorization accuracy : {ap}/{at} ({apc:.2f}%)")
    print(
        f"Average latency        : "
        f"{sum(r.latency_seconds for r in results) / len(results):.3f} seconds"
        if results else
        "Average latency        : 0.000 seconds"
    )

    print("\nCategory-wise factual accuracy")
    print("-" * 72)
    for category in sorted({r.category for r in results if r.category}):
        subset = [r.factual_correct for r in results if r.category == category]
        p, t, pct = metric(subset)
        if t:
            print(f"{category:24s}: {p}/{t} ({pct:.2f}%)")

    print("\nLanguage-wise factual accuracy")
    print("-" * 72)
    for language in sorted({r.language for r in results if r.language}):
        subset = [r.factual_correct for r in results if r.language == language]
        p, t, pct = metric(subset)
        if t:
            print(f"{language:24s}: {p}/{t} ({pct:.2f}%)")

    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="tests/chatbot_accuracy_test.csv",
    )
    parser.add_argument(
        "--output",
        default="tests/results/chatbot_accuracy_results.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path.resolve())

    results = evaluate(input_path, Path(args.output))
    print_summary(results)
    print(f"\nDetailed results saved to:\n{Path(args.output).resolve()}\n")


if __name__ == "__main__":
    main()