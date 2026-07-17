import pytest


@pytest.fixture(autouse=True)
def configure_test_security_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PATIENT_TOKEN_SECRET",
        "pytest-patient-token-secret-with-more-than-32-characters",
    )

    monkeypatch.setenv(
        "PATIENT_ENCRYPTION_KEY",
        "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
    )