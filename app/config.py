
"""
Centralized configuration for the Healthcare Mobility Monitoring System.

Configuration values are loaded from environment variables whenever possible.
Sensitive values such as API keys, patient-token secrets, encryption keys, and
AWS credentials must never be committed to source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# =============================================================================
# Project paths
# =============================================================================

APP_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = APP_DIR.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
SYNTHETIC_DATA_DIR: Final[Path] = PROJECT_ROOT / "synthetic_data"
KNOWLEDGE_BASE_DIR: Final[Path] = PROJECT_ROOT / "knowledge_base"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

# Backward-compatible string path used by older modules.
BASE_DIR: Final[str] = str(APP_DIR)


# =============================================================================
# Helper functions
# =============================================================================

def _get_bool(name: str, default: bool = False) -> bool:
    """
    Read a boolean value from an environment variable.

    Accepted true values:
        1, true, yes, y, on

    Accepted false values:
        0, false, no, n, off
    """
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()

    if normalized_value in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized_value in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name} must be a boolean value. "
        f"Received: {raw_value!r}"
    )


def _get_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read and validate an integer environment variable."""
    raw_value = os.getenv(name)

    if raw_value is None:
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {name} must be an integer. "
                f"Received: {raw_value!r}"
            ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(
            f"Environment variable {name} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"Environment variable {name} must be at most {maximum}."
        )

    return value


def _get_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read and validate a floating-point environment variable."""
    raw_value = os.getenv(name)

    if raw_value is None:
        value = default
    else:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {name} must be numeric. "
                f"Received: {raw_value!r}"
            ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(
            f"Environment variable {name} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"Environment variable {name} must be at most {maximum}."
        )

    return value


def _get_path(
    name: str,
    default: Path,
    *,
    relative_to: Path = PROJECT_ROOT,
) -> Path:
    """
    Read a filesystem path from an environment variable.

    Relative paths are resolved relative to the project root.
    """
    raw_value = os.getenv(name)

    if not raw_value:
        return default.resolve()

    path = Path(raw_value).expanduser()

    if not path.is_absolute():
        path = relative_to / path

    return path.resolve()


# =============================================================================
# Runtime environment
# =============================================================================

APP_ENV: Final[str] = os.getenv("APP_ENV", "development").strip().lower()

SUPPORTED_APP_ENVIRONMENTS: Final[set[str]] = {
    "development",
    "testing",
    "production",
}

if APP_ENV not in SUPPORTED_APP_ENVIRONMENTS:
    raise ValueError(
        f"Unsupported APP_ENV: {APP_ENV!r}. "
        f"Expected one of: {sorted(SUPPORTED_APP_ENVIRONMENTS)}"
    )

DEBUG: Final[bool] = _get_bool(
    "DEBUG",
    default=APP_ENV == "development",
)


# =============================================================================
# Data-source configuration
# =============================================================================

# Supported modes:
#   synthetic     - generated demonstration data
#   local_research - local research CSV files
#   mock          - legacy mock-data mode
#   aws           - DynamoDB/S3/IoT-backed data
DATA_MODE: Final[str] = os.getenv(
    "DATA_MODE",
    os.getenv("DATA_SOURCE", "synthetic"),
).strip().lower()

SUPPORTED_DATA_MODES: Final[set[str]] = {
    "synthetic",
    "local_research",
    "mock",
    "aws",
}

if DATA_MODE not in SUPPORTED_DATA_MODES:
    raise ValueError(
        f"Unsupported DATA_MODE: {DATA_MODE!r}. "
        f"Expected one of: {sorted(SUPPORTED_DATA_MODES)}"
    )

# Backward compatibility for existing code that imports DATA_SOURCE.
DATA_SOURCE: Final[str] = DATA_MODE

SYNTHETIC_DATA_ROOT: Final[Path] = _get_path(
    "SYNTHETIC_DATA_ROOT",
    SYNTHETIC_DATA_DIR,
)

LOCAL_DATA_ROOT: Final[Path] = _get_path(
    "LOCAL_DATA_ROOT",
    DATA_DIR,
)

MOBILITY_DATA_ROOT: Final[Path] = _get_path(
    "MOBILITY_DATA_ROOT",
    DATA_DIR / "mobility",
)

TREMOR_DATA_ROOT: Final[Path] = _get_path(
    "TREMOR_DATA_ROOT",
    DATA_DIR / "tremor",
)

DATA_SCHEMA_VERSION: Final[str] = os.getenv(
    "DATA_SCHEMA_VERSION",
    "1.0",
).strip()

SYNTHETIC_DATA_LABEL: Final[str] = os.getenv(
    "SYNTHETIC_DATA_LABEL",
    "Synthetic demonstration data — not a real patient record.",
).strip()


# =============================================================================
# Patient-identity and encryption configuration
# =============================================================================

# Used by PatientIdentityService to create stable HMAC-SHA256 patient keys.
# This value must contain at least 32 characters outside tests.
PATIENT_TOKEN_SECRET: Final[str] = os.getenv(
    "PATIENT_TOKEN_SECRET",
    "",
).strip()

# Optional Fernet key used only when reversible patient-ID encryption is needed.
PATIENT_ENCRYPTION_KEY: Final[str] = os.getenv(
    "PATIENT_ENCRYPTION_KEY",
    "",
).strip()

PATIENT_KEY_PREFIX_LENGTH: Final[int] = _get_int(
    "PATIENT_KEY_PREFIX_LENGTH",
    default=12,
    minimum=8,
    maximum=32,
)


# =============================================================================
# Authentication and session configuration
# =============================================================================

SESSION_TIMEOUT_MINUTES: Final[int] = _get_int(
    "SESSION_TIMEOUT_MINUTES",
    default=30,
    minimum=5,
    maximum=1_440,
)

MAX_LOGIN_ATTEMPTS: Final[int] = _get_int(
    "MAX_LOGIN_ATTEMPTS",
    default=5,
    minimum=1,
    maximum=100,
)

LOGIN_LOCKOUT_MINUTES: Final[int] = _get_int(
    "LOGIN_LOCKOUT_MINUTES",
    default=15,
    minimum=1,
    maximum=1_440,
)

BCRYPT_ROUNDS: Final[int] = _get_int(
    "BCRYPT_ROUNDS",
    default=12,
    minimum=4,
    maximum=16,
)


# =============================================================================
# Privacy and audit configuration
# =============================================================================

AUDIT_LOG_PATH: Final[Path] = _get_path(
    "AUDIT_LOG_PATH",
    LOG_DIR / "audit.jsonl",
)

APPLICATION_LOG_PATH: Final[Path] = _get_path(
    "APPLICATION_LOG_PATH",
    LOG_DIR / "application.log",
)

LOG_LEVEL: Final[str] = os.getenv(
    "LOG_LEVEL",
    "INFO",
).strip().upper()

SUPPORTED_LOG_LEVELS: Final[set[str]] = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}

if LOG_LEVEL not in SUPPORTED_LOG_LEVELS:
    raise ValueError(
        f"Unsupported LOG_LEVEL: {LOG_LEVEL!r}. "
        f"Expected one of: {sorted(SUPPORTED_LOG_LEVELS)}"
    )

ENABLE_AUDIT_LOGGING: Final[bool] = _get_bool(
    "ENABLE_AUDIT_LOGGING",
    default=True,
)

ENABLE_OUTPUT_PRIVACY_CHECK: Final[bool] = _get_bool(
    "ENABLE_OUTPUT_PRIVACY_CHECK",
    default=True,
)

ENABLE_PROMPT_INJECTION_GUARD: Final[bool] = _get_bool(
    "ENABLE_PROMPT_INJECTION_GUARD",
    default=True,
)

# Never enable this in production.
LOG_LLM_PROMPTS: Final[bool] = _get_bool(
    "LOG_LLM_PROMPTS",
    default=False,
)

if APP_ENV == "production" and LOG_LLM_PROMPTS:
    raise ValueError(
        "LOG_LLM_PROMPTS cannot be enabled in production because prompts "
        "may contain sensitive information."
    )


# =============================================================================
# LLM and chatbot configuration
# =============================================================================

USE_LLM: Final[bool] = _get_bool(
    "USE_LLM",
    default=False,
)

LLM_PROVIDER: Final[str] = os.getenv(
    "LLM_PROVIDER",
    "openai",
).strip().lower()

SUPPORTED_LLM_PROVIDERS: Final[set[str]] = {
    "openai",
    "local",
}

if LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r}. "
        f"Expected one of: {sorted(SUPPORTED_LLM_PROVIDERS)}"
    )

OPENAI_API_KEY: Final[str] = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

OPENAI_MODEL: Final[str] = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
).strip()

LLM_TEMPERATURE: Final[float] = _get_float(
    "LLM_TEMPERATURE",
    default=0.1,
    minimum=0.0,
    maximum=2.0,
)

LLM_MAX_OUTPUT_TOKENS: Final[int] = _get_int(
    "LLM_MAX_OUTPUT_TOKENS",
    default=800,
    minimum=1,
    maximum=16_384,
)

LLM_REQUEST_TIMEOUT_SECONDS: Final[int] = _get_int(
    "LLM_REQUEST_TIMEOUT_SECONDS",
    default=30,
    minimum=1,
    maximum=300,
)

LLM_MAX_RETRIES: Final[int] = _get_int(
    "LLM_MAX_RETRIES",
    default=2,
    minimum=0,
    maximum=10,
)

REQUIRE_APPROVED_KNOWLEDGE: Final[bool] = _get_bool(
    "REQUIRE_APPROVED_KNOWLEDGE",
    default=True,
)

KNOWLEDGE_BASE_ROOT: Final[Path] = _get_path(
    "KNOWLEDGE_BASE_ROOT",
    KNOWLEDGE_BASE_DIR,
)

KNOWLEDGE_TOP_K: Final[int] = _get_int(
    "KNOWLEDGE_TOP_K",
    default=5,
    minimum=1,
    maximum=50,
)


# =============================================================================
# AWS general configuration
# =============================================================================

AWS_REGION: Final[str] = os.getenv(
    "AWS_REGION",
    "us-west-2",
).strip()

# Keep optional. The application should normally obtain account information
# through the active AWS identity rather than depending on a hardcoded value.
AWS_ACCOUNT_ID: Final[str] = os.getenv(
    "AWS_ACCOUNT_ID",
    "",
).strip()


# =============================================================================
# AWS IoT Core
# =============================================================================

IOT_ENDPOINT: Final[str] = os.getenv(
    "IOT_ENDPOINT",
    "",
).strip()

IOT_TOPIC_PREFIX: Final[str] = os.getenv(
    "IOT_TOPIC_PREFIX",
    "mobility/sessions",
).strip()

CERT_DIR: Final[Path] = _get_path(
    "AWS_CERT_DIR",
    APP_DIR / "certs",
)

CERT_PATH: Final[Path] = _get_path(
    "AWS_CERT_PATH",
    CERT_DIR / "device.pem.crt",
)

KEY_PATH: Final[Path] = _get_path(
    "AWS_PRIVATE_KEY_PATH",
    CERT_DIR / "private.pem.key",
)

ROOT_CA_PATH: Final[Path] = _get_path(
    "AWS_ROOT_CA_PATH",
    CERT_DIR / "AmazonRootCA1.pem",
)


# =============================================================================
# DynamoDB
# =============================================================================

DYNAMODB_TABLE: Final[str] = os.getenv(
    "DYNAMODB_TABLE",
    "MobilitySessionReps",
).strip()

DYNAMODB_PATIENT_ACCESS_TABLE: Final[str] = os.getenv(
    "DYNAMODB_PATIENT_ACCESS_TABLE",
    "PatientAccessMappings",
).strip()

DYNAMODB_USER_TABLE: Final[str] = os.getenv(
    "DYNAMODB_USER_TABLE",
    "MobilityUsers",
).strip()


# =============================================================================
# Amazon S3
# =============================================================================

S3_BUCKET: Final[str] = os.getenv(
    "S3_BUCKET",
    "",
).strip()

S3_SYNTHETIC_PREFIX: Final[str] = os.getenv(
    "S3_SYNTHETIC_PREFIX",
    "synthetic/",
).strip()

S3_MOBILITY_PREFIX: Final[str] = os.getenv(
    "S3_MOBILITY_PREFIX",
    "mobility/",
).strip()


# =============================================================================
# Streamlit/UI configuration
# =============================================================================

APP_TITLE: Final[str] = os.getenv(
    "APP_TITLE",
    "Healthcare Mobility Monitoring System",
).strip()

APP_ICON: Final[str] = os.getenv(
    "APP_ICON",
    "🏥",
).strip()

SHOW_SYNTHETIC_DATA_BANNER: Final[bool] = _get_bool(
    "SHOW_SYNTHETIC_DATA_BANNER",
    default=DATA_MODE == "synthetic",
)

SHOW_INTERNAL_ERROR_DETAILS: Final[bool] = _get_bool(
    "SHOW_INTERNAL_ERROR_DETAILS",
    default=APP_ENV == "development",
)


# =============================================================================
# Typed configuration groups
# =============================================================================

@dataclass(frozen=True)
class SecuritySettings:
    """Security and authentication settings."""

    patient_token_secret: str
    patient_encryption_key: str
    patient_key_prefix_length: int
    session_timeout_minutes: int
    max_login_attempts: int
    login_lockout_minutes: int
    bcrypt_rounds: int
    audit_log_path: Path
    application_log_path: Path
    log_level: str
    enable_audit_logging: bool
    enable_output_privacy_check: bool
    enable_prompt_injection_guard: bool


@dataclass(frozen=True)
class DataSettings:
    """Data repository and schema settings."""

    data_mode: str
    synthetic_data_root: Path
    local_data_root: Path
    mobility_data_root: Path
    tremor_data_root: Path
    schema_version: str
    synthetic_data_label: str


@dataclass(frozen=True)
class LLMSettings:
    """Chatbot and language-model settings."""

    enabled: bool
    provider: str
    openai_api_key: str
    model: str
    temperature: float
    max_output_tokens: int
    request_timeout_seconds: int
    max_retries: int
    require_approved_knowledge: bool
    knowledge_base_root: Path
    knowledge_top_k: int


@dataclass(frozen=True)
class AWSSettings:
    """AWS service settings."""

    region: str
    account_id: str
    iot_endpoint: str
    iot_topic_prefix: str
    cert_path: Path
    private_key_path: Path
    root_ca_path: Path
    dynamodb_table: str
    patient_access_table: str
    user_table: str
    s3_bucket: str


@dataclass(frozen=True)
class AppSettings:
    """Complete application configuration."""

    environment: str
    debug: bool
    title: str
    icon: str
    security: SecuritySettings
    data: DataSettings
    llm: LLMSettings
    aws: AWSSettings


# =============================================================================
# Configuration loaders
# =============================================================================

def load_security_settings(
    *,
    validate_required: bool = True,
) -> SecuritySettings:
    """
    Load security settings.

    Set validate_required=False for tooling that does not access patient data,
    such as documentation generation or limited static checks.
    """
    if validate_required:
        if not PATIENT_TOKEN_SECRET:
            raise RuntimeError(
                "PATIENT_TOKEN_SECRET is required. Add it to your local .env "
                "file or deployment environment."
            )

        if len(PATIENT_TOKEN_SECRET) < 32:
            raise RuntimeError(
                "PATIENT_TOKEN_SECRET must contain at least 32 characters."
            )

    return SecuritySettings(
        patient_token_secret=PATIENT_TOKEN_SECRET,
        patient_encryption_key=PATIENT_ENCRYPTION_KEY,
        patient_key_prefix_length=PATIENT_KEY_PREFIX_LENGTH,
        session_timeout_minutes=SESSION_TIMEOUT_MINUTES,
        max_login_attempts=MAX_LOGIN_ATTEMPTS,
        login_lockout_minutes=LOGIN_LOCKOUT_MINUTES,
        bcrypt_rounds=BCRYPT_ROUNDS,
        audit_log_path=AUDIT_LOG_PATH,
        application_log_path=APPLICATION_LOG_PATH,
        log_level=LOG_LEVEL,
        enable_audit_logging=ENABLE_AUDIT_LOGGING,
        enable_output_privacy_check=ENABLE_OUTPUT_PRIVACY_CHECK,
        enable_prompt_injection_guard=ENABLE_PROMPT_INJECTION_GUARD,
    )


def load_data_settings() -> DataSettings:
    """Load repository and data-schema settings."""
    return DataSettings(
        data_mode=DATA_MODE,
        synthetic_data_root=SYNTHETIC_DATA_ROOT,
        local_data_root=LOCAL_DATA_ROOT,
        mobility_data_root=MOBILITY_DATA_ROOT,
        tremor_data_root=TREMOR_DATA_ROOT,
        schema_version=DATA_SCHEMA_VERSION,
        synthetic_data_label=SYNTHETIC_DATA_LABEL,
    )


def load_llm_settings(
    *,
    validate_required: bool = True,
) -> LLMSettings:
    """Load language-model and knowledge-retrieval settings."""
    if (
        validate_required
        and USE_LLM
        and LLM_PROVIDER == "openai"
        and not OPENAI_API_KEY
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is required when USE_LLM=true and "
            "LLM_PROVIDER=openai."
        )

    return LLMSettings(
        enabled=USE_LLM,
        provider=LLM_PROVIDER,
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        request_timeout_seconds=LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        require_approved_knowledge=REQUIRE_APPROVED_KNOWLEDGE,
        knowledge_base_root=KNOWLEDGE_BASE_ROOT,
        knowledge_top_k=KNOWLEDGE_TOP_K,
    )


def load_aws_settings(
    *,
    validate_required: bool = True,
) -> AWSSettings:
    """Load AWS service settings."""
    if validate_required and DATA_MODE == "aws":
        missing_values: list[str] = []

        if not IOT_ENDPOINT:
            missing_values.append("IOT_ENDPOINT")

        if not S3_BUCKET:
            missing_values.append("S3_BUCKET")

        if not DYNAMODB_TABLE:
            missing_values.append("DYNAMODB_TABLE")

        if missing_values:
            raise RuntimeError(
                "Missing required AWS configuration: "
                + ", ".join(missing_values)
            )

    return AWSSettings(
        region=AWS_REGION,
        account_id=AWS_ACCOUNT_ID,
        iot_endpoint=IOT_ENDPOINT,
        iot_topic_prefix=IOT_TOPIC_PREFIX,
        cert_path=CERT_PATH,
        private_key_path=KEY_PATH,
        root_ca_path=ROOT_CA_PATH,
        dynamodb_table=DYNAMODB_TABLE,
        patient_access_table=DYNAMODB_PATIENT_ACCESS_TABLE,
        user_table=DYNAMODB_USER_TABLE,
        s3_bucket=S3_BUCKET,
    )


def load_app_settings(
    *,
    validate_required: bool = True,
) -> AppSettings:
    """Load and return the complete application configuration."""
    return AppSettings(
        environment=APP_ENV,
        debug=DEBUG,
        title=APP_TITLE,
        icon=APP_ICON,
        security=load_security_settings(
            validate_required=validate_required
        ),
        data=load_data_settings(),
        llm=load_llm_settings(
            validate_required=validate_required
        ),
        aws=load_aws_settings(
            validate_required=validate_required
        ),
    )


# =============================================================================
# Configuration validation
# =============================================================================

def validate_configuration() -> list[str]:
    """
    Validate configuration and return non-fatal warnings.

    Fatal configuration errors raise RuntimeError or ValueError.
    """
    warnings: list[str] = []

    load_app_settings(validate_required=True)

    if DATA_MODE == "synthetic" and not SYNTHETIC_DATA_ROOT.exists():
        warnings.append(
            f"Synthetic data directory does not exist: "
            f"{SYNTHETIC_DATA_ROOT}"
        )

    if DATA_MODE == "local_research" and not LOCAL_DATA_ROOT.exists():
        warnings.append(
            f"Local data directory does not exist: {LOCAL_DATA_ROOT}"
        )

    if not KNOWLEDGE_BASE_ROOT.exists():
        warnings.append(
            f"Knowledge-base directory does not exist: "
            f"{KNOWLEDGE_BASE_ROOT}"
        )

    if DATA_MODE == "aws":
        for certificate_path in (
            CERT_PATH,
            KEY_PATH,
            ROOT_CA_PATH,
        ):
            if not certificate_path.exists():
                warnings.append(
                    f"AWS IoT certificate file does not exist: "
                    f"{certificate_path}"
                )

    if APP_ENV == "production":
        if DEBUG:
            warnings.append(
                "DEBUG is enabled in production."
            )

        if DATA_MODE in {"mock", "synthetic"}:
            warnings.append(
                f"Production is running with DATA_MODE={DATA_MODE!r}."
            )

        if not ENABLE_AUDIT_LOGGING:
            warnings.append(
                "Audit logging is disabled in production."
            )

        if not ENABLE_OUTPUT_PRIVACY_CHECK:
            warnings.append(
                "LLM output privacy checking is disabled in production."
            )

    return warnings


def ensure_runtime_directories() -> None:
    """
    Create non-sensitive runtime directories when they do not exist.

    This function intentionally does not create real-patient-data directories
    or AWS certificate directories.
    """
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    APPLICATION_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


__all__ = [
    # Paths
    "APP_DIR",
    "PROJECT_ROOT",
    "BASE_DIR",
    "DATA_DIR",
    "SYNTHETIC_DATA_DIR",
    "KNOWLEDGE_BASE_DIR",
    "LOG_DIR",
    # Runtime
    "APP_ENV",
    "DEBUG",
    # Data
    "DATA_MODE",
    "DATA_SOURCE",
    "SYNTHETIC_DATA_ROOT",
    "LOCAL_DATA_ROOT",
    "MOBILITY_DATA_ROOT",
    "TREMOR_DATA_ROOT",
    "DATA_SCHEMA_VERSION",
    "SYNTHETIC_DATA_LABEL",
    # Security
    "PATIENT_TOKEN_SECRET",
    "PATIENT_ENCRYPTION_KEY",
    "PATIENT_KEY_PREFIX_LENGTH",
    "SESSION_TIMEOUT_MINUTES",
    "MAX_LOGIN_ATTEMPTS",
    "LOGIN_LOCKOUT_MINUTES",
    "BCRYPT_ROUNDS",
    "AUDIT_LOG_PATH",
    "APPLICATION_LOG_PATH",
    "LOG_LEVEL",
    "ENABLE_AUDIT_LOGGING",
    "ENABLE_OUTPUT_PRIVACY_CHECK",
    "ENABLE_PROMPT_INJECTION_GUARD",
    # LLM
    "USE_LLM",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_OUTPUT_TOKENS",
    "LLM_REQUEST_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "REQUIRE_APPROVED_KNOWLEDGE",
    "KNOWLEDGE_BASE_ROOT",
    "KNOWLEDGE_TOP_K",
    # AWS
    "AWS_REGION",
    "AWS_ACCOUNT_ID",
    "IOT_ENDPOINT",
    "IOT_TOPIC_PREFIX",
    "CERT_DIR",
    "CERT_PATH",
    "KEY_PATH",
    "ROOT_CA_PATH",
    "DYNAMODB_TABLE",
    "DYNAMODB_PATIENT_ACCESS_TABLE",
    "DYNAMODB_USER_TABLE",
    "S3_BUCKET",
    # UI
    "APP_TITLE",
    "APP_ICON",
    "SHOW_SYNTHETIC_DATA_BANNER",
    "SHOW_INTERNAL_ERROR_DETAILS",
    # Dataclasses and loaders
    "SecuritySettings",
    "DataSettings",
    "LLMSettings",
    "AWSSettings",
    "AppSettings",
    "load_security_settings",
    "load_data_settings",
    "load_llm_settings",
    "load_aws_settings",
    "load_app_settings",
    "validate_configuration",
    "ensure_runtime_directories",
]

