class SecurityConfigurationError(RuntimeError):
    """Raised when a required security setting is missing."""


class AuthorizationError(PermissionError):
    """Raised when a user cannot access a patient."""


class UnsafeLLMContextError(RuntimeError):
    """Raised when identifying information is found in LLM context."""


class PrivacyViolationError(RuntimeError):
    """Raised when private information is found in generated output."""