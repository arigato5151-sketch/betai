"""
Custom exceptions for API Football service error handling.
"""


class APIFootballException(Exception):
    """Base exception for API Football service errors."""

    pass


class APITimeoutError(APIFootballException):
    """Raised when API request times out after all retries."""

    def __init__(self, path: str, attempts: int = 3):
        self.path = path
        self.attempts = attempts
        super().__init__(f"API request to '{path}' timed out after {attempts} attempts")


class APIRateLimitError(APIFootballException):
    """Raised when API rate limit (429) is exceeded."""

    def __init__(self, path: str, retry_after: int = 60):
        self.path = path
        self.retry_after = retry_after
        super().__init__(f"Rate limited on '{path}'. Retry after {retry_after}s")


class APIDataError(APIFootballException):
    """Raised when API returns unexpected data format or error status."""

    def __init__(self, path: str, status_code: int, message: str | None = None):
        self.path = path
        self.status_code = status_code
        super().__init__(
            f"API error on '{path}' (status {status_code}): {message or 'Unknown error'}"
        )


class InvalidInputError(Exception):
    """Raised when input validation fails."""

    def __init__(self, field: str, constraint: str):
        self.field = field
        self.constraint = constraint
        super().__init__(f"Invalid {field}: {constraint}")
