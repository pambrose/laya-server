"""Errors mapped onto the status codes the Jev API documents.

Jev documents 401 (missing/invalid key), 422 (validation failed), 429 (rate
limit) and 529 (overloaded). Clients and the TypeSafe SDKs already retry 429
and 529 with backoff, so surfacing the right code matters for drop-in use.
"""


class JevError(Exception):
    """An error that renders as a Jev-shaped JSON body."""

    status_code = 500
    error_type = "api_error"

    def body(self) -> dict[str, dict[str, str]]:
        return {"error": {"type": self.error_type, "message": str(self)}}


class ValidationFailed(JevError):
    status_code = 422
    error_type = "validation_error"


class Unauthorized(JevError):
    status_code = 401
    error_type = "authentication_error"


class Overloaded(JevError):
    status_code = 529
    error_type = "overloaded_error"
