"""
Structured Error Taxonomy for TubeLens OSS.

Tool errors are *tool results*, not transport failures: they come back with
``isError: true`` and a structured payload the agent can branch on. HTTP status
codes are reserved for transport-level problems and REST endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    UNAUTHORIZED = "UNAUTHORIZED"
    CHANNEL_NOT_FOUND = "CHANNEL_NOT_FOUND"
    VIDEO_UNAVAILABLE = "VIDEO_UNAVAILABLE"
    LIVE_STREAM_NOT_SUPPORTED = "LIVE_STREAM_NOT_SUPPORTED"
    VIDEO_TOO_LONG = "VIDEO_TOO_LONG"
    NO_SPEECH_DETECTED = "NO_SPEECH_DETECTED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_QUOTA = "UPSTREAM_QUOTA"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_EXPIRED = "JOB_EXPIRED"
    TRANSCRIPT_NOT_FOUND = "TRANSCRIPT_NOT_FOUND"
    CURSOR_EXPIRED = "CURSOR_EXPIRED"
    CATALOG_TOO_LARGE = "CATALOG_TOO_LARGE"
    SEARCH_FAILED = "SEARCH_FAILED"
    WINDOW_EXTRACTION_FAILED = "WINDOW_EXTRACTION_FAILED"
    RIGHTS_ATTESTATION_REQUIRED = "RIGHTS_ATTESTATION_REQUIRED"
    INTERNAL = "INTERNAL"


@dataclass(frozen=True)
class ErrorSpec:
    retryable: bool
    retry_after_s: int | None
    recovery_hint: str
    http_status: int  # HTTP status code for REST endpoints


#: Single source of truth — every raised error must resolve here.
ERROR_SPECS: dict[ErrorCode, ErrorSpec] = {
    ErrorCode.INVALID_INPUT: ErrorSpec(
        False, None,
        "Ask the user for a valid YouTube URL or channel name.",
        400,
    ),
    ErrorCode.UNAUTHORIZED: ErrorSpec(
        False, None,
        "Tell the user to check their MCP client configuration and credentials.",
        401,
    ),
    ErrorCode.CHANNEL_NOT_FOUND: ErrorSpec(
        False, None,
        "Retry once using the free-text creator name instead of a handle or ID.",
        404,
    ),
    ErrorCode.VIDEO_UNAVAILABLE: ErrorSpec(
        False, None,
        "Tell the user the video is not publicly accessible; suggest an alternative.",
        404,
    ),
    ErrorCode.LIVE_STREAM_NOT_SUPPORTED: ErrorSpec(
        False, None,
        "Tell the user to retry once the stream is archived as video-on-demand.",
        400,
    ),
    ErrorCode.VIDEO_TOO_LONG: ErrorSpec(
        False, None,
        "Suggest a narrower time range using youtube_ask_video, or decline the request.",
        400,
    ),
    ErrorCode.NO_SPEECH_DETECTED: ErrorSpec(
        False, None,
        "Fall back to metadata; tell the user the video contains no speech.",
        422,
    ),
    ErrorCode.BUDGET_EXCEEDED: ErrorSpec(
        True, 21_600,
        "Inform the user the daily research budget is reached. "
        "Metadata tools remain available at no cost.",
        429,
    ),
    ErrorCode.RATE_LIMITED: ErrorSpec(
        True, 15,
        "Honour retry_after_s; exponential backoff, at most 3 attempts.",
        429,
    ),
    ErrorCode.UPSTREAM_QUOTA: ErrorSpec(
        True, 3_600,
        "Stop. Do not tight-loop. Inform the user that upstream quota is exhausted.",
        429,
    ),
    ErrorCode.TRANSCRIPTION_FAILED: ErrorSpec(
        True, 60,
        "Retry once; then report the failure and its reason to the user.",
        500,
    ),
    ErrorCode.JOB_NOT_FOUND: ErrorSpec(
        False, None,
        "The requested background job was not found.",
        404,
    ),
    ErrorCode.JOB_EXPIRED: ErrorSpec(
        False, None,
        "The job has expired. Please re-run the request.",
        410,
    ),
    ErrorCode.TRANSCRIPT_NOT_FOUND: ErrorSpec(
        False, None,
        "No analysis or transcript found. Use youtube_ask_video or youtube_analyze_video.",
        404,
    ),
    ErrorCode.CURSOR_EXPIRED: ErrorSpec(
        False, None,
        "Restart pagination from page 1 (omit the cursor).",
        400,
    ),
    ErrorCode.CATALOG_TOO_LARGE: ErrorSpec(
        False, None,
        "Use sort='newest' together with a published_after date filter.",
        400,
    ),
    ErrorCode.SEARCH_FAILED: ErrorSpec(
        True, 30,
        "Check video_id and query format.",
        500,
    ),
    ErrorCode.WINDOW_EXTRACTION_FAILED: ErrorSpec(
        True, 30,
        "Check start_seconds and end_seconds bounds.",
        500,
    ),
    ErrorCode.RIGHTS_ATTESTATION_REQUIRED: ErrorSpec(
        False, None,
        "Set attest_rights_confirmed=true to confirm you own this video, "
        "have permission, or it is licensed for this use. "
        "Alternatively, use youtube_analyze_video or youtube_ask_video for public video analysis.",
        403,
    ),
    ErrorCode.INTERNAL: ErrorSpec(
        True, 30,
        "Report the failure to the user; retry at most once.",
        500,
    ),
}


class ToolError(Exception):
    """A tool-level failure that serializes into a consistent error envelope."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retry_after_s: int | None = None,
        recovery_hint: str | None = None,
    ) -> None:
        super().__init__(message)
        spec = ERROR_SPECS[code]
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = spec.retryable
        self.retry_after_s = retry_after_s if retry_after_s is not None else spec.retry_after_s
        self.recovery_hint = recovery_hint or spec.recovery_hint
        self.http_status = spec.http_status

    def envelope(self) -> dict[str, Any]:
        """Consistent JSON shape returned from MCP tools."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "retryable": self.retryable,
                "retry_after_s": self.retry_after_s,
                "recovery_hint": self.recovery_hint,
                "details": self.details,
            }
        }

    def __repr__(self) -> str:
        return f"ToolError({self.code.value}: {self.message})"


def is_error_envelope(payload: object) -> bool:
    """Check if a payload is a ToolError envelope."""
    return isinstance(payload, dict) and "error" in payload and isinstance(payload["error"], dict)
