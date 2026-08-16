"""
Unit tests for Configuration (config.py) and Error Handling taxonomy (errors.py).
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.errors import ERROR_SPECS, ErrorCode, ErrorSpec, ToolError, is_error_envelope


class TestConfigSettings:
    """Test Settings configuration model and model_ladder property."""

    def test_model_ladder_default(self):
        settings = Settings(
            gemini_primary_model="gemini-3.7-flash",
            gemini_fallback_model="gemini-3.6-flash",
            gemini_fallback_model_2="",
        )
        ladder = settings.model_ladder
        assert ladder == ["gemini-3.7-flash", "gemini-3.6-flash"]

    def test_model_ladder_deduplication(self):
        settings = Settings(
            gemini_primary_model="gemini-3.7-flash",
            gemini_fallback_model="gemini-3.7-flash",
            gemini_fallback_model_2="gemini-3.6-flash",
        )
        ladder = settings.model_ladder
        assert ladder == ["gemini-3.7-flash", "gemini-3.6-flash"]

    def test_model_ladder_empty_raises_value_error(self):
        settings = Settings(
            gemini_primary_model="",
            gemini_fallback_model="",
            gemini_fallback_model_2="",
        )
        with pytest.raises(ValueError, match="At least one Gemini model must be configured"):
            _ = settings.model_ladder

    def test_pricing_settings_defaults(self):
        settings = Settings()
        assert settings.prompt_token_cost_per_million == 0.15
        assert settings.candidate_token_cost_per_million == 0.60


class TestErrorTaxonomy:
    """Verify completeness and behavior of ErrorCode and ToolError."""

    def test_all_error_codes_have_specs(self):
        for code in ErrorCode:
            assert code in ERROR_SPECS, f"ErrorCode {code.name} is missing a definition in ERROR_SPECS"
            spec = ERROR_SPECS[code]
            assert isinstance(spec, ErrorSpec)
            assert spec.http_status >= 400
            assert spec.recovery_hint, f"ErrorCode {code.name} has empty recovery_hint"

    def test_tool_error_envelope_structure(self):
        err = ToolError(
            code=ErrorCode.VIDEO_UNAVAILABLE,
            message="Video is private or restricted.",
        )
        assert err.code == ErrorCode.VIDEO_UNAVAILABLE
        assert err.message == "Video is private or restricted."
        assert err.recovery_hint == "Tell the user the video is not publicly accessible; suggest an alternative."
        assert err.retryable is False

        env = err.envelope()
        assert env["error"]["code"] == "VIDEO_UNAVAILABLE"
        assert env["error"]["message"] == "Video is private or restricted."
        assert env["error"]["recovery_hint"] == "Tell the user the video is not publicly accessible; suggest an alternative."
        assert env["error"]["retryable"] is False
        assert is_error_envelope(env) is True

    def test_is_error_envelope_falsy_on_regular_payload(self):
        assert is_error_envelope({"video_id": "123", "title": "Test"}) is False
        assert is_error_envelope("not a dict") is False
        assert is_error_envelope(None) is False


class TestFixturesValidity:
    """Ensure all sample fixtures deserialize and maintain expected schemas."""

    def test_sample_channel_fixture(self, sample_channel_response):
        assert "items" in sample_channel_response
        assert len(sample_channel_response["items"]) >= 1
        item = sample_channel_response["items"][0]
        assert "snippet" in item
        assert "contentDetails" in item

    def test_sample_playlist_fixture(self, sample_playlist_response):
        assert "items" in sample_playlist_response
        assert len(sample_playlist_response["items"]) >= 1
        item = sample_playlist_response["items"][0]
        assert "snippet" in item
        assert "contentDetails" in item

    def test_sample_transcript_fixture(self, sample_transcript_payload):
        assert sample_transcript_payload["video_id"] == "test_vid_001"
        assert len(sample_transcript_payload["turns"]) == 2
        assert sample_transcript_payload["turns"][0]["speaker_name"] == "Host"
