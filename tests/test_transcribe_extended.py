"""
Extended unit tests for Gemini Multimodal Transcription & Intelligence Engine in transcribe.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from src.models import VideoDetailedMetadata
from src.transcribe import (
    TranscriptionEngine,
    _is_model_fallback_eligible,
    parse_timestamp_to_seconds,
)


class TestTimestampParsingAndFallbackSignals:
    """Test timestamp string parsing and error classification."""

    def test_parse_timestamp_to_seconds(self):
        assert parse_timestamp_to_seconds("01:30:15") == 5415
        assert parse_timestamp_to_seconds("05:30") == 330
        assert parse_timestamp_to_seconds("[00:10:05]") == 605
        assert parse_timestamp_to_seconds("125.5") == 125
        assert parse_timestamp_to_seconds("invalid") == 0
        assert parse_timestamp_to_seconds("") == 0

    def test_is_model_fallback_eligible(self):
        assert _is_model_fallback_eligible(Exception("ResourceExhausted: 429 Quota exceeded")) is True
        assert _is_model_fallback_eligible(Exception("NotFound: 404 Model gemini-3.6-flash not found")) is True
        assert _is_model_fallback_eligible(Exception("DeadlineExceeded: 504 Timeout")) is True
        assert _is_model_fallback_eligible(ValueError("Malformed prompt")) is False


class TestAnalysisPromptAndParser:
    """Test analysis prompt construction and structured markdown output parser."""

    @pytest.fixture
    def engine(self) -> TranscriptionEngine:
        return TranscriptionEngine()

    @pytest.fixture
    def mock_meta(self) -> VideoDetailedMetadata:
        return VideoDetailedMetadata(
            video_id="test_vid_xyz",
            title="Comprehensive Portfolio Guide",
            description="Deep analysis on asset allocation.",
            published_at="2026-01-01T00:00:00Z",
            duration_seconds=1200,
            duration_formatted="00:20:00",
            channel_id="UC_finance",
            channel_title="Wealth Insights",
            is_available=True,
            is_live_content=False,
        )

    def test_build_analysis_prompt_with_focus(self, engine: TranscriptionEngine):
        prompt = engine._build_analysis_prompt(focus_question="What about international equities?")
        assert "## SUMMARY" in prompt
        assert "## KEY TOPICS" in prompt
        assert "## NOTABLE QUOTES" in prompt
        assert "## TIMESTAMP INDEX" in prompt
        assert "## SPEAKERS" in prompt
        assert "<focus_question>" in prompt
        assert "What about international equities?" in prompt

    def test_parse_analysis_response_full_sections(
        self, engine: TranscriptionEngine, mock_meta: VideoDetailedMetadata
    ):
        raw_output = """
## SUMMARY
This video provides an in-depth framework for optimizing tax-efficient portfolios.

## KEY TOPICS
- Asset location between taxable and Roth accounts
- Direct indexing alpha vs standard ETFs
- Rebalancing tax triggers

## NOTABLE QUOTES
[00:04:15] Jane Doe: "Direct indexing enables customized loss harvesting without sacrificing market exposure."

## TIMESTAMP INDEX
[00:01:00] Introduction to Asset Location
[00:08:30] Tax-Loss Harvesting Case Study

## SPEAKERS
- Jane Doe (Chief Investment Officer)
"""
        analysis = engine._parse_analysis_response(raw_output, mock_meta)
        assert analysis.video_id == "test_vid_xyz"
        assert "in-depth framework" in analysis.summary
        assert len(analysis.key_topics) == 3
        assert analysis.key_topics[0] == "Asset location between taxable and Roth accounts"
        assert len(analysis.notable_quotes) == 1
        assert analysis.notable_quotes[0].speaker_name == "Jane Doe"
        assert analysis.notable_quotes[0].timestamp_seconds == 255
        assert len(analysis.timestamp_index) == 2
        assert analysis.timestamp_index[0].timestamp_seconds == 60
        assert len(analysis.speaker_registry) == 1
        assert analysis.speaker_registry[0].name == "Jane Doe"
        assert analysis.speaker_registry[0].role == "Chief Investment Officer"
        assert analysis.attribution.channel_title == "Wealth Insights"


class TestModelFallbackExecution:
    """Test model fallback ladder execution in _generate_with_fallback."""

    @pytest.mark.asyncio
    async def test_fallback_ladder_recovers_on_second_model(self):
        engine = TranscriptionEngine()
        mock_client = MagicMock()

        # First model fails with 404 / NotFound, second model succeeds
        mock_resp_chunk = MagicMock(text="Analyzed response text", usage_metadata=MagicMock(prompt_token_count=100, candidates_token_count=50))

        def generate_content_stream_mock(model: str, contents: list, config: types.GenerateContentConfig):
            if model == "gemini-3.7-flash":
                raise Exception("NotFound: 404 Model not found")
            return [mock_resp_chunk]

        mock_client.models.generate_content_stream = generate_content_stream_mock

        with patch.object(engine, "_get_client", return_value=mock_client):
            raw_text, usage, latency, actual_model = await engine._generate_with_fallback(
                contents=[],
                config=types.GenerateContentConfig(),
                timeout_seconds=5.0,
            )
            assert raw_text == "Analyzed response text"
            assert actual_model == "gemini-3.6-flash"
            assert usage.prompt_token_count == 100
