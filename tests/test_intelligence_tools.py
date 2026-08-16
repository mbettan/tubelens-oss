"""
Unit and integration tests for YouTube Intelligence Tools:
- youtube_extract_recommendations
- youtube_extract_claims
- youtube_evaluate_fit
- youtube_compare_videos
- Preflight ToolError taxonomy
- AuthMiddleware query redaction
- Ephemeral execution without disk cache
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import Response

from src.errors import ErrorCode, ToolError
from src.models import (
    CrossVideoComparisonResponse,
    PersonalizedFitAnalysis,
    VideoClaimsResponse,
    VideoDetailedMetadata,
    VideoRecommendationsResponse,
)
from src.server import (
    AuthMiddleware,
    youtube_compare_videos,
    youtube_evaluate_fit,
    youtube_extract_claims,
    youtube_extract_recommendations,
)
from src.transcribe import transcription_engine


@pytest.fixture
def mock_video_meta() -> VideoDetailedMetadata:
    return VideoDetailedMetadata(
        video_id="tst_vid_123",
        title="Comprehensive Indexing & Tax Strategy Guide",
        description="A deep dive into indexing strategies and tax arbitrage.",
        published_at="2026-01-01T00:00:00Z",
        duration_seconds=1800,
        duration_formatted="00:30:00",
        channel_id="UC_test_creator",
        channel_title="Finance Insights",
        is_available=True,
        is_live_content=False,
    )


class TestPreflightCheckTaxonomy:
    """Verifies H3: preflight_check raises structured ToolError with correct error codes."""

    @pytest.mark.asyncio
    async def test_preflight_nonexistent_video(self) -> None:
        with patch("src.youtube.youtube_client.get_videos_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = []
            with pytest.raises(ToolError) as exc_info:
                await transcription_engine.preflight_check("nonexistent_id")
            assert exc_info.value.code == ErrorCode.VIDEO_UNAVAILABLE
            assert "not found or unavailable" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_preflight_restricted_video(self, mock_video_meta: VideoDetailedMetadata) -> None:
        mock_video_meta.is_available = False
        mock_video_meta.privacy_status = "private"
        with patch("src.youtube.youtube_client.get_videos_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [mock_video_meta]
            with pytest.raises(ToolError) as exc_info:
                await transcription_engine.preflight_check("tst_vid_123")
            assert exc_info.value.code == ErrorCode.VIDEO_UNAVAILABLE
            assert "private, unlisted, or restricted" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_preflight_live_stream_rejected(self, mock_video_meta: VideoDetailedMetadata) -> None:
        mock_video_meta.is_live_content = True
        with patch("src.youtube.youtube_client.get_videos_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [mock_video_meta]
            with pytest.raises(ToolError) as exc_info:
                await transcription_engine.preflight_check("tst_vid_123")
            assert exc_info.value.code == ErrorCode.LIVE_STREAM_NOT_SUPPORTED

    @pytest.mark.asyncio
    async def test_preflight_exceeds_duration_limit(self, mock_video_meta: VideoDetailedMetadata) -> None:
        mock_video_meta.duration_seconds = 20000  # Exceeds 14400s
        with patch("src.youtube.youtube_client.get_videos_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [mock_video_meta]
            with pytest.raises(ToolError) as exc_info:
                await transcription_engine.preflight_check("tst_vid_123")
            assert exc_info.value.code == ErrorCode.VIDEO_TOO_LONG


class TestRecommendationExtraction:
    """Verifies youtube_extract_recommendations tool and parsing logic."""

    @pytest.mark.asyncio
    async def test_extract_recommendations_success(self, mock_video_meta: VideoDetailedMetadata) -> None:
        raw_llm_json = {
            "recommendations": [
                {
                    "entity_or_topic": "AVEM",
                    "action": "allocate",
                    "stance": "strongly_positive",
                    "conviction_level": "high",
                    "target_audience": "Value-tilted emerging market investors",
                    "core_thesis": "Avoids state-owned enterprise risk present in market-cap weighted EM funds",
                    "risks_or_tradeoffs": ["0.33% expense ratio", "Tracking error"],
                    "alternatives_mentioned": ["VWO", "DFEM"],
                    "evidence": [
                        {
                            "timestamp": "00:14:20",
                            "speaker_name": "Host",
                            "quote": "Market-cap weighting in EM gives too much state-owned risk.",
                        }
                    ],
                }
            ]
        }

        with (
            patch.object(transcription_engine, "preflight_check", new_callable=AsyncMock) as mock_preflight,
            patch.object(transcription_engine, "transcribe_window_direct", new_callable=AsyncMock) as mock_direct,
        ):
            mock_preflight.return_value = mock_video_meta
            mock_direct.return_value = (
                f"```json\n{json.dumps(raw_llm_json)}\n```",
                MagicMock(prompt_token_count=1500, candidates_token_count=600),
                2.5,
                "gemini-3.7-flash",
            )

            res = await youtube_extract_recommendations("https://www.youtube.com/watch?v=tst_vid_123")
            assert isinstance(res, VideoRecommendationsResponse)
            assert res.total_recommendations == 1
            rec = res.recommendations[0]
            assert rec.entity_or_topic == "AVEM"
            assert rec.action == "allocate"
            assert rec.stance == "strongly_positive"
            assert rec.conviction_level == "high"
            assert len(rec.evidence) == 1
            assert rec.evidence[0].timestamp_seconds == 860
            assert rec.evidence[0].youtube_link == "https://www.youtube.com/watch?v=tst_vid_123&t=860"
            assert res.attribution.channel_title == "Finance Insights"


class TestFactualClaimExtraction:
    """Verifies youtube_extract_claims tool."""

    @pytest.mark.asyncio
    async def test_extract_claims_success(self, mock_video_meta: VideoDetailedMetadata) -> None:
        raw_llm_json = {
            "claims": [
                {
                    "claim": "Direct indexing reduces capital gains taxes by 1.2% annualized in the first 5 years.",
                    "category": "data_or_metric",
                    "is_verifiable": True,
                    "creator_confidence": "high",
                    "verification_guidance": "Check academic backtests and Parametric tax-loss harvesting empirical whitepapers.",
                    "evidence": [
                        {
                            "timestamp": "00:08:30",
                            "speaker_name": "Analyst",
                            "quote": "Historical studies show about 120 basis points of tax alpha in the early years.",
                        }
                    ],
                }
            ]
        }

        with (
            patch.object(transcription_engine, "preflight_check", new_callable=AsyncMock) as mock_preflight,
            patch.object(transcription_engine, "transcribe_window_direct", new_callable=AsyncMock) as mock_direct,
        ):
            mock_preflight.return_value = mock_video_meta
            mock_direct.return_value = (
                f"```json\n{json.dumps(raw_llm_json)}\n```",
                MagicMock(prompt_token_count=1200, candidates_token_count=400),
                1.8,
                "gemini-3.7-flash",
            )

            res = await youtube_extract_claims("https://www.youtube.com/watch?v=tst_vid_123")
            assert isinstance(res, VideoClaimsResponse)
            assert res.total_claims == 1
            claim = res.claims[0]
            assert "Direct indexing reduces capital gains" in claim.claim
            assert claim.category == "data_or_metric"
            assert claim.is_verifiable is True
            assert "Parametric" in claim.verification_guidance
            assert claim.evidence[0].timestamp_seconds == 510


class TestPersonalizedFitEvaluation:
    """Verifies youtube_evaluate_fit tool."""

    @pytest.mark.asyncio
    async def test_evaluate_fit_success(self, mock_video_meta: VideoDetailedMetadata) -> None:
        raw_llm_json = {
            "relevance_score": 92,
            "verdict": "must_watch",
            "executive_summary": "Highly applicable for a high-income tech worker with taxable assets.",
            "applicable_points": ["Asset location rules for high marginal tax brackets", "Direct indexing threshold"],
            "conflicts_or_caveats": ["Assumes high state income taxes, but you reside in Washington State (0% tax)"],
            "custom_action_items": ["Model tax-loss harvesting alpha net of 0.25% SMA fee"],
            "evidence": [
                {
                    "timestamp": "00:05:10",
                    "speaker_name": "Speaker",
                    "quote": "If you are already in the 37% bracket, direct indexing is very advantageous.",
                }
            ],
        }

        with (
            patch.object(transcription_engine, "preflight_check", new_callable=AsyncMock) as mock_preflight,
            patch.object(transcription_engine, "transcribe_window_direct", new_callable=AsyncMock) as mock_direct,
        ):
            mock_preflight.return_value = mock_video_meta
            mock_direct.return_value = (
                f"```json\n{json.dumps(raw_llm_json)}\n```",
                MagicMock(prompt_token_count=1000, candidates_token_count=500),
                2.1,
                "gemini-3.7-flash",
            )

            res = await youtube_evaluate_fit(
                url="https://www.youtube.com/watch?v=tst_vid_123",
                user_profile="Software engineer earning $450k in Washington state",
                constraints=["Zero state income tax", "Prefers automated low-touch investments"],
            )
            assert isinstance(res, PersonalizedFitAnalysis)
            assert res.relevance_score == 92
            assert res.verdict == "must_watch"
            assert len(res.applicable_points) == 2
            assert len(res.conflicts_or_caveats) == 1
            assert "Washington State" in res.conflicts_or_caveats[0]


class TestCrossVideoComparison:
    """Verifies youtube_compare_videos tool."""

    @pytest.mark.asyncio
    async def test_compare_videos_invalid_count(self) -> None:
        res = await youtube_compare_videos(urls=["https://www.youtube.com/watch?v=vid1"], topic="Indexing")
        assert "error" in res
        assert res["error"]["code"] == ErrorCode.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_compare_videos_success(self, mock_video_meta: VideoDetailedMetadata) -> None:
        raw_llm_json = {
            "points_of_consensus": [
                "Direct indexing generates most tax alpha in years 1-5.",
                "Tracking error against standard cap-weighted indices is a real factor to manage.",
            ],
            "disagreements": [
                {
                    "dimension": "Minimum Viable Portfolio Size",
                    "creator_positions": {
                        "Ben Felix": "Requires >$1M-$2M to overcome fee drag",
                        "Rob Berger": "Viable above $250k on automated platforms",
                    },
                    "root_cause_of_divergence": "Ben assumes full-service AUM fees, whereas Rob evaluates low-cost digital platforms.",
                    "synthesis_verdict": "Use digital direct indexing if under $1M; avoid full-service AUM under $2M.",
                }
            ],
            "recommended_playbook": "1. If in top tax bracket with >$500k: consider Direct Indexing. 2. If portfolio is <$500k or tax-advantaged: stick to low-cost broad-market ETFs.",
        }

        with (
            patch.object(transcription_engine, "preflight_check", new_callable=AsyncMock) as mock_preflight,
            patch.object(transcription_engine, "_get_client") as mock_get_client,
        ):
            mock_preflight.return_value = mock_video_meta
            mock_client = MagicMock()
            mock_stream_chunk = MagicMock()
            mock_stream_chunk.text = f"```json\n{json.dumps(raw_llm_json)}\n```"
            mock_stream_chunk.usage_metadata = MagicMock(prompt_token_count=3000, candidates_token_count=800)
            mock_client.models.generate_content_stream.return_value = [mock_stream_chunk]
            mock_get_client.return_value = mock_client

            res = await youtube_compare_videos(
                urls=[
                    "https://www.youtube.com/watch?v=vid11111111",
                    "https://www.youtube.com/watch?v=vid22222222",
                ],
                topic="Direct Indexing vs ETFs",
            )
            assert isinstance(res, CrossVideoComparisonResponse)
            assert len(res.points_of_consensus) == 2
            assert len(res.disagreements) == 1
            assert res.disagreements[0].dimension == "Minimum Viable Portfolio Size"
            assert "Ben assumes full-service" in res.disagreements[0].root_cause_of_divergence
            assert len(res.attributions) == 2


class TestAuthMiddlewareQueryRedaction:
    """Verifies M4: sensitive query parameters are redacted before logging."""

    @pytest.mark.asyncio
    async def test_auth_middleware_redacts_sensitive_params(self) -> None:
        async def dummy_app(scope, receive, send):
            response = Response("OK", media_type="text/plain")
            await response(scope, receive, send)

        middleware = AuthMiddleware(dummy_app)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/healthz",
            "query_string": b"api_key=SECRET_TOKEN_123&user=alice",
            "headers": [(b"user-agent", b"pytest")],
            "client": ("127.0.0.1", 12345),
        }

        with patch("src.server.logger.info") as mock_logger_info:
            await middleware(scope, AsyncMock(), AsyncMock())
            http_in_logs = [call.args[0] for call in mock_logger_info.call_args_list if "HTTP IN" in call.args[0]]
            assert len(http_in_logs) == 1
            log_msg = http_in_logs[0]
            assert "SECRET_TOKEN_123" not in log_msg
            assert "[REDACTED]" in log_msg
            assert "'user': 'alice'" in log_msg


class TestJsonExtractionRobustness:
    """Verifies N1 & N8: robust JSON extraction using raw_decode handling nested structures."""

    def test_extract_json_with_nested_objects_in_fences(self) -> None:
        from src.transcribe import _extract_json_block

        text = """Here is the response:
```json
{
  "claims": [
    {
      "claim": "Direct indexing",
      "evidence": [{"timestamp": "00:01:00", "quote": "Quote text"}]
    }
  ]
}
```
Hope this helps!"""
        parsed = _extract_json_block(text)
        assert parsed is not None
        assert "claims" in parsed
        assert len(parsed["claims"]) == 1
        assert parsed["claims"][0]["evidence"][0]["quote"] == "Quote text"

    def test_extract_json_raw_without_fences_with_surrounding_text(self) -> None:
        from src.transcribe import _extract_json_block

        text = 'Analysis: {"points_of_consensus": ["Point 1", "Point 2"], "recommended_playbook": "Action"} End of response.'
        parsed = _extract_json_block(text)
        assert parsed is not None
        assert len(parsed["points_of_consensus"]) == 2
        assert parsed["recommended_playbook"] == "Action"

    def test_extract_json_invalid_returns_none(self) -> None:
        from src.transcribe import _extract_json_block

        assert _extract_json_block("No json in this response.") is None
        assert _extract_json_block("Malformed: {broken json without close") is None


class TestCostCalculationHelper:
    """Verifies N10: compute_gemini_cost helper accurately calculates input/output token costs."""

    def test_compute_cost_with_valid_usage(self) -> None:
        from src.transcribe import compute_gemini_cost

        usage = MagicMock(prompt_token_count=1_000_000, candidates_token_count=1_000_000)
        p, c, cost = compute_gemini_cost(usage)
        assert p == 1_000_000
        assert c == 1_000_000
        assert cost == 0.75  # $0.15 + $0.60

    def test_compute_cost_with_none_usage(self) -> None:
        from src.transcribe import compute_gemini_cost

        p, c, cost = compute_gemini_cost(None)
        assert p == 0
        assert c == 0
        assert cost == 0.0
