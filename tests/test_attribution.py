"""Tests for mandatory creator attribution in analysis outputs."""

import pytest

from src.models import CreatorAttribution, VideoAnalysis


class TestCreatorAttribution:
    """Verify creator attribution is mandatory and correct."""

    def test_attribution_has_required_fields(self):
        """CreatorAttribution must include channel info and video link."""
        attr = CreatorAttribution(
            channel_title="Tech Channel",
            channel_id="UCtest123456789012345678",
            channel_url="https://www.youtube.com/channel/UCtest123456789012345678",
            video_url="https://www.youtube.com/watch?v=test_vid_001",
        )

        assert attr.channel_title == "Tech Channel"
        assert attr.channel_id.startswith("UC")
        assert "youtube.com/channel/" in attr.channel_url
        assert "youtube.com/watch" in attr.video_url
        assert attr.notice  # Must have a non-empty notice

    def test_default_notice_text(self):
        """Default attribution notice should mention creator property."""
        attr = CreatorAttribution(
            channel_title="Test",
            channel_id="UCtest123456789012345678",
            channel_url="https://www.youtube.com/channel/UCtest123456789012345678",
            video_url="https://www.youtube.com/watch?v=test_vid_001",
        )

        assert "property of its owner" in attr.notice.lower()

    def test_analysis_without_attribution_fails_validation(self):
        """VideoAnalysis should fail pydantic validation without attribution."""
        with pytest.raises(Exception):  # ValidationError
            VideoAnalysis(
                video_id="test_vid_001",
                duration_seconds=100,
                summary="Test summary",
                key_topics=["test"],
                # attribution is missing — should fail
                generated_at="2025-01-01T00:00:00+00:00",
            )

    def test_attribution_url_formats(self):
        """URLs should follow canonical YouTube URL patterns."""
        attr = CreatorAttribution(
            channel_title="Creator Name",
            channel_id="UCabcdef123456789012345",
            channel_url="https://www.youtube.com/channel/UCabcdef123456789012345",
            video_url="https://www.youtube.com/watch?v=test_vid_001",
        )

        # Channel URL format
        assert attr.channel_url == f"https://www.youtube.com/channel/{attr.channel_id}"
        # Video URL format
        assert attr.video_url.startswith("https://www.youtube.com/watch?v=")
