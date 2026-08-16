"""Tests for the VideoAnalysis model and analysis-mode data structures."""


from src.models import (
    CreatorAttribution,
    QuotedExcerpt,
    SpeakerInfo,
    TimestampEntry,
    VideoAnalysis,
)


class TestVideoAnalysisModel:
    """Verify the VideoAnalysis response model enforces mandatory fields."""

    def test_analysis_requires_attribution(self):
        """Every VideoAnalysis must include a CreatorAttribution."""
        analysis = VideoAnalysis(
            video_id="test_vid_001",
            duration_seconds=212,
            summary="Test summary of the video.",
            key_topics=["topic1", "topic2"],
            attribution=CreatorAttribution(
                channel_title="Test Channel",
                channel_id="UCtest123456",
                channel_url="https://www.youtube.com/channel/UCtest123456",
                video_url="https://www.youtube.com/watch?v=test_vid_001",
            ),
            generated_at="2025-01-01T00:00:00+00:00",
        )

        assert analysis.attribution.channel_title == "Test Channel"
        assert analysis.attribution.video_url == "https://www.youtube.com/watch?v=test_vid_001"
        assert "property of its owner" in analysis.attribution.notice

    def test_analysis_default_mode_is_public(self):
        """Default analysis_mode should be 'public'."""
        analysis = VideoAnalysis(
            video_id="test_vid_001",
            duration_seconds=100,
            summary="Test",
            key_topics=[],
            attribution=CreatorAttribution(
                channel_title="Test",
                channel_id="UCtest123456",
                channel_url="https://www.youtube.com/channel/UCtest123456",
                video_url="https://www.youtube.com/watch?v=test_vid_001",
            ),
            generated_at="2025-01-01T00:00:00+00:00",
        )

        assert analysis.analysis_mode == "public"

    def test_quoted_excerpt_has_youtube_link(self):
        """Each QuotedExcerpt must contain a direct YouTube timestamp link."""
        excerpt = QuotedExcerpt(
            text="This is a notable quote from the video.",
            speaker_name="John Doe",
            timestamp_seconds=120,
            timestamp_formatted="00:02:00",
            youtube_link="https://www.youtube.com/watch?v=test_vid_001&t=120",
        )

        assert excerpt.youtube_link.startswith("https://www.youtube.com/watch")
        assert "t=120" in excerpt.youtube_link
        assert len(excerpt.text.split()) <= 50  # Max ~50 words

    def test_timestamp_entry_structure(self):
        """TimestampEntry must map a topic to a timestamp with YouTube link."""
        entry = TimestampEntry(
            topic="Introduction and background",
            timestamp_seconds=0,
            timestamp_formatted="00:00:00",
            youtube_link="https://www.youtube.com/watch?v=test_vid_001&t=0",
        )

        assert entry.topic == "Introduction and background"
        assert entry.timestamp_seconds == 0

    def test_analysis_serialization_roundtrip(self):
        """VideoAnalysis should survive JSON serialization and deserialization."""
        analysis = VideoAnalysis(
            video_id="test_vid_001",
            title="Test Video",
            duration_seconds=212,
            summary="A comprehensive test summary.",
            key_topics=["AI", "Technology"],
            notable_quotes=[
                QuotedExcerpt(
                    text="This is interesting.",
                    speaker_name="Speaker 1",
                    timestamp_seconds=30,
                    timestamp_formatted="00:00:30",
                    youtube_link="https://www.youtube.com/watch?v=test_vid_001&t=30",
                )
            ],
            timestamp_index=[
                TimestampEntry(
                    topic="Introduction",
                    timestamp_seconds=0,
                    timestamp_formatted="00:00:00",
                    youtube_link="https://www.youtube.com/watch?v=test_vid_001&t=0",
                )
            ],
            speaker_registry=[SpeakerInfo(name="Speaker 1", role="Host")],
            attribution=CreatorAttribution(
                channel_title="Test Channel",
                channel_id="UCtest123456",
                channel_url="https://www.youtube.com/channel/UCtest123456",
                video_url="https://www.youtube.com/watch?v=test_vid_001",
            ),
            generated_at="2025-01-01T00:00:00+00:00",
        )

        data = analysis.model_dump()
        restored = VideoAnalysis.model_validate(data)

        assert restored.video_id == analysis.video_id
        assert restored.attribution.channel_title == "Test Channel"
        assert len(restored.notable_quotes) == 1
        assert len(restored.timestamp_index) == 1
