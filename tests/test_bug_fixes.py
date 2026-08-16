"""Comprehensive tests verifying bug fixes across youtube extraction, repetition detection, and server tools."""

import pytest

from src.models import (
    TranscriptTurn,
)
from src.validators import (
    detect_repetition_loops,
    stitch_transcript_windows,
)
from src.youtube import extract_video_id


class TestYouTubeURLExtraction:
    """Verify extract_video_id handles /live/, /v/, /shorts/, youtu.be, and standard URLs."""

    def test_extract_live_url(self):
        assert extract_video_id("https://www.youtube.com/live/tst_vid_001") == "tst_vid_001"
        assert extract_video_id("https://youtube.com/live/tst_vid_001?feature=share") == "tst_vid_001"

    def test_extract_v_url(self):
        assert extract_video_id("https://www.youtube.com/v/tst_vid_001") == "tst_vid_001"

    def test_extract_standard_urls(self):
        assert extract_video_id("https://www.youtube.com/watch?v=tst_vid_001") == "tst_vid_001"
        assert extract_video_id("https://youtu.be/tst_vid_001") == "tst_vid_001"
        assert extract_video_id("https://www.youtube.com/shorts/tst_vid_001") == "tst_vid_001"
        assert extract_video_id("tst_vid_001") == "tst_vid_001"


class TestWindowStitchingFallback:
    """Verify window stitching deduplication when similarity ratio is low."""

    def test_stitching_deduplicates_by_timestamp_on_low_similarity(self):
        window_1 = [
            TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="A", text="Hello and welcome."),
            TranscriptTurn(timestamp_seconds=15, timestamp_formatted="00:00:15", speaker_name="B", text="Thanks for having me."),
            TranscriptTurn(timestamp_seconds=1800, timestamp_formatted="00:30:00", speaker_name="A", text="End of part one."),
        ]
        window_2 = [
            # Overlap turn with slightly different transcription text (low diff ratio)
            TranscriptTurn(timestamp_seconds=1795, timestamp_formatted="00:29:55", speaker_name="A", text="Completely rephrased turn."),
            TranscriptTurn(timestamp_seconds=1805, timestamp_formatted="00:30:05", speaker_name="B", text="Start of part two."),
        ]
        stitched = stitch_transcript_windows([window_1, window_2], overlap_seconds=20)
        # Should not duplicate turns prior to 1800s from window_2
        assert len(stitched) == 4
        assert stitched[-1].text == "Start of part two."
        assert stitched[-1].timestamp_seconds == 1805


class TestRepetitionLoopLocality:
    """Verify repetition detector catches localized loops without false-positive triggers on distant phrases."""

    def test_consecutive_repetition_loop_detected(self):
        repeated_text = "we are going to talk about the system architecture and its design today " * 5
        turns = [
            TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="Host", text=repeated_text),
        ]
        assert detect_repetition_loops(turns, n_gram_size=8, threshold=3) is True

    def test_distant_phrases_do_not_trigger_loop(self):
        # A standard phrase spoken once in the intro and once in the outro
        intro = "Thank you so much for joining us on the podcast today it is a pleasure to have you here."
        middle = " ".join(f"unique_word_{i} discussing topic number {i} with distinct concepts" for i in range(100))
        outro = "Thank you so much for joining us on the podcast today it is a pleasure to have you here."
        turns = [
            TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="Host", text=intro),
            TranscriptTurn(timestamp_seconds=100, timestamp_formatted="00:01:40", speaker_name="Guest", text=middle),
            TranscriptTurn(timestamp_seconds=3600, timestamp_formatted="01:00:00", speaker_name="Host", text=outro),
        ]
        assert detect_repetition_loops(turns, n_gram_size=10, threshold=3, window_words=100) is False


class TestServerAskVideoTool:
    """Verify server tool definition for youtube_ask_video."""

    @pytest.mark.asyncio
    async def test_ask_video_invalid_url_returns_error(self):
        from src.errors import ErrorCode
        from src.server import youtube_ask_video

        result = await youtube_ask_video(url="not-a-youtube-url", query="Test question?")
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"]["code"] == ErrorCode.INVALID_INPUT
