"""
Extended tests for transcript validation, WPM calculation, speaker normalization, and window stitching in validators.py.
"""

from __future__ import annotations

from src.models import TranscriptTurn
from src.validators import (
    calculate_wpm,
    canonicalize_speakers,
    check_timestamp_coverage,
    stitch_transcript_windows,
    validate_transcript_quality,
)


class TestValidatorsWpmAndCoverage:
    """Test WPM calculation and timestamp coverage ratio."""

    def test_calculate_wpm_normal(self):
        turns = [
            TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="Alice", text="Hello world and welcome to this video."),
            TranscriptTurn(timestamp_seconds=30, timestamp_formatted="00:00:30", speaker_name="Alice", text="We are discussing modern index investing strategies."),
        ]
        # 14 words over 60 seconds = 14 WPM
        wpm = calculate_wpm(turns, 60)
        assert wpm == 14.0

    def test_calculate_wpm_zero_duration(self):
        turns = [TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="A", text="Word")]
        wpm = calculate_wpm(turns, 0)
        assert wpm == 0.0

    def test_check_timestamp_coverage(self):
        turns = [
            TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="A", text="Start"),
            TranscriptTurn(timestamp_seconds=500, timestamp_formatted="00:08:20", speaker_name="A", text="End"),
        ]
        coverage = check_timestamp_coverage(turns, 1000)
        assert coverage == 0.5


class TestValidateTranscriptQuality:
    """Test multilingual and threshold behaviors of validate_transcript_quality."""

    def test_empty_turns_fails(self):
        res = validate_transcript_quality([], duration_seconds=600)
        assert res.status == "failed"
        assert "EMPTY_TRANSCRIPT" in res.flags

    def test_cjk_language_cpm(self):
        turns = [
            TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="A", text="这是一个关于投资和指数基金的中文视频。"),
            TranscriptTurn(timestamp_seconds=30, timestamp_formatted="00:00:30", speaker_name="A", text="今天我们将详细讨论直接索引策略。"),
        ]
        res = validate_transcript_quality(turns, duration_seconds=60, language="zh")
        assert res.wpm > 0
        assert res.status == "degraded"  # low coverage (30/60)

    def test_repetition_with_low_coverage_fails(self):
        looping_text = "This is a degenerate loop repeating exactly the same sentence over and over again without stopping."
        turns = [
            TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="A", text=looping_text),
            TranscriptTurn(timestamp_seconds=5, timestamp_formatted="00:00:05", speaker_name="A", text=looping_text),
            TranscriptTurn(timestamp_seconds=10, timestamp_formatted="00:00:10", speaker_name="A", text=looping_text),
            TranscriptTurn(timestamp_seconds=15, timestamp_formatted="00:00:15", speaker_name="A", text=looping_text),
        ]
        # Duration is 1000s, max timestamp is 15s (coverage = 0.015 < 0.4)
        res = validate_transcript_quality(turns, duration_seconds=1000)
        assert res.status == "failed"
        assert "REPETITION_DETECTED" in res.flags


class TestCanonicalizeSpeakers:
    """Test speaker name normalization and deduplication."""

    def test_merges_first_name_to_full_name(self):
        turns = [
            TranscriptTurn(timestamp_seconds=0, timestamp_formatted="00:00:00", speaker_name="Thomas Kopelman", text="Hello everyone."),
            TranscriptTurn(timestamp_seconds=10, timestamp_formatted="00:00:10", speaker_name="Thomas", text="Welcome back."),
            TranscriptTurn(timestamp_seconds=20, timestamp_formatted="00:00:20", speaker_name="Sarah", text="Hi Thomas."),
        ]
        norm_turns, registry = canonicalize_speakers(turns)
        assert norm_turns[0].speaker_name == "Thomas Kopelman"
        assert norm_turns[1].speaker_name == "Thomas Kopelman"
        assert norm_turns[2].speaker_name == "Sarah"

        names = [s.name for s in registry]
        assert "Thomas Kopelman" in names
        assert "Sarah" in names
        assert "Thomas" not in names


class TestStitchTranscriptWindows:
    """Test stitching of windowed transcript turns with temporal and textual overlap matching."""

    def test_stitch_with_temporal_overlap(self):
        window1 = [
            TranscriptTurn(timestamp_seconds=1780, timestamp_formatted="00:29:40", speaker_name="Host", text="Let us summarize direct indexing advantages."),
            TranscriptTurn(timestamp_seconds=1795, timestamp_formatted="00:29:55", speaker_name="Host", text="First, tax loss harvesting alpha."),
        ]
        window2 = [
            TranscriptTurn(timestamp_seconds=1795, timestamp_formatted="00:29:55", speaker_name="Host", text="First, tax loss harvesting alpha."),
            TranscriptTurn(timestamp_seconds=1810, timestamp_formatted="00:30:10", speaker_name="Host", text="Second, personalized ESG customization."),
        ]
        merged = stitch_transcript_windows([window1, window2], overlap_seconds=20)
        assert len(merged) == 3
        assert merged[0].text == "Let us summarize direct indexing advantages."
        assert merged[1].text == "First, tax loss harvesting alpha."
        assert merged[2].text == "Second, personalized ESG customization."

    def test_stitch_fallback_by_timestamp_when_distant(self):
        window1 = [
            TranscriptTurn(timestamp_seconds=100, timestamp_formatted="00:01:40", speaker_name="Host", text="Alpha text."),
        ]
        window2 = [
            TranscriptTurn(timestamp_seconds=50, timestamp_formatted="00:00:50", speaker_name="Host", text="Old text."),
            TranscriptTurn(timestamp_seconds=200, timestamp_formatted="00:03:20", speaker_name="Host", text="Beta text."),
        ]
        merged = stitch_transcript_windows([window1, window2], overlap_seconds=20)
        assert len(merged) == 2
        assert merged[0].timestamp_seconds == 100
        assert merged[1].timestamp_seconds == 200
