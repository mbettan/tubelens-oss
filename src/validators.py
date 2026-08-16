"""
Quality Validators, Repetition Loop Detectors & Boundary Stitching for Transcripts
"""

from __future__ import annotations

import difflib
import re
from typing import Literal

from src.models import SpeakerInfo, TranscriptTurn, ValidationResult


def calculate_wpm(turns: list[TranscriptTurn], duration_seconds: float) -> float:
    """Calculate words per minute (WPM) across all transcript turns."""
    if duration_seconds <= 0 or not turns:
        return 0.0
    total_words = sum(len(turn.text.split()) for turn in turns)
    minutes = duration_seconds / 60.0
    return round(total_words / minutes, 2)


def check_timestamp_coverage(
    turns: list[TranscriptTurn], expected_duration_seconds: float
) -> float:
    """Check how much of the video duration is covered by timestamps."""
    if expected_duration_seconds <= 0 or not turns:
        return 0.0
    last_timestamp = max(turn.timestamp_seconds for turn in turns)
    return round(min(1.0, last_timestamp / expected_duration_seconds), 4)


def detect_repetition_loops(
    turns: list[TranscriptTurn], n_gram_size: int = 10, threshold: int = 3, window_words: int = 300
) -> bool:
    """
    Detects degenerate LLM repetition loops by checking if any n-gram of words
    repeats consecutively or with high frequency within a localized window.
    """
    all_words: list[str] = []
    for turn in turns:
        all_words.extend(re.findall(r"\b\w+\b", turn.text.lower()))

    if len(all_words) < n_gram_size * 2:
        return False

    # Check for consecutive or localized repetition loops
    for i in range(len(all_words) - n_gram_size + 1):
        target_ngram = tuple(all_words[i : i + n_gram_size])
        count = 1
        # Scan ahead within a localized window
        search_limit = min(len(all_words) - n_gram_size + 1, i + window_words)
        j = i + n_gram_size
        while j < search_limit:
            candidate = tuple(all_words[j : j + n_gram_size])
            if candidate == target_ngram:
                count += 1
                if count >= threshold:
                    return True
                j += n_gram_size  # Jump ahead
            else:
                j += 1

    return False


def validate_transcript_quality(
    turns: list[TranscriptTurn],
    duration_seconds: float,
    min_wpm: float = 40.0,
    max_wpm: float = 320.0,
    min_coverage_ratio: float = 0.85,
    language: str = "en",
) -> ValidationResult:
    """
    Validates the structural quality of a transcript against WPM, coverage, and repetition loops.
    Supports language-aware WPM bands:
      - Latin (en, es, fr, de, pt, it, ...): 40–320 WPM (words per minute)
      - CJK  (zh, ja, ko): 100–700 CPM (characters per minute)
      - RTL  (ar, he, fa): 40–300 WPM (words per minute)
    """
    flags: list[str] = []
    if not turns:
        return ValidationResult(
            status="failed",
            wpm=0.0,
            coverage_ratio=0.0,
            flags=["EMPTY_TRANSCRIPT"],
        )

    # Language-aware WPM/CPM bands (use language defaults if default arguments provided)
    lang_prefix = language[:2].lower() if language else "en"
    if min_wpm == 40.0 and max_wpm == 320.0:
        if lang_prefix in ("zh", "ja", "ko"):
            # CJK: measure characters per minute instead of words
            total_chars = sum(len(turn.text) for turn in turns)
            minutes = max(duration_seconds / 60.0, 0.01)
            wpm = round(total_chars / minutes, 2)
            min_wpm, max_wpm = 100.0, 700.0
        elif lang_prefix in ("ar", "he", "fa"):
            # RTL: slightly narrower WPM range
            wpm = calculate_wpm(turns, duration_seconds)
            min_wpm, max_wpm = 40.0, 300.0
        else:
            # Latin-script default
            wpm = calculate_wpm(turns, duration_seconds)
    else:
        wpm = calculate_wpm(turns, duration_seconds)

    coverage = check_timestamp_coverage(turns, duration_seconds)
    has_loop = detect_repetition_loops(turns)

    if has_loop:
        flags.append("REPETITION_DETECTED")
    if wpm < min_wpm:
        flags.append(f"LOW_WPM_{wpm}")
    elif wpm > max_wpm:
        flags.append(f"HIGH_WPM_{wpm}")
    if coverage < min_coverage_ratio:
        flags.append(f"LOW_COVERAGE_{int(coverage * 100)}PCT")

    status: Literal["passed", "degraded", "failed"] = "passed"
    if has_loop and coverage < 0.4:
        status = "failed"
    elif flags:
        status = "degraded"

    return ValidationResult(
        status=status,
        wpm=wpm,
        coverage_ratio=coverage,
        flags=flags,
    )


def canonicalize_speakers(
    turns: list[TranscriptTurn],
) -> tuple[list[TranscriptTurn], list[SpeakerInfo]]:
    """
    Normalizes speaker names by mapping short first-name mentions to full canonical names
    (e.g., merging 'Thomas' to 'Thomas Kopelman' if both appear).
    """
    # Collect unique speaker names from turns
    raw_names: set[str] = {turn.speaker_name.strip() for turn in turns if turn.speaker_name.strip()}
    # Sort by length descending, then alphabetically for deterministic order
    sorted_names = sorted(raw_names, key=lambda n: (-len(n), n))

    name_mapping: dict[str, str] = {}
    canonical_names: list[str] = []

    for name in sorted_names:
        matching_canonicals = [
            canonical
            for canonical in canonical_names
            if name in canonical.split() or canonical.startswith(name + " ") or canonical.endswith(" " + name)
        ]
        # Only merge if exactly ONE canonical name matches unambiguously
        if len(matching_canonicals) == 1:
            name_mapping[name] = matching_canonicals[0]
        else:
            name_mapping[name] = name
            if name not in canonical_names:
                canonical_names.append(name)

    normalized_turns: list[TranscriptTurn] = []
    for turn in turns:
        normalized_name = name_mapping.get(turn.speaker_name.strip(), turn.speaker_name.strip())
        normalized_turns.append(
            TranscriptTurn(
                timestamp_seconds=turn.timestamp_seconds,
                timestamp_formatted=turn.timestamp_formatted,
                speaker_name=normalized_name,
                text=turn.text,
            )
        )

    speaker_registry = [SpeakerInfo(name=name) for name in sorted(canonical_names)]
    return normalized_turns, speaker_registry


def stitch_transcript_windows(
    window_turns_list: list[list[TranscriptTurn]],
    overlap_seconds: int = 20,
) -> list[TranscriptTurn]:
    """
    Stitches multiple 30-minute transcript windows together using difflib SequenceMatcher
    and temporal proximity on the overlap boundary to remove duplicate turns across window seams.
    """
    if not window_turns_list:
        return []
    if len(window_turns_list) == 1:
        return window_turns_list[0]

    merged_turns: list[TranscriptTurn] = list(window_turns_list[0])

    for i in range(1, len(window_turns_list)):
        current_window = window_turns_list[i]
        if not current_window:
            continue
        if not merged_turns:
            merged_turns.extend(current_window)
            continue

        tail_turns = merged_turns[-10:]
        head_turns = current_window[:10]

        best_match_idx_curr = 0
        best_ratio = 0.0

        for head_idx, head_turn in enumerate(head_turns):
            head_text = head_turn.text.strip().lower()
            for tail_turn in tail_turns:
                # Require temporal proximity to prevent accidental false matches across distant topics
                if abs(head_turn.timestamp_seconds - tail_turn.timestamp_seconds) <= (overlap_seconds + 30):
                    tail_text = tail_turn.text.strip().lower()
                    matcher = difflib.SequenceMatcher(None, tail_text, head_text)
                    ratio = matcher.ratio()
                    if ratio > 0.7 and ratio > best_ratio:
                        best_ratio = ratio
                        best_match_idx_curr = head_idx + 1

        if best_ratio > 0.7:
            turns_to_append = current_window[best_match_idx_curr:]
        else:
            # Fallback: deduplicate strictly by timestamp
            last_timestamp = merged_turns[-1].timestamp_seconds
            turns_to_append = [t for t in current_window if t.timestamp_seconds > last_timestamp]

        merged_turns.extend(turns_to_append)

    return merged_turns
