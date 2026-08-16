"""
Pytest Fixtures and Mock Initializers for TubeLens
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_channel_response() -> dict[str, Any]:
    """Loads sample channel response fixture."""
    file_path = FIXTURES_DIR / "sample_channel_response.json"
    return json.loads(file_path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_playlist_response() -> dict[str, Any]:
    """Loads sample playlist response fixture."""
    file_path = FIXTURES_DIR / "sample_playlist_response.json"
    return json.loads(file_path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_transcript_payload() -> dict[str, Any]:
    """Loads sample transcript fixture."""
    file_path = FIXTURES_DIR / "sample_transcript_payload.json"
    return json.loads(file_path.read_text(encoding="utf-8"))
