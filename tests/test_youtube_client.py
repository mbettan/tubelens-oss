"""
Unit tests for YouTube Data API v3 client functions:
- parse_iso8601_duration
- format_seconds_to_hhmmss
- derive_uploads_playlist_id
- YouTubeClient channel resolution, catalog browsing, playlist enumeration, batch metadata
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import (
    ChannelResolutionResult,
    PlaylistVideosPage,
    VideoCard,
    VideoDetailedMetadata,
)
from src.youtube import (
    YouTubeClient,
    derive_uploads_playlist_id,
    extract_video_id,
    format_seconds_to_hhmmss,
    parse_iso8601_duration,
)


class TestYouTubeHelpers:
    """Test pure helper functions in youtube.py."""

    def test_parse_iso8601_duration_various_formats(self):
        assert parse_iso8601_duration("PT1H2M3S") == 3723
        assert parse_iso8601_duration("PT5M") == 300
        assert parse_iso8601_duration("PT30S") == 30
        assert parse_iso8601_duration("PT2H") == 7200
        assert parse_iso8601_duration("P1DT2H3M4S") == 86400 + 7200 + 180 + 4
        assert parse_iso8601_duration("P1W") == 7 * 86400
        assert parse_iso8601_duration("INVALID") == 0
        assert parse_iso8601_duration("") == 0

    def test_format_seconds_to_hhmmss(self):
        assert format_seconds_to_hhmmss(0) == "00:00:00"
        assert format_seconds_to_hhmmss(45) == "00:00:45"
        assert format_seconds_to_hhmmss(75) == "00:01:15"
        assert format_seconds_to_hhmmss(3665) == "01:01:05"
        assert format_seconds_to_hhmmss(-10) == "00:00:00"

    def test_derive_uploads_playlist_id(self):
        assert derive_uploads_playlist_id("UC1234567890abcdef") == "UU1234567890abcdef"
        assert derive_uploads_playlist_id("HC1234567890abcdef") == "UUHC1234567890abcdef"

    def test_extract_video_id(self):
        assert extract_video_id("https://www.youtube.com/watch?v=tst_vid_001") == "tst_vid_001"
        assert extract_video_id("https://youtu.be/tst_vid_001?t=10") == "tst_vid_001"
        assert extract_video_id("https://www.youtube.com/shorts/tst_vid_001") == "tst_vid_001"
        assert extract_video_id("https://www.youtube.com/embed/tst_vid_001") == "tst_vid_001"
        assert extract_video_id("tst_vid_001") == "tst_vid_001"
        assert extract_video_id("not a valid url at all") is None


class TestYouTubeClientAsyncMethods:
    """Test YouTubeClient methods with mocked Google API Resource."""

    @pytest.fixture
    def client(self) -> YouTubeClient:
        return YouTubeClient(api_key="fake-test-key")

    @pytest.mark.asyncio
    async def test_resolve_channel_by_id(self, client: YouTubeClient):
        mock_response = {
            "items": [
                {
                    "id": "UC1234567890123456789012",
                    "snippet": {
                        "title": "Finance Test Channel",
                        "description": "Educational content",
                        "customUrl": "@financetest",
                    },
                    "contentDetails": {
                        "relatedPlaylists": {"uploads": "UU1234567890123456789012"}
                    },
                    "statistics": {
                        "subscriberCount": "100000",
                        "videoCount": "250",
                        "viewCount": "5000000",
                    },
                }
            ]
        }

        mock_service = MagicMock()
        mock_service.channels().list().execute.return_value = mock_response
        client._service = mock_service

        result = await client.resolve_channel("UC1234567890123456789012")
        assert isinstance(result, ChannelResolutionResult)
        assert result.resolved is True
        assert result.channel is not None
        assert result.channel.channel_id == "UC1234567890123456789012"
        assert result.channel.title == "Finance Test Channel"
        assert result.channel.handle == "@financetest"
        assert result.channel.uploads_playlist_id == "UU1234567890123456789012"
        assert result.channel.subscriber_count == 100000

    @pytest.mark.asyncio
    async def test_resolve_channel_by_handle_search(self, client: YouTubeClient):
        mock_search_response = {
            "items": [
                {
                    "id": {"channelId": "UC_search_found"},
                }
            ]
        }
        mock_channels_found = {
            "items": [
                {
                    "id": "UC_search_found",
                    "snippet": {
                        "title": "Found Creator",
                        "description": "Tech reviews",
                        "customUrl": "@foundcreator",
                    },
                    "contentDetails": {
                        "relatedPlaylists": {"uploads": "UU_search_found"}
                    },
                    "statistics": {
                        "subscriberCount": "50000",
                        "videoCount": "100",
                        "viewCount": "2000000",
                    },
                }
            ]
        }

        mock_service = MagicMock()
        # The search path goes: search().list().execute() → channels().list(id=...).execute()
        mock_service.channels().list().execute.return_value = mock_channels_found
        mock_service.search().list().execute.return_value = mock_search_response
        client._service = mock_service

        result = await client.resolve_channel("foundcreator")
        assert result.resolved is True
        assert result.channel is not None
        assert result.channel.channel_id == "UC_search_found"
        assert result.channel.title == "Found Creator"

    @pytest.mark.asyncio
    async def test_get_videos_batch(self, client: YouTubeClient):
        mock_response = {
            "items": [
                {
                    "id": "vid_001",
                    "snippet": {
                        "title": "Direct Indexing Masterclass",
                        "description": "Full guide to direct indexing.",
                        "publishedAt": "2026-01-15T12:00:00Z",
                        "channelId": "UC_creator",
                        "channelTitle": "Wealth Advisors",
                    },
                    "contentDetails": {
                        "duration": "PT25M30S",
                    },
                    "statistics": {
                        "viewCount": "45000",
                        "likeCount": "1200",
                        "commentCount": "85",
                    },
                    "status": {
                        "privacyStatus": "public",
                        "uploadStatus": "processed",
                    },
                }
            ]
        }

        mock_service = MagicMock()
        mock_service.videos().list().execute.return_value = mock_response
        client._service = mock_service

        videos = await client.get_videos_batch(["vid_001"])
        assert len(videos) == 1
        video = videos[0]
        assert isinstance(video, VideoDetailedMetadata)
        assert video.video_id == "vid_001"
        assert video.title == "Direct Indexing Masterclass"
        assert video.duration_seconds == 1530
        assert video.duration_formatted == "00:25:30"
        assert video.is_available is True
        assert video.view_count == 45000

    @pytest.mark.asyncio
    async def test_list_playlist_videos(self, client: YouTubeClient):
        mock_playlist_items = {
            "items": [
                {
                    "snippet": {
                        "title": "Episode 1",
                        "description": "Intro to investing",
                        "publishedAt": "2026-01-01T00:00:00Z",
                    },
                    "contentDetails": {
                        "videoId": "vid_ep1",
                    },
                }
            ],
            "nextPageToken": None,
            "prevPageToken": None,
            "pageInfo": {"totalResults": 1},
        }

        mock_video_cards = {
            "vid_ep1": VideoCard(
                video_id="vid_ep1",
                title="Episode 1",
                duration_seconds=600,
                duration_formatted="00:10:00",
                view_count=1000,
                published_at="2026-01-01T00:00:00Z",
                channel_id="UC_test_ch",
            )
        }

        mock_service = MagicMock()
        mock_service.playlistItems().list().execute.return_value = mock_playlist_items
        client._service = mock_service

        with patch.object(client, "_get_video_cards_map", new_callable=AsyncMock, return_value=mock_video_cards):
            page = await client.list_playlist_videos("PL_test_playlist", page_size=10)
            assert isinstance(page, PlaylistVideosPage)
            assert page.playlist_id == "PL_test_playlist"
            assert len(page.videos) == 1
            assert page.videos[0].video_id == "vid_ep1"
            assert page.videos[0].title == "Episode 1"
            assert page.videos[0].duration_seconds == 600
