"""
YouTube Data API v3 Async Client Wrapper
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import datetime

from googleapiclient.discovery import Resource, build

from src.config import settings
from src.models import (
    CatalogVideosPage,
    ChannelCandidate,
    ChannelResolutionResult,
    PlaylistVideosPage,
    VideoCard,
    VideoDetailedMetadata,
)

logger = logging.getLogger("youtube_client")


def extract_video_id(url_or_id: str) -> str | None:
    """Extract standard 11-character YouTube video ID from various URL formats or raw ID."""
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()

    # watch?v=...
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)

    # youtu.be/...
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)

    # embed/...
    match = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)

    # shorts/...
    match = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)

    # live/...
    match = re.search(r"youtube\.com/live/([A-Za-z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)

    # v/...
    match = re.search(r"youtube\.com/v/([A-Za-z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)

    # Raw 11-char ID (must not contain URL slashes/dots/colons)
    if "/" not in url_or_id and "." not in url_or_id and ":" not in url_or_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    return None


def parse_iso8601_duration(duration_str: str) -> int:
    """Parse ISO 8601 duration (e.g. PT1H2M3S, PT30M, PT45S, P1W) into total seconds."""
    if not duration_str:
        return 0
    # Handle week format P1W
    week_match = re.match(r"^P(?P<weeks>\d+)W$", duration_str)
    if week_match:
        return int(week_match.group("weeks")) * 7 * 86400

    match = re.match(
        r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$",
        duration_str,
    )
    if not match:
        return 0
    parts = match.groupdict()
    days = int(parts.get("days") or 0)
    hours = int(parts.get("hours") or 0)
    minutes = int(parts.get("minutes") or 0)
    seconds = int(parts.get("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_seconds_to_hhmmss(seconds: int) -> str:
    """Format seconds into HH:MM:SS string."""
    seconds = max(seconds, 0)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_iso_datetime(date_str: str) -> datetime | None:
    """Parse ISO 8601 string or YYYY-MM-DD into a timezone-aware datetime."""
    if not date_str:
        return None
    try:
        clean = date_str.replace("Z", "+00:00")
        if len(clean) == 10:
            clean += "T00:00:00+00:00"
        return datetime.fromisoformat(clean)
    except Exception:
        return None


def derive_uploads_playlist_id(channel_id: str) -> str:
    """Derive standard uploads playlist ID (UU...) from channel ID (UC...)."""
    if channel_id.startswith("UC"):
        return "UU" + channel_id[2:]
    return "UU" + channel_id


class YouTubeClient:
    """Async wrapper for YouTube Data API v3 operations."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.youtube_api_key
        self._service: Resource | None = None
        self._lock = threading.Lock()

    def _get_service(self) -> Resource:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    # Build service with API key if provided, or default credentials
                    if self.api_key:
                        self._service = build(
                            "youtube", "v3", developerKey=self.api_key, cache_discovery=False
                        )
                    else:
                        self._service = build("youtube", "v3", cache_discovery=False)
        return self._service

    # --------------------------------------------------------------------------
    # Channel Resolution
    # --------------------------------------------------------------------------

    async def resolve_channel(self, query: str, max_candidates: int = 5) -> ChannelResolutionResult:
        """
        Resolves a creator handle, channel ID, channel URL, or creator name into canonical channels.
        """
        query = query.strip()

        # Handle URL formats
        if "youtube.com/@" in query:
            handle_match = re.search(r"youtube\.com/@([A-Za-z0-9_.-]+)", query)
            if handle_match:
                query = "@" + handle_match.group(1)
        elif "youtube.com/channel/" in query:
            cid_match = re.search(r"youtube\.com/channel/([A-Za-z0-9_-]+)", query)
            if cid_match:
                query = cid_match.group(1)

        # 1. Check if query is exact channel ID (starts with UC and is 24 chars)
        if query.startswith("UC") and len(query) == 24:
            candidate = await self._fetch_channel_by_id(query)
            if candidate:
                return ChannelResolutionResult(
                    query=query,
                    resolved=True,
                    match_type="exact_id",
                    channel=candidate,
                    candidates=[candidate],
                )

        # 2. Check if query is handle (e.g. @lexfridman)
        if query.startswith("@"):
            handle_name = query[1:]
            candidate = await self._fetch_channel_by_handle(handle_name)
            if candidate:
                return ChannelResolutionResult(
                    query=query,
                    resolved=True,
                    match_type="exact_handle",
                    channel=candidate,
                    candidates=[candidate],
                )

        # 3. Fall back to search.list with free-text query
        candidates = await self._search_channels_by_text(query, max_results=max_candidates)
        if not candidates:
            return ChannelResolutionResult(
                query=query,
                resolved=False,
                match_type="none",
                channel=None,
                candidates=[],
            )

        if len(candidates) == 1:
            return ChannelResolutionResult(
                query=query,
                resolved=True,
                match_type="search_match",
                channel=candidates[0],
                candidates=candidates,
            )

        # Multiple candidates found (ambiguous)
        return ChannelResolutionResult(
            query=query,
            resolved=False,
            match_type="search_match",
            channel=None,
            candidates=candidates,
        )

    async def _fetch_channel_by_id(self, channel_id: str) -> ChannelCandidate | None:
        def _call() -> dict | None:
            svc = self._get_service()
            res = (
                svc.channels()
                .list(
                    part="snippet,statistics,contentDetails",
                    id=channel_id,
                )
                .execute()
            )
            items = res.get("items", [])
            return items[0] if items else None

        try:
            item = await asyncio.to_thread(_call)
            return self._parse_channel_item(item) if item else None
        except Exception as e:
            logger.warning(f"Error fetching channel by id {channel_id}: {e}")
            return None

    async def _fetch_channel_by_handle(self, handle: str) -> ChannelCandidate | None:
        def _call() -> dict | None:
            svc = self._get_service()
            # Try with @ prefix first
            h = handle if handle.startswith("@") else f"@{handle}"
            try:
                res = (
                    svc.channels()
                    .list(
                        part="snippet,statistics,contentDetails",
                        forHandle=h,
                    )
                    .execute()
                )
                items = res.get("items", [])
                if items:
                    return items[0]
            except Exception as e:
                logger.debug(f"forHandle={h} attempt failed: {e}")

            # Try without @ prefix
            try:
                h_raw = handle.lstrip("@")
                res = (
                    svc.channels()
                    .list(
                        part="snippet,statistics,contentDetails",
                        forHandle=h_raw,
                    )
                    .execute()
                )
                items = res.get("items", [])
                return items[0] if items else None
            except Exception as e:
                logger.warning(f"Error fetching channel by handle {handle}: {e}")
                return None

        try:
            item = await asyncio.to_thread(_call)
            return self._parse_channel_item(item) if item else None
        except Exception as e:
            logger.warning(f"Error fetching channel by handle {handle}: {e}")
            return None


    async def _search_channels_by_text(
        self, query: str, max_results: int = 5
    ) -> list[ChannelCandidate]:
        def _call() -> list[str]:
            svc = self._get_service()
            res = (
                svc.search()
                .list(
                    part="snippet",
                    type="channel",
                    q=query,
                    maxResults=max_results,
                )
                .execute()
            )
            return [
                item["id"]["channelId"]
                for item in res.get("items", [])
                if item.get("id", {}).get("channelId")
            ]

        try:
            channel_ids = await asyncio.to_thread(_call)
            if not channel_ids:
                return []
            return await self._fetch_channels_batch(channel_ids)
        except Exception as e:
            logger.warning(f"Error searching channels for query '{query}': {e}")
            return []

    async def _fetch_channels_batch(self, channel_ids: list[str]) -> list[ChannelCandidate]:
        def _call() -> list[dict]:
            svc = self._get_service()
            res = (
                svc.channels()
                .list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(channel_ids),
                )
                .execute()
            )
            return res.get("items", [])

        try:
            items = await asyncio.to_thread(_call)
            return [self._parse_channel_item(item) for item in items if item]
        except Exception as e:
            logger.warning(f"Error fetching batch channels: {e}")
            return []

    def _parse_channel_item(self, item: dict) -> ChannelCandidate:
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        content_details = item.get("contentDetails", {})
        related = content_details.get("relatedPlaylists", {})
        channel_id = item.get("id", "")

        uploads_id = related.get("uploads") or derive_uploads_playlist_id(channel_id)
        thumbs = snippet.get("thumbnails", {})
        thumb_url = (
            thumbs.get("high", {}).get("url")
            or thumbs.get("medium", {}).get("url")
            or thumbs.get("default", {}).get("url")
        )

        sub_count = statistics.get("subscriberCount")
        vid_count = statistics.get("videoCount")

        return ChannelCandidate(
            channel_id=channel_id,
            title=snippet.get("title", ""),
            handle=snippet.get("customUrl"),
            description=snippet.get("description", ""),
            thumbnail_url=thumb_url,
            subscriber_count=int(sub_count) if sub_count is not None else None,
            video_count=int(vid_count) if vid_count is not None else None,
            uploads_playlist_id=uploads_id,
        )

    # --------------------------------------------------------------------------
    # Catalog Video Listing
    # --------------------------------------------------------------------------

    async def list_channel_videos(
        self,
        channel_id: str,
        cursor: str | None = None,
        page_size: int = 25,
        published_after: str | None = None,
        published_before: str | None = None,
        text_query: str | None = None,
        min_duration_s: int | None = None,
        max_duration_s: int | None = None,
        sort: str = "newest",
    ) -> CatalogVideosPage:
        """
        Enumerates videos from a channel's uploads playlist (`UU...`).
        Supports filtering and cursor pagination.
        """
        page_size = min(max(1, page_size), 100)
        uploads_playlist_id = derive_uploads_playlist_id(channel_id)

        # For sorting by popular or oldest, or complex filtering, we fetch catalog batch
        if (
            sort in ("popular", "oldest")
            or min_duration_s is not None
            or max_duration_s is not None
        ):
            return await self._list_channel_videos_sorted(
                channel_id=channel_id,
                uploads_playlist_id=uploads_playlist_id,
                cursor=cursor,
                page_size=page_size,
                published_after=published_after,
                published_before=published_before,
                text_query=text_query,
                min_duration_s=min_duration_s,
                max_duration_s=max_duration_s,
                sort=sort,
            )

        # Direct pagination for standard newest sort
        return await self._list_playlist_items(
            playlist_id=uploads_playlist_id,
            channel_id=channel_id,
            cursor=cursor,
            page_size=page_size,
            published_after=published_after,
            published_before=published_before,
            text_query=text_query,
        )

    async def _list_playlist_items(
        self,
        playlist_id: str,
        channel_id: str,
        cursor: str | None = None,
        page_size: int = 25,
        published_after: str | None = None,
        published_before: str | None = None,
        text_query: str | None = None,
    ) -> CatalogVideosPage:
        def _call() -> dict:
            svc = self._get_service()
            req = svc.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=page_size,
                pageToken=cursor if cursor else None,
            )
            return req.execute()

        try:
            res = await asyncio.to_thread(_call)
            items = res.get("items", [])
            next_page_token = res.get("nextPageToken")
            prev_page_token = res.get("prevPageToken")
            page_info = res.get("pageInfo", {})
            total_results = page_info.get("totalResults", len(items))

            video_ids = [
                item.get("contentDetails", {}).get("videoId")
                for item in items
                if item.get("contentDetails", {}).get("videoId")
            ]

            enriched_map = await self._get_video_cards_map(video_ids)

            cards: list[VideoCard] = []
            for item in items:
                vid = item.get("contentDetails", {}).get("videoId")
                if not vid or vid not in enriched_map:
                    continue
                card = enriched_map[vid]

                # Apply filters
                if published_after:
                    target_after = _parse_iso_datetime(published_after)
                    card_dt = _parse_iso_datetime(card.published_at)
                    if target_after and card_dt:
                        if card_dt < target_after:
                            continue
                    elif card.published_at < published_after:
                        continue

                if published_before:
                    target_before = _parse_iso_datetime(published_before)
                    card_dt = _parse_iso_datetime(card.published_at)
                    if target_before and card_dt:
                        if card_dt > target_before:
                            continue
                    elif card.published_at > published_before:
                        continue

                if text_query and (
                    text_query.lower() not in card.title.lower()
                    and text_query.lower() not in card.description.lower()
                ):
                    continue

                cards.append(card)

            return CatalogVideosPage(
                channel_id=channel_id,
                videos=cards,
                next_cursor=next_page_token,
                prev_cursor=prev_page_token,
                total_results=total_results,
            )
        except Exception as e:
            logger.error(f"Error listing playlist items for playlist {playlist_id}: {e}")
            return CatalogVideosPage(
                channel_id=channel_id,
                videos=[],
                total_results=0,
            )

    async def _list_channel_videos_sorted(
        self,
        channel_id: str,
        uploads_playlist_id: str,
        cursor: str | None,
        page_size: int,
        published_after: str | None,
        published_before: str | None,
        text_query: str | None,
        min_duration_s: int | None,
        max_duration_s: int | None,
        sort: str,
    ) -> CatalogVideosPage:
        # Decode offset cursor if present (format: offset_int)
        offset = 0
        if cursor:
            try:
                offset = int(cursor)
            except ValueError:
                offset = 0

        max_to_fetch = min(settings.max_catalog_videos, 500)
        all_video_ids: list[str] = []
        page_token: str | None = None

        def _fetch_page(token: str | None) -> dict:
            svc = self._get_service()
            return (
                svc.playlistItems()
                .list(
                    part="contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=token,
                )
                .execute()
            )

        while len(all_video_ids) < max_to_fetch:
            try:
                res = await asyncio.to_thread(_fetch_page, page_token)
                items = res.get("items", [])
                for item in items:
                    vid = item.get("contentDetails", {}).get("videoId")
                    if vid:
                        all_video_ids.append(vid)
                page_token = res.get("nextPageToken")
                if not page_token or not items:
                    break
            except Exception as e:
                logger.warning(f"Error fetching catalog chunk: {e}")
                break

        enriched_map = await self._get_video_cards_map(all_video_ids)
        filtered_cards: list[VideoCard] = []

        for vid in all_video_ids:
            if vid not in enriched_map:
                continue
            card = enriched_map[vid]

            if published_after:
                target_after = _parse_iso_datetime(published_after)
                card_dt = _parse_iso_datetime(card.published_at)
                if target_after and card_dt:
                    if card_dt < target_after:
                        continue
                elif card.published_at < published_after:
                    continue

            if published_before:
                target_before = _parse_iso_datetime(published_before)
                card_dt = _parse_iso_datetime(card.published_at)
                if target_before and card_dt:
                    if card_dt > target_before:
                        continue
                elif card.published_at > published_before:
                    continue

            if min_duration_s is not None and card.duration_seconds < min_duration_s:
                continue
            if max_duration_s is not None and card.duration_seconds > max_duration_s:
                continue
            if text_query and (
                text_query.lower() not in card.title.lower()
                and text_query.lower() not in card.description.lower()
            ):
                continue

            filtered_cards.append(card)

        # Sort
        if sort == "popular":
            filtered_cards.sort(key=lambda x: x.view_count or 0, reverse=True)
        elif sort == "oldest":
            filtered_cards.sort(key=lambda x: x.published_at)
        else:
            filtered_cards.sort(key=lambda x: x.published_at, reverse=True)

        total_matching = len(filtered_cards)
        page_slice = filtered_cards[offset : offset + page_size]
        next_offset = offset + page_size if (offset + page_size) < total_matching else None
        prev_offset = max(0, offset - page_size) if offset > 0 else None

        return CatalogVideosPage(
            channel_id=channel_id,
            videos=page_slice,
            next_cursor=str(next_offset) if next_offset is not None else None,
            prev_cursor=str(prev_offset) if prev_offset is not None else None,
            total_results=total_matching,
        )

    # --------------------------------------------------------------------------
    # Playlist Enumeration
    # --------------------------------------------------------------------------

    async def list_playlist_videos(
        self,
        playlist_id: str,
        cursor: str | None = None,
        page_size: int = 25,
    ) -> PlaylistVideosPage:
        """Enumerates video items directly from any public playlist ID (PL...)."""
        page_size = min(max(1, page_size), 100)

        def _call() -> dict:
            svc = self._get_service()
            req = svc.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=page_size,
                pageToken=cursor if cursor else None,
            )
            return req.execute()

        try:
            res = await asyncio.to_thread(_call)
            items = res.get("items", [])
            next_page_token = res.get("nextPageToken")
            page_info = res.get("pageInfo", {})
            total_results = page_info.get("totalResults", len(items))

            video_ids = [
                item.get("contentDetails", {}).get("videoId")
                for item in items
                if item.get("contentDetails", {}).get("videoId")
            ]

            enriched_map = await self._get_video_cards_map(video_ids)
            cards = [enriched_map[vid] for vid in video_ids if vid in enriched_map]

            return PlaylistVideosPage(
                playlist_id=playlist_id,
                videos=cards,
                next_cursor=next_page_token,
                total_results=total_results,
            )
        except Exception as e:
            logger.error(f"Error listing playlist {playlist_id}: {e}")
            return PlaylistVideosPage(
                playlist_id=playlist_id,
                videos=[],
                total_results=0,
            )

    # --------------------------------------------------------------------------
    # Batch Video Metadata
    # --------------------------------------------------------------------------

    async def get_videos_batch(self, video_ids: list[str]) -> list[VideoDetailedMetadata]:
        """Fetches detailed metadata for up to 50 video IDs."""
        if not video_ids:
            return []

        # Deduplicate and cap at 50 per batch
        unique_ids = list(dict.fromkeys(video_ids))[:50]

        def _call() -> list[dict]:
            svc = self._get_service()
            req = svc.videos().list(
                part="snippet,contentDetails,statistics,status",
                id=",".join(unique_ids),
            )
            return req.execute().get("items", [])

        try:
            items = await asyncio.to_thread(_call)
            results: list[VideoDetailedMetadata] = []
            for item in items:
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                statistics = item.get("statistics", {})
                status = item.get("status", {})

                duration_s = parse_iso8601_duration(content_details.get("duration", ""))
                view_cnt = statistics.get("viewCount")
                like_cnt = statistics.get("likeCount")
                comm_cnt = statistics.get("commentCount")
                live_broadcast = snippet.get("liveBroadcastContent", "none")

                results.append(
                    VideoDetailedMetadata(
                        video_id=item.get("id", ""),
                        title=snippet.get("title", ""),
                        description=snippet.get("description", ""),
                        published_at=snippet.get("publishedAt", ""),
                        duration_seconds=duration_s,
                        duration_formatted=format_seconds_to_hhmmss(duration_s),
                        is_short=(0 < duration_s <= 180),
                        channel_id=snippet.get("channelId", ""),
                        channel_title=snippet.get("channelTitle", ""),
                        view_count=int(view_cnt) if view_cnt is not None else None,
                        like_count=int(like_cnt) if like_cnt is not None else None,
                        comment_count=int(comm_cnt) if comm_cnt is not None else None,
                        tags=snippet.get("tags", []),
                        is_live_content=live_broadcast in ("live", "upcoming"),
                        is_available=(
                            status.get("privacyStatus", "public") == "public"
                            and bool(status.get("embeddable", True))
                        ),
                        privacy_status=status.get("privacyStatus", "public"),
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error fetching batch video metadata: {e}")
            return []

    async def _get_video_cards_map(self, video_ids: list[str]) -> dict[str, VideoCard]:
        if not video_ids:
            return {}

        details = await self.get_videos_batch(video_ids)
        result: dict[str, VideoCard] = {}
        for d in details:
            result[d.video_id] = VideoCard(
                video_id=d.video_id,
                title=d.title,
                description=d.description[:200] + "..."
                if len(d.description) > 200
                else d.description,
                published_at=d.published_at,
                duration_seconds=d.duration_seconds,
                duration_formatted=d.duration_formatted,
                is_short=d.is_short,
                thumbnail_url=f"https://i.ytimg.com/vi/{d.video_id}/hqdefault.jpg",
                view_count=d.view_count,
                channel_id=d.channel_id,
                channel_title=d.channel_title,
            )
        return result


# Global singleton instance
youtube_client = YouTubeClient()
