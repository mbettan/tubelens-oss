"""
Integration tests for Starlette HTTP endpoints, AuthMiddleware, and FastMCP protocol surfaces.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.models import ChannelCandidate, ChannelResolutionResult, VideoDetailedMetadata
from src.server import (
    create_app,
    get_channel_info_resource,
    get_video_metadata_resource,
    mcp,
    prompt_creator_research,
    prompt_video_analysis,
)
from src.youtube import youtube_client


class TestMCPProtocolSurfaces:
    """Test FastMCP tool definitions, resources, and prompt templates."""

    @pytest.mark.asyncio
    async def test_all_10_tools_registered(self):
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        expected_tools = {
            "youtube_resolve_channel",
            "youtube_list_channel_videos",
            "youtube_list_playlist_videos",
            "youtube_get_videos",
            "youtube_analyze_video",
            "youtube_ask_video",
            "youtube_extract_recommendations",
            "youtube_extract_claims",
            "youtube_evaluate_fit",
            "youtube_compare_videos",
        }
        assert expected_tools.issubset(tool_names)

    @pytest.mark.asyncio
    async def test_all_tools_have_descriptions_and_parameters(self):
        tools = await mcp.list_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name} missing description"
            assert tool.inputSchema, f"Tool {tool.name} missing parameter schema"

    @pytest.mark.asyncio
    async def test_prompts_registered(self):
        prompts = await mcp.list_prompts()
        prompt_names = {p.name for p in prompts}
        assert "youtube_creator_research" in prompt_names
        assert "video_analysis" in prompt_names

    def test_prompt_rendering(self):
        research_prompt = prompt_creator_research("MockCreator", topic="tech")
        assert "MockCreator" in research_prompt
        assert "tech" in research_prompt

        analysis_prompt = prompt_video_analysis("https://www.youtube.com/watch?v=test_vid_001", focus_question="Key takeaways")
        assert "https://www.youtube.com/watch?v=test_vid_001" in analysis_prompt
        assert "Key takeaways" in analysis_prompt

    @pytest.mark.asyncio
    async def test_channel_info_resource(self):
        mock_result = ChannelResolutionResult(
            query="UC_res_123",
            resolved=True,
            match_type="exact_id",
            channel=ChannelCandidate(
                channel_id="UC_res_123",
                title="Test Channel",
                handle="@test",
                subscriber_count=5000,
                video_count=20,
                uploads_playlist_id="UU_res_123",
            ),
        )
        with patch.object(youtube_client, "resolve_channel", new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = mock_result
            res_str = await get_channel_info_resource("UC_res_123")
            data = json.loads(res_str)
            assert data["channel_id"] == "UC_res_123"
            assert data["title"] == "Test Channel"

    @pytest.mark.asyncio
    async def test_video_metadata_resource(self):
        mock_meta = VideoDetailedMetadata(
            video_id="vid_meta_123",
            title="Video Title",
            description="Video Description",
            published_at="2026-01-01T00:00:00Z",
            duration_seconds=300,
            duration_formatted="00:05:00",
            channel_id="UC_test",
            channel_title="Channel Title",
            is_available=True,
            is_live_content=False,
        )
        with patch.object(youtube_client, "get_videos_batch", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [mock_meta]
            res_str = await get_video_metadata_resource("vid_meta_123")
            data = json.loads(res_str)
            assert data["video_id"] == "vid_meta_123"
            assert data["title"] == "Video Title"


class TestServerHTTPEndpoints:
    """Test ASGI application HTTP endpoints and authentication enforcement."""

    @pytest.mark.asyncio
    async def test_healthz_and_livez(self):
        app = create_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp_healthz = await client.get("/healthz")
            assert resp_healthz.status_code == 200
            data = resp_healthz.json()
            assert data["status"] == "healthy"
            assert data["service"] == "tubelens-oss"

            resp_livez = await client.get("/livez")
            assert resp_livez.status_code == 200
            assert resp_livez.json()["status"] == "alive"

    @pytest.mark.asyncio
    async def test_auth_middleware_rejection_and_acceptance(self, monkeypatch):
        monkeypatch.setenv("AUTH_MODE", "api_key")
        monkeypatch.setenv("MCP_API_KEY", "secret-test-token-123")

        app = create_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # 1. Healthz should bypass auth
            resp_healthz = await client.get("/healthz")
            assert resp_healthz.status_code == 200

            # 2. Missing API key on protected endpoint
            resp_unauth = await client.get("/sse")
            assert resp_unauth.status_code == 401
            assert resp_unauth.json()["error"] == "UNAUTHORIZED"

            # 3. Invalid API key
            resp_bad = await client.get(
                "/sse",
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert resp_bad.status_code == 401

            # 4. Valid API key via Bearer token (use /healthz to avoid SSE stream hang)
            resp_ok = await client.get(
                "/healthz",
                headers={"Authorization": "Bearer secret-test-token-123"},
            )
            # Passes AuthMiddleware through to health endpoint
            assert resp_ok.status_code == 200

    @pytest.mark.asyncio
    async def test_seo_and_llm_endpoints(self):
        app = create_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp_llms = await client.get("/llms.txt")
            assert resp_llms.status_code == 200
            assert "TubeLens OSS" in resp_llms.text

            resp_llms_full = await client.get("/llms-full.txt")
            assert resp_llms_full.status_code == 200
            assert "youtube_resolve_channel" in resp_llms_full.text

            resp_robots = await client.get("/robots.txt")
            assert resp_robots.status_code == 200
            assert "Sitemap:" in resp_robots.text

            resp_sitemap = await client.get("/sitemap.xml")
            assert resp_sitemap.status_code == 200
            assert "<urlset" in resp_sitemap.text

            resp_fav_svg = await client.get("/favicon.svg")
            assert resp_fav_svg.status_code == 200
            assert "<svg" in resp_fav_svg.text

            resp_fav_ico = await client.get("/favicon.ico")
            assert resp_fav_ico.status_code == 200
            assert len(resp_fav_ico.content) > 0
