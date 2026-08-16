"""
FastMCP Server Entrypoint for TubeLens OSS
Exposes 10 Video Intelligence & Research Tools, 2 Resources, and 2 Prompts over Streamable HTTP (/mcp) + Legacy SSE.

Intelligence Suite (100% Fair Use / In-Cloud Ingestion):
  - youtube_ask_video: Ephemeral Deep Video Q&A with timestamped citations
  - youtube_analyze_video: Structured summary, key topics, notable quotes, topic index
  - youtube_extract_recommendations: Decisions, tickers, tools, stance, conviction, risks, alternatives
  - youtube_extract_claims: Verifiable factual claims vs opinion, verification guidance
  - youtube_evaluate_fit: Personalized relevance scoring and custom action items
  - youtube_compare_videos: Cross-video consensus, disagreement matrix, and root cause analysis
  - youtube_resolve_channel: Channel/creator profile resolution
  - youtube_list_channel_videos: Catalog pagination, date/duration/search filtering
  - youtube_list_playlist_videos: Public playlist enumeration
  - youtube_get_videos: Batch detailed metadata fetching
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src import __version__
from src.config import settings
from src.errors import ErrorCode, ToolError
from src.models import (
    CatalogVideosPage,
    ChannelResolutionResult,
    PlaylistVideosPage,
    VideoDetailedMetadata,
)
from src.oauth import (
    handle_dynamic_client_registration,
    handle_oauth_authorize,
    handle_oauth_metadata,
    handle_oauth_token,
    handle_oauth_userinfo,
)
from src.transcribe import transcription_engine
from src.youtube import extract_video_id, youtube_client

# --------------------------------------------------------------------------
# Structured Logging with Redaction
# --------------------------------------------------------------------------

#: Field names that must never reach a log sink.
SENSITIVE_FIELDS: frozenset[str] = frozenset(
    {
        "text", "segments", "segment", "transcript", "transcript_text",
        "description", "description_snippet", "first_segments",
        "prompt", "content", "raw", "body",
        "api_key", "authorization", "youtube_api_key", "mcp_api_key", "signed_url",
    }
)

_LOG_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message", "asctime", "taskName",
}

MAX_LOG_RECORD_BYTES = 2048


class RedactingFilter(logging.Filter):
    """Strips sensitive keys from ``extra`` payloads before formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if key.lower() in SENSITIVE_FIELDS:
                record.__dict__[key] = "[redacted]"
        return True


class CloudLoggingFormatter(logging.Formatter):
    """One JSON object per line, in Cloud Logging's ``severity`` convention."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if len(msg) > MAX_LOG_RECORD_BYTES:
            msg = msg[:MAX_LOG_RECORD_BYTES - 20] + "... [truncated]"

        payload: dict[str, Any] = {
            "severity": record.levelname,
            "message": msg,
            "logger": record.name,
        }
        for key, value in record.__dict__.items():
            if key in _LOG_RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)[-512:]

        raw = json.dumps(payload, default=str, ensure_ascii=False)
        if len(raw) > MAX_LOG_RECORD_BYTES:
            raw = raw[:MAX_LOG_RECORD_BYTES - 20] + '..."}'
        return raw


def _configure_logging() -> None:
    """Cloud Run detection: JSON for production, human-friendly for local/pytest."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())

    if os.getenv("K_SERVICE"):
        handler.setFormatter(CloudLoggingFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

    root_logger.addHandler(handler)


_configure_logging()
logger = logging.getLogger("tubelens_mcp")


# ------------------------------------------------------------------------------
# FastMCP Server Initialization
# ------------------------------------------------------------------------------

mcp = FastMCP(
    name="tubelens",
    instructions=(
        "TubeLens provides structured video intelligence, decision extraction, and YouTube research tools. "
        "DECISION & RECOMMENDATION EXTRACTION: Use youtube_extract_recommendations to get structured "
        "portfolios/tools/actions with stance, conviction, risks, and alternatives. "
        "FACTUAL CLAIM EXTRACTION: Use youtube_extract_claims to isolate falsifiable claims with verification guidance. "
        "PERSONALIZED FIT EVALUATION: Use youtube_evaluate_fit to score relevance (0-100) and generate tailored action items. "
        "CROSS-VIDEO COMPARISON: Use youtube_compare_videos to synthesize consensus and disagreements across 2-5 videos. "
        "EPHEMERAL DEEP VIDEO Q&A: Use youtube_ask_video to ask deep questions with timestamped citations. "
        "PUBLIC VIDEO ANALYSIS: Use youtube_analyze_video for structured summaries, key topics, and topic indexes. "
        "METADATA & CATALOGS: Use youtube_resolve_channel, youtube_list_channel_videos, and youtube_get_videos. "
        "Always attribute content to the original creator with an inline clickable timestamp link: [Title @ HH:MM:SS](URL&t=SECONDS)."
    ),
    host=settings.host,
    port=settings.port,
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# ------------------------------------------------------------------------------
# MCP Tools (10 Intelligence & Research Tools)
# ------------------------------------------------------------------------------


@mcp.tool(
    name="youtube_resolve_channel",
    description=(
        "Resolves a YouTube handle, channel URL, channel ID ('UC...'), or creator name to a canonical YouTube channel. "
        "Use before listing a creator's videos when the channel ID is unknown. "
        "Do not use for searching individual videos by topic or keyword."
    ),
)
async def youtube_resolve_channel(
    query: str,
    max_candidates: int = 5,
) -> ChannelResolutionResult:
    """Resolve creator or channel query to canonical channel candidates."""
    logger.info(f"Resolving channel with query: '{query}'")
    return await youtube_client.resolve_channel(query=query, max_candidates=max_candidates)


@mcp.tool(
    name="youtube_list_channel_videos",
    description=(
        "Lists and filters videos from a creator's uploads catalog ('UU...'). "
        "Use to browse recent uploads, find Shorts (max_duration_s=180), or filter videos by topic. "
        "Requires a canonical channel ID ('UC...')."
    ),
)
async def youtube_list_channel_videos(
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
    """List videos in a channel's upload catalog with filtering and pagination."""
    logger.info(f"Listing channel videos for {channel_id} (sort={sort}, cursor={cursor})")
    return await youtube_client.list_channel_videos(
        channel_id=channel_id,
        cursor=cursor,
        page_size=page_size,
        published_after=published_after,
        published_before=published_before,
        text_query=text_query,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        sort=sort,
    )


@mcp.tool(
    name="youtube_list_playlist_videos",
    description=(
        "Enumerates video items directly from any public YouTube playlist ID ('PL...'). "
        "Use when the user provides a specific playlist ID or link. Do not use for channel uploads."
    ),
)
async def youtube_list_playlist_videos(
    playlist_id: str,
    cursor: str | None = None,
    page_size: int = 25,
) -> PlaylistVideosPage:
    """List video items from a public playlist."""
    logger.info(f"Listing playlist videos for {playlist_id} (cursor={cursor})")
    return await youtube_client.list_playlist_videos(
        playlist_id=playlist_id,
        cursor=cursor,
        page_size=page_size,
    )


@mcp.tool(
    name="youtube_get_videos",
    description=(
        "Fetches batch detailed metadata for up to 50 YouTube video IDs in a single call. "
        "Use to retrieve view counts, durations, and descriptions for known video IDs."
    ),
)
async def youtube_get_videos(
    video_ids: list[str],
) -> list[VideoDetailedMetadata]:
    """Fetch batch detailed metadata for video IDs."""
    logger.info(f"Fetching batch metadata for {len(video_ids)} videos")
    return await youtube_client.get_videos_batch(video_ids=video_ids)


@mcp.tool(
    name="youtube_analyze_video",
    description=(
        "Analyzes a public YouTube video: generates an AI summary, key topics, "
        "short quoted excerpts with speaker attribution and timestamps, and a "
        "timestamped topic index. Does NOT produce a full verbatim transcript. "
        "Includes mandatory creator attribution in every response."
    ),
)
async def youtube_analyze_video(
    url: str,
    focus_question: str | None = None,
) -> Any:
    """Analyze a public video using Gemini — produces transformative, non-reproductive output."""
    video_id = extract_video_id(url)
    if not video_id:
        return ToolError(ErrorCode.INVALID_INPUT, f"Malformed YouTube URL: '{url}'").envelope()

    logger.info(f"Analyzing video {video_id} (focus={focus_question})")

    try:
        video_meta = await transcription_engine.preflight_check(video_id)
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        logger.warning(f"Preflight check failed for {video_id}: {e}")
        return ToolError(ErrorCode.VIDEO_UNAVAILABLE, str(e)).envelope()

    try:
        analysis = await transcription_engine.analyze_video(
            url=url,
            video_meta=video_meta,
            focus_question=focus_question,
        )
        return analysis
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        logger.error(f"Analysis error on {video_id}: {e}")
        return ToolError(ErrorCode.TRANSCRIPTION_FAILED, str(e)).envelope()


@mcp.tool(
    name="youtube_ask_video",
    description=(
        "Ask deep, specific questions about any public YouTube video. "
        "Processes the video multimodally and returns a direct, comprehensive, "
        "reasoned answer backed by exact speaker-attributed citations and clickable timestamp links. "
        "Does NOT expose or store the raw verbatim transcript (Fair Use compliant)."
    ),
)
async def youtube_ask_video(
    url: str,
    query: str,
    time_window_start_s: int | None = None,
    time_window_end_s: int | None = None,
) -> Any:
    """Answer deep questions about a video with evidence citations without storing transcripts."""
    video_id = extract_video_id(url)
    if not video_id:
        return ToolError(ErrorCode.INVALID_INPUT, f"Malformed YouTube URL: '{url}'").envelope()

    logger.info(f"Answering query on video {video_id}: '{query}'")

    try:
        video_meta = await transcription_engine.preflight_check(video_id)
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        logger.warning(f"Preflight check failed for {video_id}: {e}")
        return ToolError(ErrorCode.VIDEO_UNAVAILABLE, str(e)).envelope()

    try:
        answer = await transcription_engine.ask_video(
            url=url,
            query=query,
            video_meta=video_meta,
            time_window_start_s=time_window_start_s,
            time_window_end_s=time_window_end_s,
        )
        return answer
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        logger.error(f"Q&A error on {video_id}: {e}")
        return ToolError(ErrorCode.TRANSCRIPTION_FAILED, str(e)).envelope()


@mcp.tool(
    name="youtube_extract_recommendations",
    description=(
        "Extracts structured explicit and implicit recommendations, actions, and decisions from a video. "
        "Returns specific entities (ETFs, stocks, software, strategies), action type, stance, conviction, "
        "target audience, core thesis, risks/tradeoffs, alternatives compared, and timestamped evidence citations."
    ),
)
async def youtube_extract_recommendations(
    url: str,
    focus_category: str | None = None,
) -> Any:
    """Extract structured decisions and recommendations from a video."""
    video_id = extract_video_id(url)
    if not video_id:
        return ToolError(ErrorCode.INVALID_INPUT, f"Malformed YouTube URL: '{url}'").envelope()

    logger.info(f"Extracting recommendations from {video_id} (category={focus_category})")

    try:
        video_meta = await transcription_engine.preflight_check(video_id)
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        return ToolError(ErrorCode.VIDEO_UNAVAILABLE, str(e)).envelope()

    try:
        return await transcription_engine.extract_recommendations(
            url=url,
            video_meta=video_meta,
            focus_category=focus_category,
        )
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        logger.error(f"Recommendations error on {video_id}: {e}")
        return ToolError(ErrorCode.TRANSCRIPTION_FAILED, str(e)).envelope()


@mcp.tool(
    name="youtube_extract_claims",
    description=(
        "Deconstructs a video into discrete, falsifiable factual claims versus creator opinion. "
        "Returns claim statements, category (data/metric, historical, technical, market forecast, opinion), "
        "confidence level, independent verification guidance, and timestamped citations."
    ),
)
async def youtube_extract_claims(
    url: str,
    category_filter: str | None = None,
) -> Any:
    """Extract falsifiable factual claims with verification guidance."""
    video_id = extract_video_id(url)
    if not video_id:
        return ToolError(ErrorCode.INVALID_INPUT, f"Malformed YouTube URL: '{url}'").envelope()

    logger.info(f"Extracting claims from {video_id} (filter={category_filter})")

    try:
        video_meta = await transcription_engine.preflight_check(video_id)
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        return ToolError(ErrorCode.VIDEO_UNAVAILABLE, str(e)).envelope()

    try:
        return await transcription_engine.extract_claims(
            url=url,
            video_meta=video_meta,
            category_filter=category_filter,
        )
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        logger.error(f"Claims error on {video_id}: {e}")
        return ToolError(ErrorCode.TRANSCRIPTION_FAILED, str(e)).envelope()


@mcp.tool(
    name="youtube_evaluate_fit",
    description=(
        "Evaluates the personalized relevance and applicability of a video against a user profile and constraints. "
        "Calculates relevance score (0-100), verdict (must_watch, key_takeaways_only, partially_applicable, not_applicable), "
        "matching points, conflicting assumptions/caveats, and concrete custom action items."
    ),
)
async def youtube_evaluate_fit(
    url: str,
    user_profile: str,
    constraints: list[str] | None = None,
) -> Any:
    """Evaluate personalized fit and generate custom action items."""
    video_id = extract_video_id(url)
    if not video_id:
        return ToolError(ErrorCode.INVALID_INPUT, f"Malformed YouTube URL: '{url}'").envelope()

    logger.info(f"Evaluating fit for {video_id} against user profile")

    try:
        video_meta = await transcription_engine.preflight_check(video_id)
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        return ToolError(ErrorCode.VIDEO_UNAVAILABLE, str(e)).envelope()

    try:
        return await transcription_engine.evaluate_fit(
            url=url,
            video_meta=video_meta,
            user_profile=user_profile,
            constraints=constraints,
        )
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        logger.error(f"Fit evaluation error on {video_id}: {e}")
        return ToolError(ErrorCode.TRANSCRIPTION_FAILED, str(e)).envelope()


@mcp.tool(
    name="youtube_compare_videos",
    description=(
        "Performs cross-video synthesis on 2 to 5 videos discussing the same topic. "
        "Identifies points of consensus, breaks down disagreements into dimensions with creator positions, "
        "analyzes the root cause of divergence (taxes, scale, timeframe, risk), and generates a recommended decision playbook."
    ),
)
async def youtube_compare_videos(
    urls: list[str],
    topic: str,
) -> Any:
    """Compare 2-5 videos for consensus, disagreement matrix, and root causes."""
    logger.info(f"Comparing {len(urls)} videos on topic: '{topic}'")
    try:
        return await transcription_engine.compare_videos(urls=urls, topic=topic)
    except ToolError as te:
        return te.envelope()
    except Exception as e:
        logger.error(f"Cross-video comparison error: {e}")
        return ToolError(ErrorCode.TRANSCRIPTION_FAILED, str(e)).envelope()


# ------------------------------------------------------------------------------
# MCP Resources (3 Resources)
# ------------------------------------------------------------------------------


@mcp.resource("youtube://fair-use-policy")
async def get_fair_use_policy_resource() -> str:
    """Read-only statement of Fair Use and creator attribution principles."""
    return json.dumps(
        {
            "policy": "TubeLens OSS Fair Use & Attribution Policy",
            "statute": "17 U.S.C. Section 107",
            "principles": [
                "Transformative analysis, synthesis, Q&A, and fact verification",
                "Zero local media storage or downloads",
                "Mandatory creator attribution with original timestamps",
            ],
        },
        indent=2,
    )


@mcp.resource("youtube://channels/{channel_id}/info")
async def get_channel_info_resource(channel_id: str) -> str:
    """Read-only live channel resolution and info."""
    result = await youtube_client.resolve_channel(channel_id, max_candidates=1)
    if result and result.resolved and result.channel:
        return json.dumps(result.channel.model_dump(), indent=2)
    return json.dumps({"channel_id": channel_id, "status": "not_found"})


@mcp.resource("youtube://videos/{video_id}/metadata")
async def get_video_metadata_resource(video_id: str) -> str:
    """Read-only live video metadata."""
    metas = await youtube_client.get_videos_batch([video_id])
    if metas:
        return json.dumps(metas[0].model_dump(), indent=2)
    return json.dumps({"video_id": video_id, "status": "not_found"})


# ------------------------------------------------------------------------------
# MCP Prompts (2 Prompts)
# ------------------------------------------------------------------------------


@mcp.prompt("youtube_creator_research")
def prompt_creator_research(channel_query: str, topic: str = "") -> str:
    """Prompt template guiding cost-effective, metadata-first creator research."""
    topic_clause = f" matching the topic '{topic}'" if topic else ""
    return (
        f"You are an expert research assistant. Use TubeLens to investigate the creator '{channel_query}'.\n"
        f"Step 1: Resolve the creator channel using `youtube_resolve_channel`.\n"
        f"Step 2: List recent videos{topic_clause} using `youtube_list_channel_videos`.\n"
        f"Step 3: Extract structured decisions using `youtube_extract_recommendations` or analyze with `youtube_analyze_video`.\n"
        f"Step 4: Synthesize key insights citing exact speaker turns: [Video Title @ HH:MM:SS](URL&t=s)."
    )


@mcp.prompt("video_analysis")
def prompt_video_analysis(video_url: str, focus_question: str) -> str:
    """Prompt template guiding question-answering workflow for a specific video."""
    return (
        f"Analyze the video at '{video_url}' to answer: '{focus_question}'.\n"
        f"Step 1: Use `youtube_ask_video` to ask deep, specific questions and get direct evidence-backed answers, or `youtube_extract_claims` to audit facts.\n"
        f"Step 2: Review the direct answer and key evidence quotes.\n"
        f"Step 3: Formulate a concise, speaker-attributed answer citing exact timestamps: [Title @ HH:MM:SS](URL&t=s)."
    )


# ------------------------------------------------------------------------------
# Health Checks
# ------------------------------------------------------------------------------

async def healthz(request: Request) -> JSONResponse:
    """Liveness probe."""
    return JSONResponse(
        {
            "status": "healthy",
            "service": "tubelens-oss",
            "version": __version__,
            "model": settings.gemini_primary_model,
            "endpoint": settings.google_cloud_location,
        }
    )


async def livez(request: Request) -> JSONResponse:
    """Readiness probe."""
    return JSONResponse({"status": "alive"})


# ------------------------------------------------------------------------------
# Authentication Middleware
# ------------------------------------------------------------------------------

class AuthMiddleware:
    """Pure ASGI Authentication & Structured Logging Middleware."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        path = request.url.path
        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Redact sensitive parameters from query string before logging
        safe_params = dict(request.query_params)
        for sensitive_key in ("api_key", "token", "key", "secret", "authorization", "password"):
            if sensitive_key in safe_params:
                safe_params[sensitive_key] = "[REDACTED]"
        query_str = str(safe_params) if safe_params else ""

        logger.info(f"➡️ [HTTP IN] {method} {path} (client={client_ip}, ua={user_agent}, query={query_str})")

        # Allow health checks, docs, static assets, and OAuth discovery/authorization endpoints
        if (
            path in (
                "/healthz",
                "/livez",
                "/docs",
                "/docs/",
                "/llms.txt",
                "/llms-full.txt",
                "/robots.txt",
                "/sitemap.xml",
                "/favicon.ico",
                "/favicon.svg",
                "/.well-known/oauth-authorization-server",
                "/.well-known/openid-configuration",
                "/oauth/register",
                "/oauth/authorize",
                "/oauth/token",
            )
            or path.startswith(("/docs/", "/static/", "/.well-known/", "/oauth/"))
        ):
            await self.app(scope, receive, send)
            return

        # Check for OAuth 2.0 Bearer JWT token first
        auth_header = request.headers.get("authorization", "")
        bearer_token = ""
        if auth_header.lower().startswith("bearer "):
            bearer_token = auth_header[7:].strip()

        if bearer_token:
            from src.oauth import oauth_server
            jwt_payload = oauth_server.verify_token(bearer_token)
            if jwt_payload:
                logger.info(f"🔑 [AUTH OK] Authorized via OAuth 2.0 JWT for {method} {path} (sub={jwt_payload.get('sub')})")
                await self.app(scope, receive, send)
                return

        # API Key authentication mode check
        auth_mode = os.environ.get("AUTH_MODE", settings.auth_mode)
        expected_key = (os.environ.get("MCP_API_KEY") or settings.mcp_api_key or "").strip()

        if auth_mode in ("api_key", "oauth"):
            client_key = (
                request.headers.get("x-mcp-api-key", "")
                or request.query_params.get("api_key", "")
                or bearer_token
            ).strip()

            if client_key and expected_key and secrets.compare_digest(client_key, expected_key):
                logger.info(f"🔑 [AUTH OK] Authorized via MCP API Key for {method} {path}")
                await self.app(scope, receive, send)
                return

            logger.warning(f"🚫 [AUTH REJECTED] Missing/invalid credentials for {method} {path} from {client_ip}")
            error_response = JSONResponse(
                {
                    "error": "UNAUTHORIZED",
                    "message": "Missing or invalid OAuth token or API key.",
                    "recovery_hint": "Provide a valid 'Authorization: Bearer <TOKEN>' header or authenticate via OAuth 2.0 (/oauth/authorize).",
                },
                status_code=401,
            )
            await error_response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# ------------------------------------------------------------------------------
# Application Factory
# ------------------------------------------------------------------------------

def create_app() -> Starlette:
    """
    Creates the complete Starlette ASGI app combining:
    - Streamable HTTP FastMCP app on /mcp (and root /)
    - Legacy SSE app on /sse
    - Health checks /healthz and /livez
    - Interactive docs portal on /docs
    - API Key and IAM Authentication middleware
    """
    from mcp.server.sse import SseServerTransport

    mcp_http_app = mcp.streamable_http_app()
    sse_transport = SseServerTransport(
        "/messages/",
        security_settings=mcp.settings.transport_security,
    )

    async def handle_sse(request: Request) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        ua = request.headers.get("user-agent", "unknown")
        logger.info(f"⚡ [SSE STREAM OPEN] Client connected from {client_ip} (ua={ua})")
        try:
            async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
                await mcp._mcp_server.run(
                    streams[0],
                    streams[1],
                    mcp._mcp_server.create_initialization_options(),
                )
            logger.info(f"⚡ [SSE STREAM CLOSED] Client from {client_ip} closed session normally")
        except Exception as exc:
            logger.error(f"⚡ [SSE STREAM ERROR] Session error for {client_ip}: {exc}", exc_info=True)
        return Response()

    routes = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/livez", livez, methods=["GET"]),
        Mount("/sse/messages", app=sse_transport.handle_post_message),
        Mount("/messages", app=sse_transport.handle_post_message),
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        # OAuth 2.0 Authorization Server Endpoints (RFC 8414, RFC 7591, RFC 6749, RFC 7636)
        Route("/.well-known/oauth-authorization-server", endpoint=handle_oauth_metadata, methods=["GET"]),
        Route("/.well-known/openid-configuration", endpoint=handle_oauth_metadata, methods=["GET"]),
        Route("/oauth/register", endpoint=handle_dynamic_client_registration, methods=["POST"]),
        Route("/oauth/authorize", endpoint=handle_oauth_authorize, methods=["GET", "POST"]),
        Route("/oauth/token", endpoint=handle_oauth_token, methods=["POST"]),
        Route("/oauth/userinfo", endpoint=handle_oauth_userinfo, methods=["GET"]),
        Route("/oauth/me", endpoint=handle_oauth_userinfo, methods=["GET"]),
    ]

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    if docs_dir.exists():
        routes.append(Mount("/docs", app=StaticFiles(directory=str(docs_dir), html=True), name="docs"))

        async def serve_llms_txt(request: Request) -> Response:
            file_path = docs_dir / "llms.txt"
            if file_path.exists():
                return Response(content=file_path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")
            return Response("Not Found", status_code=404)

        async def serve_llms_full_txt(request: Request) -> Response:
            file_path = docs_dir / "llms-full.txt"
            if file_path.exists():
                return Response(content=file_path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")
            return Response("Not Found", status_code=404)

        async def serve_robots_txt(request: Request) -> Response:
            file_path = docs_dir / "robots.txt"
            if file_path.exists():
                return Response(content=file_path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")
            return Response("Not Found", status_code=404)

        async def serve_sitemap_xml(request: Request) -> Response:
            file_path = docs_dir / "sitemap.xml"
            if file_path.exists():
                return Response(content=file_path.read_text(encoding="utf-8"), media_type="application/xml; charset=utf-8")
            return Response("Not Found", status_code=404)

        async def serve_favicon_svg(request: Request) -> Response:
            file_path = docs_dir / "favicon.svg"
            if file_path.exists():
                return Response(content=file_path.read_text(encoding="utf-8"), media_type="image/svg+xml")
            return Response("Not Found", status_code=404)

        async def serve_favicon_ico(request: Request) -> Response:
            file_path = docs_dir / "favicon.ico"
            if file_path.exists():
                return Response(content=file_path.read_bytes(), media_type="image/x-icon")
            return Response("Not Found", status_code=404)

        routes.append(Route("/llms.txt", endpoint=serve_llms_txt, methods=["GET"]))
        routes.append(Route("/llms-full.txt", endpoint=serve_llms_full_txt, methods=["GET"]))
        routes.append(Route("/robots.txt", endpoint=serve_robots_txt, methods=["GET"]))
        routes.append(Route("/sitemap.xml", endpoint=serve_sitemap_xml, methods=["GET"]))
        routes.append(Route("/favicon.svg", endpoint=serve_favicon_svg, methods=["GET"]))
        routes.append(Route("/favicon.ico", endpoint=serve_favicon_ico, methods=["GET"]))
    else:
        async def redirect_docs(request: Request) -> Response:
            return RedirectResponse("https://mbettan.github.io/tubelens-oss/", status_code=301)

        routes.append(Route("/docs", endpoint=redirect_docs, methods=["GET"]))
        routes.append(Route("/docs/", endpoint=redirect_docs, methods=["GET"]))
        routes.append(Route("/docs/{rest:path}", endpoint=redirect_docs, methods=["GET"]))

    # Mount FastMCP Streamable HTTP app
    routes.append(Mount("/mcp", app=mcp_http_app))
    routes.append(Mount("/", app=mcp_http_app))

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        Middleware(AuthMiddleware),
    ]

    app = Starlette(
        debug=False,
        routes=routes,
        middleware=middleware,
        lifespan=mcp_http_app.router.lifespan_context,
    )
    return app


app = create_app()


def run_server() -> None:
    """Run production server using uvicorn."""
    logger.info(
        f"Starting TubeLens OSS v{__version__} on {settings.host}:{settings.port} "
        f"(auth_mode={settings.auth_mode}, transport=streamable-http /mcp)"
    )
    uvicorn.run(
        "src.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()

