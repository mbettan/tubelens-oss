"""
Pydantic Domain Models and Data Contracts for TubeLens OSS
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# Channel Models
# ------------------------------------------------------------------------------


class ChannelCandidate(BaseModel):
    """Normalized YouTube creator channel profile."""

    channel_id: str = Field(description="Canonical YouTube Channel ID (UC...)")
    title: str = Field(description="Channel title or creator name")
    handle: str | None = Field(default=None, description="YouTube @handle if available")
    description: str = Field(default="", description="Channel description / bio snippet")
    thumbnail_url: str | None = Field(default=None, description="Channel avatar thumbnail URL")
    subscriber_count: int | None = Field(default=None, description="Total subscriber count")
    video_count: int | None = Field(default=None, description="Total published video count")
    uploads_playlist_id: str = Field(description="Uploads playlist ID (UU...)")


class ChannelResolutionResult(BaseModel):
    """Result of channel resolution query."""

    query: str = Field(description="Original user search query or handle")
    resolved: bool = Field(description="True if an unambiguous channel was found")
    match_type: Literal["exact_id", "exact_handle", "search_match", "none"] = Field(
        description="Type of match resolved"
    )
    channel: ChannelCandidate | None = Field(
        default=None, description="The canonical resolved channel"
    )
    candidates: list[ChannelCandidate] = Field(
        default_factory=list,
        description="Alternative candidate matches if resolution is ambiguous",
    )


# ------------------------------------------------------------------------------
# Video Metadata Models
# ------------------------------------------------------------------------------


class VideoCard(BaseModel):
    """Summary metadata card for a video in a catalog or playlist."""

    video_id: str = Field(description="11-character YouTube video ID")
    title: str = Field(description="Video title")
    description: str = Field(default="", description="Truncated video description snippet")
    published_at: str = Field(description="ISO 8601 publication timestamp")
    duration_seconds: int = Field(default=0, description="Duration in seconds")
    duration_formatted: str = Field(default="00:00:00", description="Formatted duration (HH:MM:SS)")
    is_short: bool = Field(default=False, description="True if video is a YouTube Short (<=180 seconds)")
    thumbnail_url: str | None = Field(default=None, description="High-resolution thumbnail URL")
    view_count: int | None = Field(default=None, description="View count")
    channel_id: str = Field(description="Channel ID")
    channel_title: str = Field(default="", description="Channel title")


class CatalogVideosPage(BaseModel):
    """Paginated catalog of videos from a channel uploads playlist."""

    channel_id: str = Field(description="Channel ID")
    videos: list[VideoCard] = Field(
        default_factory=list, description="List of video metadata cards"
    )
    next_cursor: str | None = Field(default=None, description="Cursor for the next page")
    prev_cursor: str | None = Field(default=None, description="Cursor for the previous page")
    total_results: int = Field(default=0, description="Total matching items available")


class PlaylistVideosPage(BaseModel):
    """Paginated list of videos in a public YouTube playlist."""

    playlist_id: str = Field(description="Playlist ID (PL...)")
    videos: list[VideoCard] = Field(
        default_factory=list, description="List of video metadata cards"
    )
    next_cursor: str | None = Field(default=None, description="Cursor for the next page")
    total_results: int = Field(default=0, description="Total items in playlist")


class VideoDetailedMetadata(BaseModel):
    """Detailed metadata for a specific YouTube video."""

    video_id: str = Field(description="YouTube video ID")
    title: str = Field(description="Video title")
    description: str = Field(default="", description="Full video description")
    published_at: str = Field(description="ISO 8601 publication timestamp")
    duration_seconds: int = Field(description="Duration in seconds")
    duration_formatted: str = Field(description="Formatted duration (HH:MM:SS)")
    is_short: bool = Field(default=False, description="True if video is a YouTube Short (<=180 seconds)")
    channel_id: str = Field(description="Channel ID")
    channel_title: str = Field(description="Channel title")
    view_count: int | None = Field(default=None, description="Total view count")
    like_count: int | None = Field(default=None, description="Total like count")
    comment_count: int | None = Field(default=None, description="Total comment count")
    tags: list[str] = Field(default_factory=list, description="Video topic tags")
    is_live_content: bool = Field(default=False, description="True if live stream or premiere")
    is_available: bool = Field(default=True, description="True if public and playable")
    privacy_status: str = Field(default="public", description="public, unlisted, or private")


# ------------------------------------------------------------------------------
# Transcript Models
# ------------------------------------------------------------------------------


class SpeakerInfo(BaseModel):
    """Diarized speaker profile."""

    name: str = Field(description="Full name or canonical identifier of speaker")
    role: str | None = Field(default=None, description="Role in video (e.g. Host, Guest)")


class TranscriptTurn(BaseModel):
    """Verbatim transcript speaker turn with timestamp."""

    timestamp_seconds: int = Field(description="Offset timestamp in seconds from video start")
    timestamp_formatted: str = Field(description="Formatted timestamp (HH:MM:SS)")
    speaker_name: str = Field(description="Name of speaker")
    text: str = Field(description="Spoken text in this turn")


class TranscriptSegment(BaseModel):
    """Windowed transcript segment."""

    start_seconds: int = Field(description="Start time in seconds")
    end_seconds: int = Field(description="End time in seconds")
    speaker_name: str = Field(description="Speaker name")
    text: str = Field(description="Spoken text in segment")


class ValidationResult(BaseModel):
    """Structural quality validation metrics."""

    status: Literal["passed", "degraded", "failed"] = Field(description="Overall validation status")
    wpm: float = Field(description="Words per minute speaking rate")
    coverage_ratio: float = Field(description="Ratio of transcript time coverage to video duration")
    flags: list[str] = Field(default_factory=list, description="Quality flags or warnings")


class TranscriptSummary(BaseModel):
    """Structured transcript summary returned for short videos or initial window."""

    video_id: str = Field(description="YouTube video ID")
    title: str | None = Field(default=None, description="Video title")
    duration_seconds: int = Field(description="Video duration in seconds")
    status: Literal["ready", "degraded", "processing", "failed"] = Field(
        default="ready", description="Transcript readiness status"
    )
    speaker_registry: list[SpeakerInfo] = Field(
        default_factory=list, description="List of identified speakers"
    )
    total_turns: int = Field(default=0, description="Total number of speaker turns")
    total_words: int = Field(default=0, description="Total word count")
    wpm: float = Field(default=0.0, description="Calculated words per minute")
    initial_segments: list[TranscriptTurn] = Field(
        default_factory=list, description="First batch of transcript turns"
    )
    has_more: bool = Field(default=False, description="True if more segments exist")
    next_cursor: int | None = Field(
        default=None, description="Offset index for subsequent segments"
    )
    validation_flags: list[str] = Field(
        default_factory=list, description="Validation issues detected"
    )
    tokens_input: int | None = Field(default=None, description="Gemini input token count")
    tokens_output: int | None = Field(default=None, description="Gemini output token count")
    cost_usd: float | None = Field(default=None, description="Estimated compute cost in USD")


class MatchedSegmentSnippet(BaseModel):
    """Search match snippet from a transcript."""

    video_id: str = Field(description="YouTube video ID")
    start_seconds: int = Field(description="Timestamp in seconds")
    end_seconds: int = Field(description="End timestamp in seconds")
    timestamp_formatted: str = Field(description="Formatted timestamp (HH:MM:SS)")
    speaker_name: str = Field(description="Speaker name")
    text: str = Field(description="Matched verbatim text")
    context_before: str | None = Field(default=None, description="Preceding turn text")
    context_after: str | None = Field(default=None, description="Succeeding turn text")
    relevance_score: float = Field(default=1.0, description="Match score")
    youtube_link: str = Field(description="Clickable YouTube link with timestamp")


# ------------------------------------------------------------------------------
# Video Analysis Models (Public Video Analysis Mode — Fair Use)
# ------------------------------------------------------------------------------


class CreatorAttribution(BaseModel):
    """Mandatory creator credit included in every analysis output."""

    channel_title: str = Field(description="Creator/channel name")
    channel_id: str = Field(description="YouTube channel ID")
    channel_url: str = Field(description="YouTube channel URL")
    video_url: str = Field(description="Direct link to original video")
    notice: str = Field(
        default="Content by this creator. Video remains the property of its owner.",
        description="Attribution notice",
    )


class QuotedExcerpt(BaseModel):
    """Short fair-use excerpt with full attribution."""

    text: str = Field(description="Quoted text (max ~50 words / 2 sentences)")
    speaker_name: str = Field(description="Speaker who said this")
    timestamp_seconds: int = Field(description="Timestamp in seconds")
    timestamp_formatted: str = Field(description="Formatted timestamp (HH:MM:SS)")
    youtube_link: str = Field(description="Direct YouTube link to timestamp")


class TimestampEntry(BaseModel):
    """Topic-to-timestamp index entry."""

    topic: str = Field(description="Topic description")
    timestamp_seconds: int = Field(description="Timestamp in seconds")
    timestamp_formatted: str = Field(description="Formatted timestamp (HH:MM:SS)")
    youtube_link: str = Field(description="Direct YouTube link to timestamp")


class VideoAnalysis(BaseModel):
    """AI-generated analysis of a public video — NOT a full transcript."""

    video_id: str = Field(description="YouTube video ID")
    title: str | None = Field(default=None, description="Video title")
    duration_seconds: int = Field(description="Video duration in seconds")
    summary: str = Field(description="AI-generated analytical summary (500-800 words)")
    key_topics: list[str] = Field(description="Main topics discussed")
    notable_quotes: list[QuotedExcerpt] = Field(
        default_factory=list, description="Short attributed excerpts"
    )
    timestamp_index: list[TimestampEntry] = Field(
        default_factory=list, description="Topic-to-timestamp mapping"
    )
    speaker_registry: list[SpeakerInfo] = Field(default_factory=list)
    attribution: CreatorAttribution = Field(description="Mandatory creator credit")
    generated_at: str = Field(description="ISO 8601 generation timestamp")
    analysis_mode: str = Field(default="public", description="'public' or 'licensed'")
    tokens_input: int | None = Field(default=None, description="Gemini input token count")
    tokens_output: int | None = Field(default=None, description="Gemini output token count")
    cost_usd: float | None = Field(default=None, description="Estimated compute cost in USD")


class VideoEvidence(BaseModel):
    """Timestamped quoted excerpt providing direct factual backing for an answer."""

    timestamp_seconds: int = Field(description="Timestamp in seconds")
    timestamp_formatted: str = Field(description="Formatted timestamp (HH:MM:SS)")
    speaker_name: str = Field(description="Identified speaker")
    quote: str = Field(description="Short factual quote / excerpt")
    youtube_link: str = Field(description="Direct YouTube link to timestamp")


class VideoAnswerResponse(BaseModel):
    """Direct, synthesized answer to a deep question on a video (Fair Use — Ephemeral)."""

    video_id: str = Field(description="YouTube video ID")
    title: str | None = Field(default=None, description="Video title")
    query: str = Field(description="The user query or deep question asked")
    answer: str = Field(description="Comprehensive, deeply reasoned answer synthesizing the video content")
    evidence: list[VideoEvidence] = Field(
        default_factory=list, description="Timestamped citations and quotes backing the answer"
    )
    attribution: CreatorAttribution = Field(description="Mandatory creator credit")
    tokens_input: int | None = Field(default=None, description="Gemini input token count")
    tokens_output: int | None = Field(default=None, description="Gemini output token count")
    cost_usd: float | None = Field(default=None, description="Estimated compute cost in USD")


# ------------------------------------------------------------------------------
# YouTube Intelligence & Decision Synthesis Models
# ------------------------------------------------------------------------------


class RecommendationItem(BaseModel):
    """Structured explicit or implicit decision / recommendation extracted from video."""

    entity_or_topic: str = Field(description="The subject, ETF, stock, tool, strategy, or concept")
    action: Literal["buy", "sell", "allocate", "avoid", "adopt", "monitor", "review"] = Field(
        description="Recommended action"
    )
    stance: Literal["strongly_positive", "positive", "neutral", "cautious", "negative"] = Field(
        description="Creator's stance"
    )
    conviction_level: Literal["high", "medium", "low"] = Field(
        default="medium", description="Perceived conviction level"
    )
    target_audience: str = Field(
        default="", description="Who this recommendation applies to (e.g. high-income earners, startups)"
    )
    core_thesis: str = Field(description="Primary argument or justification for this recommendation")
    risks_or_tradeoffs: list[str] = Field(
        default_factory=list, description="Explicitly mentioned risks, fees, drawbacks, or edge cases"
    )
    alternatives_mentioned: list[str] = Field(
        default_factory=list, description="Alternative products, tools, or methods compared in video"
    )
    evidence: list[VideoEvidence] = Field(
        default_factory=list, description="Timestamped speaker citations backing this recommendation"
    )


class VideoRecommendationsResponse(BaseModel):
    """Structured inventory of recommendations and decisions made in a video."""

    video_id: str = Field(description="YouTube video ID")
    title: str | None = Field(default=None, description="Video title")
    focus_category: str | None = Field(default=None, description="Specific category filtered on, if any")
    total_recommendations: int = Field(default=0, description="Total recommendations extracted")
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    attribution: CreatorAttribution = Field(description="Mandatory creator credit")
    tokens_input: int | None = Field(default=None)
    tokens_output: int | None = Field(default=None)
    cost_usd: float | None = Field(default=None)


class FactualClaimItem(BaseModel):
    """Discrete factual claim or premise extracted from a video with verification context."""

    claim: str = Field(description="Discrete, falsifiable factual statement or argument made by creator")
    category: Literal[
        "data_or_metric",
        "historical_event",
        "technical_mechanic",
        "market_forecast",
        "creator_opinion",
    ] = Field(description="Type of claim")
    is_verifiable: bool = Field(default=True, description="True if claim is based on objective external data")
    creator_confidence: Literal["high", "moderate", "speculative"] = Field(default="high")
    verification_guidance: str = Field(
        default="",
        description="How an independent agent or user can verify this claim (e.g. check SEC filings, docs)",
    )
    evidence: list[VideoEvidence] = Field(
        default_factory=list, description="Timestamped citations where the claim is stated"
    )


class VideoClaimsResponse(BaseModel):
    """Structured inventory of discrete factual claims extracted from a video."""

    video_id: str = Field(description="YouTube video ID")
    title: str | None = Field(default=None, description="Video title")
    category_filter: str | None = Field(default=None)
    total_claims: int = Field(default=0)
    claims: list[FactualClaimItem] = Field(default_factory=list)
    attribution: CreatorAttribution = Field(description="Mandatory creator credit")
    tokens_input: int | None = Field(default=None)
    tokens_output: int | None = Field(default=None)
    cost_usd: float | None = Field(default=None)


class PersonalizedFitAnalysis(BaseModel):
    """Personalized relevance and applicability analysis of a video against user circumstances."""

    video_id: str = Field(description="YouTube video ID")
    title: str | None = Field(default=None, description="Video title")
    relevance_score: int = Field(description="Relevance score from 0 to 100")
    verdict: Literal[
        "must_watch",
        "key_takeaways_only",
        "partially_applicable",
        "not_applicable",
    ] = Field(description="Actionable verdict for user")
    executive_summary: str = Field(
        description="2-3 sentence personalized summary explaining why and how this video matters to the user"
    )
    applicable_points: list[str] = Field(
        default_factory=list, description="Key insights directly matching user profile or goals"
    )
    conflicts_or_caveats: list[str] = Field(
        default_factory=list,
        description="Assumptions in the video that do NOT match user constraints or profile",
    )
    custom_action_items: list[str] = Field(
        default_factory=list, description="Concrete next steps or adjustments tailored to the user"
    )
    evidence: list[VideoEvidence] = Field(
        default_factory=list, description="Timestamped evidence backing the fit evaluation"
    )
    attribution: CreatorAttribution = Field(description="Mandatory creator credit")
    tokens_input: int | None = Field(default=None)
    tokens_output: int | None = Field(default=None)
    cost_usd: float | None = Field(default=None)


class DisagreementDimension(BaseModel):
    """Structured breakdown of a specific point of divergence between videos/creators."""

    dimension: str = Field(description="The topic or technical dimension under disagreement")
    creator_positions: dict[str, str] = Field(
        description="Mapping of creator/video title to their stated position"
    )
    root_cause_of_divergence: str = Field(
        description="Why they disagree: different assumptions (taxes, scale, timeframe), different risk tolerances, or product definitions"
    )
    synthesis_verdict: str = Field(
        description="Objective breakdown reconciling when Creator A's view applies vs Creator B's"
    )


class CrossVideoComparisonResponse(BaseModel):
    """Structured cross-video comparison, consensus scorecard, and disagreement analysis."""

    topic: str = Field(description="Core topic being compared across videos")
    video_ids: list[str] = Field(description="List of YouTube video IDs compared")
    points_of_consensus: list[str] = Field(
        default_factory=list, description="Key principles or facts all analyzed creators agree on"
    )
    disagreements: list[DisagreementDimension] = Field(
        default_factory=list, description="Specific disagreements with identified root causes"
    )
    recommended_playbook: str = Field(
        description="Synthesized decision guide showing which approach to follow depending on user circumstances"
    )
    attributions: list[CreatorAttribution] = Field(
        default_factory=list, description="Mandatory attribution for each compared video"
    )
    tokens_input: int | None = Field(default=None)
    tokens_output: int | None = Field(default=None)
    cost_usd: float | None = Field(default=None)


# ------------------------------------------------------------------------------
# Error Taxonomy
# ------------------------------------------------------------------------------


class MCPErrorResponse(BaseModel):
    """Structured JSON-RPC error payload with recovery hints."""

    code: str = Field(description="Standardized error code (e.g. INVALID_INPUT, VIDEO_UNAVAILABLE)")
    message: str = Field(description="Human-readable error description")
    recovery_hint: str = Field(description="Actionable instruction for the AI agent / skill")
    retryable: bool = Field(default=False, description="True if caller should retry with backoff")
