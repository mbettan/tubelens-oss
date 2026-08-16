"""
Google Gemini 3.7 Flash Multimodal Video Intelligence & Research Engine
(Auto-fallback to Gemini 3.6 Flash)

Extracts structured decision intelligence, factual claims, personalized fit evaluations,
cross-video consensus, analytical summaries, and targeted Q&A directly from video audio/visuals
without downloading media files or storing raw verbatim transcripts (100% Fair Use compliant).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any, Literal

from google import genai
from google.genai import types

from src.config import settings
from src.errors import ErrorCode, ToolError
from src.models import (
    CreatorAttribution,
    CrossVideoComparisonResponse,
    DisagreementDimension,
    FactualClaimItem,
    PersonalizedFitAnalysis,
    QuotedExcerpt,
    RecommendationItem,
    SpeakerInfo,
    TimestampEntry,
    VideoAnalysis,
    VideoAnswerResponse,
    VideoClaimsResponse,
    VideoDetailedMetadata,
    VideoEvidence,
    VideoRecommendationsResponse,
)
from src.youtube import extract_video_id, format_seconds_to_hhmmss, youtube_client

logger = logging.getLogger("transcribe_engine")


def parse_timestamp_to_seconds(ts_str: str) -> int:
    """Parse HH:MM:SS or MM:SS or float string to total integer seconds."""
    try:
        cleaned = ts_str.strip().strip("[]\"'")
        parts = cleaned.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(float(h)) * 3600 + int(float(m)) * 60 + int(float(s))
        elif len(parts) == 2:
            m, s = parts
            return int(float(m)) * 60 + int(float(s))
        elif len(parts) == 1 and parts[0]:
            return int(float(parts[0]))
    except (ValueError, TypeError):
        pass
    return 0


def compute_gemini_cost(usage: Any | None) -> tuple[int, int, float]:
    """Extracts prompt/candidate token counts and computes estimated USD cost from configurable rates."""
    prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    cand_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    cost_usd = round(
        (prompt_tokens / 1_000_000.0) * settings.prompt_token_cost_per_million
        + (cand_tokens / 1_000_000.0) * settings.candidate_token_cost_per_million,
        5,
    )
    return prompt_tokens, cand_tokens, cost_usd


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Safely extracts structured JSON payload using raw_decode to handle nested objects."""
    cleaned = text.strip()
    # Check for markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        fenced_content = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced_content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            cleaned = fenced_content

    # Find the first opening brace and use raw_decode
    idx = cleaned.find("{")
    if idx >= 0:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(cleaned, idx)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            logger.warning(f"Failed to decode JSON block from model response: {e}")
    else:
        logger.warning("No JSON object '{' found in model response")
    return None


def _infer_claim_category(claim_text: str, llm_category: str, is_verifiable: bool) -> str:
    """
    Post-processing heuristic to correct LLM category bias.

    The LLM tends to default everything to 'data_or_metric'. This function
    overrides ONLY when the text strongly signals a different category.
    If the LLM chose a specific non-default category, we respect it.
    """
    # Only override the LLM's default bias — if it picked something specific, trust it
    if llm_category != "data_or_metric":
        return llm_category

    text_lower = claim_text.lower()

    # Strong signal: is_verifiable=false almost always means opinion/judgment
    if not is_verifiable:
        return "creator_opinion"

    # Opinion / judgment / subjective framework patterns
    _OPINION_PATTERNS = (
        "should", "best", "worst", "most important", "key to", "secret to",
        "fallacy", "hierarchy", "framework", "philosophy", "mindset",
        "believe", "recommend", "better than", "worse than", "overrated",
        "underrated", "essential", "unnecessary", "mistake", "wrong approach",
        "right approach", "ideal", "superior", "inferior",
    )
    if any(pat in text_lower for pat in _OPINION_PATTERNS):
        # Check it's not actually a metric (contains numbers/percentages)
        if not re.search(r"\d+\.?\d*\s*%|\$\d|\d+[xX]\b", claim_text):
            return "creator_opinion"

    # Forecast / prediction patterns (forward-looking language)
    _FORECAST_PATTERNS = (
        "will ", "going to", "predict", "forecast", "expect",
        "by 20", "in the next", "in the coming", "future",
        "projected", "anticipated",
    )
    if any(pat in text_lower for pat in _FORECAST_PATTERNS):
        return "market_forecast"

    # Technical mechanic patterns (how things work)
    _MECHANIC_PATTERNS = (
        "how ", "works by", "requires", "process of", "mechanism",
        "defined as", "means that", "operates", "functions by",
        "compound", "leverage works", "scales by",
    )
    if any(pat in text_lower for pat in _MECHANIC_PATTERNS):
        return "technical_mechanic"

    # Historical event patterns (past events with dates/names)
    _EVENT_PATTERNS = (
        "in 19", "in 20", "founded", "launched", "acquired",
        "went public", "was established", "historically",
    )
    if any(pat in text_lower for pat in _EVENT_PATTERNS):
        # But only if it doesn't have a clear metric focus
        if not re.search(r"\d+\.?\d*\s*%|\$[\d,]+", claim_text):
            return "historical_event"

    # Default: keep the LLM's original category
    return llm_category


def _is_model_fallback_eligible(exc: Exception) -> bool:
    """Whether an error should trigger fallback to the next model in the ladder."""
    text = f"{type(exc).__name__}: {exc}".upper()
    fallback_signals = (
        "NOT_FOUND", "404", "INVALID_ARGUMENT",
        "RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED", "429",
        "PERMISSION_DENIED", "403", "TIMEOUT", "DEADLINE_EXCEEDED",
    )
    return any(signal in text for signal in fallback_signals)


class TranscriptionEngine:
    """
    Multimodal Video Intelligence & Research Engine using Gemini on Vertex AI.
    Processes public YouTube videos in-cloud via Part.from_uri() with zero local downloads.
    """

    def __init__(self) -> None:
        self._genai_client: genai.Client | None = None
        self._lock = threading.Lock()

    def _get_client(self) -> genai.Client:
        if self._genai_client is None:
            with self._lock:
                if self._genai_client is None:
                    self._genai_client = genai.Client(
                        vertexai=True,
                        project=settings.google_cloud_project or None,
                        location=settings.google_cloud_location,
                        http_options=types.HttpOptions(api_version="v1"),
                    )
        return self._genai_client

    # --------------------------------------------------------------------------
    # Pre-Flight Metadata Gate (Step 0)
    # --------------------------------------------------------------------------

    async def preflight_check(self, video_id: str) -> VideoDetailedMetadata:
        """
        Executes pre-flight validation on YouTube Data API:
        1. Checks video existence and public availability
        2. Rejects live streams or upcoming premieres
        3. Rejects videos exceeding hard duration limit (4 hours)
        """
        details_list = await youtube_client.get_videos_batch([video_id])
        if not details_list:
            raise ToolError(
                ErrorCode.VIDEO_UNAVAILABLE,
                f"Video '{video_id}' not found or unavailable on YouTube.",
            )

        video_meta = details_list[0]

        if not video_meta.is_available:
            raise ToolError(
                ErrorCode.VIDEO_UNAVAILABLE,
                f"Video '{video_id}' is private, unlisted, or restricted ({video_meta.privacy_status}).",
            )

        if video_meta.is_live_content:
            raise ToolError(
                ErrorCode.LIVE_STREAM_NOT_SUPPORTED,
                f"Video '{video_id}' is an active live stream or premiere. Analysis requires on-demand VoD completion.",
            )

        if video_meta.duration_seconds > settings.max_video_duration_seconds:
            raise ToolError(
                ErrorCode.VIDEO_TOO_LONG,
                f"Video '{video_id}' duration ({video_meta.duration_seconds}s) exceeds maximum limit ({settings.max_video_duration_seconds}s).",
            )

        return video_meta

    # --------------------------------------------------------------------------
    # Multimodal Video Execution via Gemini
    # --------------------------------------------------------------------------

    async def _generate_with_fallback(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        timeout_seconds: float = 300.0,
    ) -> tuple[str, Any | None, float, str]:
        """
        Executes Gemini content generation stream with fallback ladder support.
        Returns (raw_text, usage_metadata, latency_seconds, actual_model_used).
        """
        client = self._get_client()
        start_time = time.time()
        ladder = settings.model_ladder
        last_error: Exception | None = None

        for position, model_id in enumerate(ladder):
            def _call_gemini(mid: str = model_id) -> tuple[str, Any | None]:
                response_stream = client.models.generate_content_stream(
                    model=mid,
                    contents=contents,
                    config=config,
                )
                chunks: list[str] = []
                final_usage = None
                for chunk in response_stream:
                    if chunk.usage_metadata:
                        final_usage = chunk.usage_metadata
                    if chunk.text:
                        chunks.append(chunk.text)
                return "".join(chunks), final_usage

            try:
                raw_text, usage = await asyncio.wait_for(
                    asyncio.to_thread(_call_gemini),
                    timeout=timeout_seconds,
                )
                latency = round(time.time() - start_time, 2)
                if position > 0:
                    logger.info(
                        f"Model fallback succeeded: {model_id} (after {position} failed model(s))"
                    )
                return raw_text, usage, latency, model_id
            except Exception as exc:
                last_error = exc
                if position + 1 >= len(ladder) or not _is_model_fallback_eligible(exc):
                    break
                logger.warning(
                    f"Model {model_id} unavailable ({type(exc).__name__}), "
                    f"falling back to {ladder[position + 1]}"
                )

        latency = round(time.time() - start_time, 2)
        if last_error:
            raise last_error
        raise RuntimeError("No models available in fallback ladder")

    async def transcribe_window_direct(
        self,
        url: str,
        diarize: bool = True,
        temperature: float = 0.1,
        start_seconds: int = 0,
        end_seconds: int = 0,
        custom_prompt: str | None = None,
        timeout_seconds: float = 300.0,
    ) -> tuple[str, Any | None, float, str]:
        """
        Invokes Gemini on Vertex AI for a public video URL.
        Tries models in the fallback ladder on unavailable or quota errors.
        Returns (raw_text, usage_metadata, latency_seconds, actual_model_used).
        """
        prompt = custom_prompt or (
            "Analyze this video and provide a comprehensive structured overview with timestamped citations."
        )

        vid = extract_video_id(url)
        canonical_url = f"https://www.youtube.com/watch?v={vid}" if vid else url
        video_part = types.Part.from_uri(
            file_uri=canonical_url,
            mime_type="video/mp4",
        )

        if start_seconds > 0 or end_seconds > 0:
            video_part.video_metadata = types.VideoMetadata(
                start_offset=f"{start_seconds}s" if start_seconds > 0 else None,
                end_offset=f"{end_seconds}s" if end_seconds > 0 else None,
            )

        contents = [
            types.Content(
                role="user",
                parts=[
                    video_part,
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            max_output_tokens=65536,
            temperature=temperature,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        return await self._generate_with_fallback(
            contents=contents,
            config=generate_content_config,
            timeout_seconds=timeout_seconds,
        )

    # --------------------------------------------------------------------------
    # 1. Public Video Analysis Mode (Fair Use — Transformative Overview)
    # --------------------------------------------------------------------------

    def _build_analysis_prompt(self, focus_question: str | None = None) -> str:
        prompt_parts = [
            "Analyze this video and produce a structured analysis (NOT a verbatim transcript).\n\n"
            "IMPORTANT: Write the summary in YOUR OWN WORDS. Do NOT reproduce dialogue verbatim "
            "except in the NOTABLE QUOTES section where short attributed excerpts are allowed.\n\n"
            "Output the following sections using these exact headers:\n\n"
            "## SUMMARY\n"
            "A comprehensive analytical summary (500-800 words) covering the video's content, "
            "arguments, key insights, and conclusions. Write analytically.\n\n"
            "## KEY TOPICS\n"
            "A bulleted list of the main subjects, themes, and arguments discussed.\n\n"
            "## NOTABLE QUOTES\n"
            f"Up to {settings.max_excerpts_per_analysis} short attributed excerpts "
            f"(each max ~{settings.max_excerpt_words} words / 2 sentences) that capture key insights. "
            "Format each as:\n"
            "[HH:MM:SS] Speaker Name: \"quoted text\"\n\n"
            "## TIMESTAMP INDEX\n"
            "Map each major topic or segment to its approximate start time.\n"
            "Format each as:\n"
            "[HH:MM:SS] Topic description\n\n"
            "## SPEAKERS\n"
            "List each speaker identified in the video with their role if known.\n"
            "Format each as:\n"
            "- Speaker Name (Role)\n\n"
            "Always identify speakers by name using visual and audio cues "
            "(lower-third graphics, introductions, name badges)."
        ]
        if focus_question and focus_question.strip():
            clean_q = focus_question.strip()[:1000]
            prompt_parts.append(
                f"\n\nFOCUS QUESTION:\n<focus_question>\nPay special attention to content related to: '{clean_q}'. "
                "Prioritize excerpts and topics relevant to this question.\n</focus_question>"
            )
        return "".join(prompt_parts)

    def _parse_analysis_response(
        self, raw_text: str, video_meta: VideoDetailedMetadata
    ) -> VideoAnalysis:
        video_id = video_meta.video_id
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # 1. Summary
        summary = ""
        summary_match = re.search(
            r"##\s*SUMMARY\s*\n+(.*?)(?=\n+##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if summary_match:
            summary = summary_match.group(1).strip()

        # 2. Key Topics
        topics: list[str] = []
        topics_match = re.search(
            r"##\s*KEY TOPICS\s*\n+(.*?)(?=\n+##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if topics_match:
            for line in topics_match.group(1).splitlines():
                clean_line = re.sub(r"^[\s*\-•\d.]+\s*", "", line).strip()
                if clean_line and len(clean_line) > 2:
                    topics.append(clean_line)

        # 3. Notable Quotes / Short Excerpts
        quotes: list[QuotedExcerpt] = []
        quotes_match = re.search(
            r"##\s*NOTABLE QUOTES\s*\n+(.*?)(?=\n+##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if quotes_match:
            for line in quotes_match.group(1).splitlines():
                line = line.strip()
                if not line:
                    continue
                # Strip optional bullet prefix
                clean = re.sub(r"^[-*•]\s*", "", line)
                if not clean:
                    continue
                # Parse: [00:14:20] Speaker Name: "quoted text"  (prompt format)
                # Also handles: [00:14:20] "Quote text" — Speaker Name  (alternative)
                m = re.search(
                    r"\[(?P<ts>[\d:]+)\]\s*(?P<rest>.+)",
                    clean,
                )
                if not m:
                    continue
                ts_str = m.group("ts")
                sec = parse_timestamp_to_seconds(ts_str)
                rest = m.group("rest").strip()
                speaker = "Speaker"
                quote_text = rest

                # Try prompt format: Speaker Name: "quoted text"
                colon_match = re.match(
                    r'(?P<speaker>[^:"\u201c]+):\s*["\u201c\u201d\'\u2018\u2019]?(?P<text>[^"\u201c\u201d\n\r]+)["\u201c\u201d\'\u2018\u2019]?',
                    rest,
                )
                if colon_match:
                    speaker = colon_match.group("speaker").strip()
                    quote_text = colon_match.group("text").strip()
                else:
                    # Try alternative: "Quote text" — Speaker Name
                    alt_match = re.match(
                        r'["\u201c\u201d\'\u2018\u2019](?P<text>[^"\u201c\u201d\n\r]+)["\u201c\u201d\'\u2018\u2019]?\s*[—\-–]\s*(?P<speaker>.+)',
                        rest,
                    )
                    if alt_match:
                        quote_text = alt_match.group("text").strip()
                        speaker = alt_match.group("speaker").strip()

                # Enforce max excerpt length (~50 words for transformative compliance)
                words = quote_text.split()
                if len(words) > settings.max_excerpt_words:
                    quote_text = " ".join(words[: settings.max_excerpt_words]) + "..."
                quotes.append(
                    QuotedExcerpt(
                        text=quote_text,
                        speaker_name=speaker,
                        timestamp_seconds=sec,
                        timestamp_formatted=format_seconds_to_hhmmss(sec),
                        youtube_link=f"{video_url}&t={sec}",
                    )
                )

        # 4. Timestamp Index
        index_entries: list[TimestampEntry] = []
        index_match = re.search(
            r"##\s*TIMESTAMP INDEX\s*\n+(.*?)(?=\n+##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if index_match:
            for line in index_match.group(1).splitlines():
                line = line.strip()
                if not line:
                    continue
                # Strip optional bullet prefix
                clean = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
                if not clean:
                    continue
                m = re.search(r"\[(?P<ts>[\d:]+)\]\s*(?P<topic>[^\n\r]+)", clean)
                if m:
                    ts_str = m.group("ts")
                    sec = parse_timestamp_to_seconds(ts_str)
                    topic_text = m.group("topic").strip().lstrip(":- ").strip()
                    index_entries.append(
                        TimestampEntry(
                            topic=topic_text,
                            timestamp_seconds=sec,
                            timestamp_formatted=format_seconds_to_hhmmss(sec),
                            youtube_link=f"{video_url}&t={sec}",
                        )
                    )

        # 5. Speaker Registry
        speakers: list[SpeakerInfo] = []
        speakers_match = re.search(
            r"##\s*SPEAKERS\s*\n+(.*?)(?=\n+##|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if speakers_match:
            for line in speakers_match.group(1).splitlines():
                line = line.strip()
                if not line or not (line.startswith("-") or line.startswith("*")):
                    continue
                m = re.search(r"[\-*]\s*(?P<name>[^:\(—–]+)(?:\s*[\(:—–]\s*(?P<role>[^\)\n\r]+)\)?)?", line)
                if m:
                    name = m.group("name").strip()
                    role = (m.group("role") or "").strip()
                    if name:
                        speakers.append(SpeakerInfo(name=name, role=role if role else None))

        attribution = CreatorAttribution(
            channel_title=video_meta.channel_title,
            channel_id=video_meta.channel_id,
            channel_url=f"https://www.youtube.com/channel/{video_meta.channel_id}",
            video_url=video_url,
        )

        fallback_summary = raw_text.strip()[:1000] if raw_text.strip() else "Comprehensive analytical overview generated from video audio/visuals."
        return VideoAnalysis(
            video_id=video_id,
            title=video_meta.title,
            duration_seconds=video_meta.duration_seconds,
            summary=summary or fallback_summary,
            key_topics=topics[:15],
            notable_quotes=quotes[: settings.max_excerpts_per_analysis],
            timestamp_index=index_entries,
            speaker_registry=speakers,
            attribution=attribution,
            generated_at=datetime.now(UTC).isoformat(),
            analysis_mode="public",
        )

    async def analyze_video(
        self,
        url: str,
        video_meta: VideoDetailedMetadata,
        focus_question: str | None = None,
    ) -> VideoAnalysis:
        """Generates structured analysis (summary, excerpts, timestamps) under Fair Use ephemerally."""
        video_id = video_meta.video_id
        try:
            prompt = self._build_analysis_prompt(focus_question)
            raw_text, usage, latency, model = await self.transcribe_window_direct(
                url=url,
                diarize=True,
                temperature=0.1,
                custom_prompt=prompt,
            )

            analysis = self._parse_analysis_response(raw_text, video_meta)
            prompt_tokens, cand_tokens, cost_usd = compute_gemini_cost(usage)
            analysis.tokens_input = prompt_tokens
            analysis.tokens_output = cand_tokens
            analysis.cost_usd = cost_usd
            return analysis

        except Exception as e:
            if not isinstance(e, ToolError):
                logger.error(f"Error during analysis of {video_id}: {e}")
            raise

    # --------------------------------------------------------------------------
    # 2. Ephemeral Deep Video Q&A
    # --------------------------------------------------------------------------

    def _build_ask_video_prompt(self, query: str) -> str:
        clean_query = query.strip()[:2000]
        return (
            "You are an expert video analyst providing an authoritative, deeply reasoned answer "
            "to the following user query about this video:\n\n"
            f"USER QUERY:\n<user_query>\n{clean_query}\n</user_query>\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the query thoroughly, analytically, and directly in YOUR OWN WORDS.\n"
            "2. Ground your answer in exact evidence, arguments, discussions, on-screen demonstrations, and slides.\n"
            "3. Do NOT produce a full verbatim transcript. This is a targeted analytical Q&A response (Fair Use).\n"
            "4. Provide timestamped citations for key claims.\n\n"
            "OUTPUT FORMAT:\n"
            "Structure your output using these exact headers:\n\n"
            "## DIRECT ANSWER\n"
            "A comprehensive, structured, in-depth explanation answering the user query directly.\n\n"
            "## KEY EVIDENCE\n"
            "Up to 5 specific, short quoted excerpts with speaker attribution and timestamps that substantiate the answer.\n"
            "Format each as:\n"
            "[HH:MM:SS] Speaker Name: \"short quote / excerpt\"\n"
        )

    def _parse_ask_video_response(
        self, raw_text: str, query: str, video_meta: VideoDetailedMetadata
    ) -> VideoAnswerResponse:
        video_id = video_meta.video_id
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        answer = ""
        evidence: list[VideoEvidence] = []

        answer_match = re.search(
            r"#+\s*DIRECT ANSWER\s*\n(.*?)(?=\n#+\s|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = re.split(r"#+\s*KEY EVIDENCE", raw_text, flags=re.IGNORECASE)[0].strip()

        evidence_match = re.search(
            r"#+\s*KEY EVIDENCE\s*\n(.*?)(?=\n#+\s|\Z)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if evidence_match:
            quote_pattern = re.compile(
                r'\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<speaker>[^:\n]+):\s*["“\']?(?P<text>[^"”\n\r]+)["”\']?',
            )
            for qm in quote_pattern.finditer(evidence_match.group(1)):
                ts_seconds = parse_timestamp_to_seconds(qm.group("ts"))
                evidence.append(
                    VideoEvidence(
                        timestamp_seconds=ts_seconds,
                        timestamp_formatted=format_seconds_to_hhmmss(ts_seconds),
                        speaker_name=qm.group("speaker").strip(),
                        quote=qm.group("text").strip(),
                        youtube_link=f"{video_url}&t={ts_seconds}",
                    )
                )

        attribution = CreatorAttribution(
            channel_title=video_meta.channel_title,
            channel_id=video_meta.channel_id,
            channel_url=f"https://www.youtube.com/channel/{video_meta.channel_id}",
            video_url=video_url,
        )

        return VideoAnswerResponse(
            video_id=video_id,
            title=video_meta.title,
            query=query,
            answer=answer,
            evidence=evidence,
            attribution=attribution,
        )

    async def ask_video(
        self,
        url: str,
        query: str,
        video_meta: VideoDetailedMetadata,
        time_window_start_s: int | None = None,
        time_window_end_s: int | None = None,
    ) -> VideoAnswerResponse:
        """Answers a deep, specific question about a video without storing or exposing transcripts."""
        video_id = video_meta.video_id
        try:
            prompt = self._build_ask_video_prompt(query)
            raw_text, usage, latency, model = await self.transcribe_window_direct(
                url=url,
                diarize=True,
                temperature=0.1,
                custom_prompt=prompt,
            )

            answer = self._parse_ask_video_response(raw_text, query, video_meta)
            prompt_tokens, cand_tokens, cost_usd = compute_gemini_cost(usage)
            answer.tokens_input = prompt_tokens
            answer.tokens_output = cand_tokens
            answer.cost_usd = cost_usd
            return answer
        except Exception as e:
            if not isinstance(e, ToolError):
                logger.error(f"Error in ask_video for {video_id}: {e}")
            raise

    # --------------------------------------------------------------------------
    # 3. Recommendations & Decision Intelligence
    # --------------------------------------------------------------------------

    def _build_recommendations_prompt(self, focus_category: str | None = None) -> str:
        cat_clause = f" focusing especially on '{focus_category}'" if focus_category else ""
        return (
            f"Extract all explicit and implicit recommendations from this video{cat_clause}.\n"
            "For EACH recommendation or decision item, provide:\n"
            "1. Item/Topic (e.g. tool name, fund ticker, architectural pattern, practice)\n"
            "2. Stance: strongly_positive | positive | neutral | negative | strongly_negative\n"
            "3. Action: allocate | adopt | avoid | hold | monitor | research_further\n"
            "4. Conviction: high | medium | low\n"
            "5. Core thesis (1-2 sentences)\n"
            "6. Tradeoffs & risks mentioned\n"
            "7. Alternatives discussed\n"
            "8. Timestamp citation backing the recommendation\n\n"
            "Format output strictly as JSON:\n"
            "```json\n"
            "{\n"
            '  "recommendations": [\n'
            "    {\n"
            '      "item_or_topic": "Direct Indexing / SMA",\n'
            '      "category": "investing",\n'
            '      "action": "allocate",\n'
            '      "stance": "positive",\n'
            '      "conviction": "high",\n'
            '      "thesis": "Enables granular tax-loss harvesting and custom factor tilts for large taxable portfolios.",\n'
            '      "tradeoffs_and_risks": ["Tracking error vs S&P 500", "0.25% fee drag"],\n'
            '      "alternatives_mentioned": ["VOO", "VTI"],\n'
            '      "evidence": [{"timestamp": "00:14:20", "speaker_name": "Host", "quote": "For portfolios over $500k in taxable accounts, direct indexing provides 100-150 bps in tax alpha."}]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )

    def _parse_recommendations_response(
        self, raw_text: str, video_meta: VideoDetailedMetadata, focus_category: str | None
    ) -> VideoRecommendationsResponse:
        video_id = video_meta.video_id
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        json_obj = _extract_json_block(raw_text)

        recs: list[RecommendationItem] = []
        if json_obj and isinstance(json_obj.get("recommendations"), list):
            for item in json_obj["recommendations"]:
                evidence_list: list[VideoEvidence] = []
                for ev in item.get("evidence", []):
                    ts_str = str(ev.get("timestamp", "00:00:00"))
                    sec = parse_timestamp_to_seconds(ts_str)
                    evidence_list.append(
                        VideoEvidence(
                            timestamp_seconds=sec,
                            timestamp_formatted=format_seconds_to_hhmmss(sec),
                            speaker_name=str(ev.get("speaker_name", "Speaker")),
                            quote=str(ev.get("quote", "")),
                            youtube_link=f"{video_url}&t={sec}",
                        )
                    )

                action_raw = str(item.get("action", "allocate")).lower()
                valid_actions = ("buy", "sell", "allocate", "avoid", "adopt", "monitor", "review")
                action_val = action_raw if action_raw in valid_actions else "allocate"

                stance_raw = str(item.get("stance", "positive")).lower()
                valid_stances = ("strongly_positive", "positive", "neutral", "cautious", "negative")
                stance_val = stance_raw if stance_raw in valid_stances else "positive"

                conv_raw = str(item.get("conviction", item.get("conviction_level", "medium"))).lower()
                valid_convs = ("high", "medium", "low")
                conv_val = conv_raw if conv_raw in valid_convs else "medium"

                recs.append(
                    RecommendationItem(
                        entity_or_topic=str(item.get("item_or_topic", item.get("entity_or_topic", "General"))),
                        action=action_val,  # type: ignore[arg-type]
                        stance=stance_val,  # type: ignore[arg-type]
                        conviction_level=conv_val,  # type: ignore[arg-type]
                        target_audience=str(item.get("target_audience", "")),
                        core_thesis=str(item.get("thesis", item.get("core_thesis", ""))),
                        risks_or_tradeoffs=[str(x) for x in item.get("tradeoffs_and_risks", item.get("risks_or_tradeoffs", []))],
                        alternatives_mentioned=[str(x) for x in item.get("alternatives_mentioned", [])],
                        evidence=evidence_list,
                    )
                )

        attribution = CreatorAttribution(
            channel_title=video_meta.channel_title,
            channel_id=video_meta.channel_id,
            channel_url=f"https://www.youtube.com/channel/{video_meta.channel_id}",
            video_url=video_url,
        )

        return VideoRecommendationsResponse(
            video_id=video_id,
            title=video_meta.title,
            focus_category=focus_category,
            total_recommendations=len(recs),
            recommendations=recs,
            attribution=attribution,
        )

    async def extract_recommendations(
        self,
        url: str,
        video_meta: VideoDetailedMetadata,
        focus_category: str | None = None,
    ) -> VideoRecommendationsResponse:
        """Extracts structured explicit and implicit recommendations from a video ephemerally."""
        video_id = video_meta.video_id
        try:
            prompt = self._build_recommendations_prompt(focus_category)
            raw_text, usage, latency, model = await self.transcribe_window_direct(
                url=url,
                diarize=True,
                temperature=0.1,
                custom_prompt=prompt,
            )

            result = self._parse_recommendations_response(raw_text, video_meta, focus_category)
            prompt_tokens, cand_tokens, cost_usd = compute_gemini_cost(usage)
            result.tokens_input = prompt_tokens
            result.tokens_output = cand_tokens
            result.cost_usd = cost_usd
            return result
        except Exception as e:
            if not isinstance(e, ToolError):
                logger.error(f"Error extracting recommendations for {video_id}: {e}")
            raise

    # --------------------------------------------------------------------------
    # 4. Factual Claims & Verification Deconstruction
    # --------------------------------------------------------------------------

    def _build_claims_prompt(self, category_filter: str | None = None) -> str:
        cat_clause = f" focusing on claims related to '{category_filter}'" if category_filter else ""
        return (
            f"Deconstruct this video into its core factual claims and key assertions{cat_clause}.\n"
            "Distinguish verifiable factual assertions from subjective commentary.\n\n"
            "For EACH claim:\n"
            "1. Claim text (concise factual statement)\n"
            "2. Category — choose the SINGLE most accurate category:\n"
            "   - data_or_metric: Specific numbers, percentages, statistics, or measurable quantities\n"
            "   - historical_event: References to past events, dates, or documented occurrences\n"
            "   - technical_mechanic: Descriptions of how systems, frameworks, or processes work\n"
            "   - market_forecast: Forward-looking predictions about markets, trends, or outcomes\n"
            "   - creator_opinion: Subjective judgments, value statements, beliefs, or editorial positions\n"
            "3. Verifiable: true if it can be verified against independent data/literature; false if purely subjective\n"
            "4. Confidence: high | moderate | speculative\n"
            "5. Verification guidance: What source, benchmark, or empirical dataset should be consulted?\n"
            "6. Timestamp citation backing the claim\n\n"
            "Format output strictly as JSON:\n"
            "```json\n"
            "{\n"
            '  "claims": [\n'
            "    {\n"
            '      "claim_text": "Direct indexing generated an average of 1.2% annual tax alpha over 10-year holding periods.",\n'
            '      "category": "data_or_metric",\n'
            '      "verifiable": true,\n'
            '      "confidence": "high",\n'
            '      "verification_guidance": "Cross-reference empirical studies from Parametric/Vanguard TLH whitepapers.",\n'
            '      "evidence": [{"timestamp": "00:08:30", "speaker_name": "Speaker", "quote": "Historical studies show about 120 basis points of tax alpha."}]\n'
            "    },\n"
            "    {\n"
            '      "claim_text": "The hierarchy of leverage is: code, media, capital, then labor — in that order.",\n'
            '      "category": "creator_opinion",\n'
            '      "verifiable": false,\n'
            '      "confidence": "high",\n'
            '      "verification_guidance": "This is a subjective ranking. Compare with frameworks from Naval Ravikant, Paul Graham, etc.",\n'
            '      "evidence": [{"timestamp": "00:15:00", "speaker_name": "Speaker", "quote": "Code and media are the highest leverage."}]\n'
            "    },\n"
            "    {\n"
            '      "claim_text": "Compound interest requires keeping capital invested through market drawdowns.",\n'
            '      "category": "technical_mechanic",\n'
            '      "verifiable": true,\n'
            '      "confidence": "high",\n'
            '      "verification_guidance": "Standard financial mathematics — verify with any compound growth model.",\n'
            '      "evidence": [{"timestamp": "00:22:10", "speaker_name": "Speaker", "quote": "The math only works if you stay in."}]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )

    def _parse_claims_response(
        self, raw_text: str, video_meta: VideoDetailedMetadata, category_filter: str | None
    ) -> VideoClaimsResponse:
        video_id = video_meta.video_id
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        json_obj = _extract_json_block(raw_text)

        claims: list[FactualClaimItem] = []
        if json_obj and isinstance(json_obj.get("claims"), list):
            for item in json_obj["claims"]:
                evidence_list: list[VideoEvidence] = []
                for ev in item.get("evidence", []):
                    ts_str = str(ev.get("timestamp", "00:00:00"))
                    sec = parse_timestamp_to_seconds(ts_str)
                    evidence_list.append(
                        VideoEvidence(
                            timestamp_seconds=sec,
                            timestamp_formatted=format_seconds_to_hhmmss(sec),
                            speaker_name=str(ev.get("speaker_name", "Speaker")),
                            quote=str(ev.get("quote", "")),
                            youtube_link=f"{video_url}&t={sec}",
                        )
                    )

                # Map model output to valid category Literals (handles legacy prompt names too)
                cat_raw = str(item.get("category", item.get("claim_type", ""))).lower().strip()
                _CATEGORY_MAP: dict[str, str] = {
                    "data_or_metric": "data_or_metric",
                    "historical_event": "historical_event",
                    "technical_mechanic": "technical_mechanic",
                    "market_forecast": "market_forecast",
                    "creator_opinion": "creator_opinion",
                    # Legacy prompt mappings
                    "empirical_observation": "historical_event",
                    "technical_assertion": "technical_mechanic",
                    "prediction_or_forecast": "market_forecast",
                    "opinion_or_judgment": "creator_opinion",
                    "opinion": "creator_opinion",
                    "forecast": "market_forecast",
                    "prediction": "market_forecast",
                }
                cat_val = _CATEGORY_MAP.get(cat_raw, "data_or_metric")

                # Map confidence values (handles legacy medium/low from old prompts)
                conf_raw = str(item.get("confidence", item.get("creator_confidence", "high"))).lower().strip()
                _CONFIDENCE_MAP: dict[str, str] = {
                    "high": "high",
                    "moderate": "moderate",
                    "speculative": "speculative",
                    "medium": "moderate",
                    "low": "speculative",
                }
                conf_val = _CONFIDENCE_MAP.get(conf_raw, "high")

                claim_text = str(item.get("claim_text", item.get("claim", "")))
                is_verifiable = bool(item.get("verifiable", item.get("is_verifiable", True)))

                # Post-process: override category when text signals a clear mismatch
                cat_val = _infer_claim_category(claim_text, cat_val, is_verifiable)

                claims.append(
                    FactualClaimItem(
                        claim=claim_text,
                        category=cat_val,  # type: ignore[arg-type]
                        is_verifiable=is_verifiable,
                        creator_confidence=conf_val,  # type: ignore[arg-type]
                        verification_guidance=str(item.get("verification_guidance", "")),
                        evidence=evidence_list,
                    )
                )

        attribution = CreatorAttribution(
            channel_title=video_meta.channel_title,
            channel_id=video_meta.channel_id,
            channel_url=f"https://www.youtube.com/channel/{video_meta.channel_id}",
            video_url=video_url,
        )

        return VideoClaimsResponse(
            video_id=video_id,
            title=video_meta.title,
            category_filter=category_filter,
            total_claims=len(claims),
            claims=claims,
            attribution=attribution,
        )

    async def extract_claims(
        self,
        url: str,
        video_meta: VideoDetailedMetadata,
        category_filter: str | None = None,
    ) -> VideoClaimsResponse:
        """Extracts discrete verifiable factual claims from a video ephemerally."""
        video_id = video_meta.video_id
        try:
            prompt = self._build_claims_prompt(category_filter)
            raw_text, usage, latency, model = await self.transcribe_window_direct(
                url=url,
                diarize=True,
                temperature=0.1,
                custom_prompt=prompt,
            )

            result = self._parse_claims_response(raw_text, video_meta, category_filter)
            prompt_tokens, cand_tokens, cost_usd = compute_gemini_cost(usage)
            result.tokens_input = prompt_tokens
            result.tokens_output = cand_tokens
            result.cost_usd = cost_usd
            return result
        except Exception as e:
            if not isinstance(e, ToolError):
                logger.error(f"Error extracting claims for {video_id}: {e}")
            raise

    # --------------------------------------------------------------------------
    # 5. Personalized Fit Evaluation ("What Matters to Me?")
    # --------------------------------------------------------------------------

    def _build_fit_prompt(self, user_profile: str, constraints: list[str] | None = None) -> str:
        trimmed_profile = user_profile.strip()
        if len(trimmed_profile) > 2000:
            logger.warning("user_profile exceeded 2000 characters and was truncated.")
        clean_profile = trimmed_profile[:2000]
        clean_constraints = [c.strip()[:500] for c in (constraints or [])][:20]
        constraints_str = ", ".join(clean_constraints) if clean_constraints else "None specified"
        return (
            "Evaluate how this video's arguments and recommendations apply to the following user profile and constraints.\n\n"
            f"USER PROFILE:\n<user_profile>\n{clean_profile}\n</user_profile>\n\n"
            f"USER CONSTRAINTS / OBJECTIVES:\n<user_constraints>\n{constraints_str}\n</user_constraints>\n\n"
            "INSTRUCTIONS:\n"
            "1. Score relevance (0 to 100).\n"
            "2. Decide verdict: must_watch | key_takeaways_only | partially_applicable | not_applicable.\n"
            "3. Write a 2-3 sentence personalized executive summary explaining relevance to this specific user.\n"
            "4. List applicable points directly matching user goals.\n"
            "5. List conflicts or caveats (assumptions in video that do NOT match user situation).\n"
            "6. List custom action items tailored to the user.\n"
            "7. Provide timestamped citations backing your evaluation.\n\n"
            "Format output as JSON:\n"
            "```json\n"
            "{\n"
            '  "relevance_score": 88,\n'
            '  "verdict": "key_takeaways_only",\n'
            '  "executive_summary": "The discussion on Roth conversions directly applies to your high income bracket, but the real estate section can be skipped.",\n'
            '  "applicable_points": ["Roth conversion tax bracket arbitrage", "Asset location rules"],\n'
            '  "conflicts_or_caveats": ["Assumes California state tax rather than Texas zero state tax"],\n'
            '  "custom_action_items": ["Model backdoor Roth conversion before year-end", "Adjust fixed income allocation in 401(k)"],\n'
            '  "evidence": [\n'
            '    {"timestamp": "00:11:45", "speaker_name": "Speaker", "quote": "If you are in the top bracket today, converting during a market dip is optimal."}\n'
            "  ]\n"
            "}\n"
            "```"
        )

    def _parse_fit_response(
        self, raw_text: str, video_meta: VideoDetailedMetadata
    ) -> PersonalizedFitAnalysis:
        video_id = video_meta.video_id
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        json_obj = _extract_json_block(raw_text)

        relevance = 50
        verdict_val: Literal["must_watch", "key_takeaways_only", "partially_applicable", "not_applicable"] = "partially_applicable"
        summary = "Personalized evaluation completed."
        applicable_pts: list[str] = []
        conflicts: list[str] = []
        actions: list[str] = []
        evidence_list: list[VideoEvidence] = []

        if json_obj:
            relevance = int(json_obj.get("relevance_score", 50))
            raw_v = str(json_obj.get("verdict", "partially_applicable")).lower()
            if raw_v in ("must_watch", "key_takeaways_only", "partially_applicable", "not_applicable"):
                verdict_val = raw_v  # type: ignore[assignment]
            summary = str(json_obj.get("executive_summary", summary))
            applicable_pts = [str(x) for x in json_obj.get("applicable_points", [])]
            conflicts = [str(x) for x in json_obj.get("conflicts_or_caveats", [])]
            actions = [str(x) for x in json_obj.get("custom_action_items", [])]
            for ev in json_obj.get("evidence", []):
                ts_str = str(ev.get("timestamp", "00:00:00"))
                sec = parse_timestamp_to_seconds(ts_str)
                evidence_list.append(
                    VideoEvidence(
                        timestamp_seconds=sec,
                        timestamp_formatted=format_seconds_to_hhmmss(sec),
                        speaker_name=str(ev.get("speaker_name", "Speaker")),
                        quote=str(ev.get("quote", "")),
                        youtube_link=f"{video_url}&t={sec}",
                    )
                )

        attribution = CreatorAttribution(
            channel_title=video_meta.channel_title,
            channel_id=video_meta.channel_id,
            channel_url=f"https://www.youtube.com/channel/{video_meta.channel_id}",
            video_url=video_url,
        )

        return PersonalizedFitAnalysis(
            video_id=video_id,
            title=video_meta.title,
            relevance_score=relevance,
            verdict=verdict_val,
            executive_summary=summary,
            applicable_points=applicable_pts,
            conflicts_or_caveats=conflicts,
            custom_action_items=actions,
            evidence=evidence_list,
            attribution=attribution,
        )

    async def evaluate_fit(
        self,
        url: str,
        video_meta: VideoDetailedMetadata,
        user_profile: str,
        constraints: list[str] | None = None,
    ) -> PersonalizedFitAnalysis:
        """Evaluates video relevance and creates tailored action items for a specific user profile ephemerally."""
        video_id = video_meta.video_id
        try:
            prompt = self._build_fit_prompt(user_profile, constraints)
            raw_text, usage, latency, model = await self.transcribe_window_direct(
                url=url,
                diarize=True,
                temperature=0.1,
                custom_prompt=prompt,
            )

            result = self._parse_fit_response(raw_text, video_meta)
            prompt_tokens, cand_tokens, cost_usd = compute_gemini_cost(usage)
            result.tokens_input = prompt_tokens
            result.tokens_output = cand_tokens
            result.cost_usd = cost_usd
            return result
        except Exception as e:
            if not isinstance(e, ToolError):
                logger.error(f"Error evaluating fit for {video_id}: {e}")
            raise

    # --------------------------------------------------------------------------
    # 6. Cross-Video Comparison & Disagreement Matrix
    # --------------------------------------------------------------------------

    def _build_comparison_prompt(self, topic: str, video_titles: list[str]) -> str:
        clean_topic = topic.strip()[:1000]
        v_list = "\n".join(f"Video {i+1}: {t}" for i, t in enumerate(video_titles))
        return (
            f"You are an expert analyst comparing {len(video_titles)} YouTube videos discussing the following topic.\n\n"
            f"TOPIC TO COMPARE:\n<topic>\n{clean_topic}\n</topic>\n\n"
            f"VIDEOS ANALYZED:\n{v_list}\n\n"
            "INSTRUCTIONS:\n"
            "1. Extract the core points of consensus (where all or most speakers agree).\n"
            "2. Identify specific key disagreements and explain the ROOT CAUSE of divergence (different tax assumptions, scale constraints, time horizons, fee tolerances, risk definitions).\n"
            "3. Synthesize an objective decision playbook (when Creator A's view applies vs Creator B's).\n\n"
            "Format output strictly as JSON:\n"
            "```json\n"
            "{\n"
            '  "points_of_consensus": [\n'
            '    "Tax-loss harvesting value drops substantially after year 5-7 due to portfolio maturation.",\n'
            '    "Direct indexing introduces tracking error against standard market-cap indices."\n'
            "  ],\n"
            '  "disagreements": [\n'
            "    {\n"
            '      "dimension": "Account Size Threshold",\n'
            '      "creator_positions": {\n'
            '        "Creator A": "Viable above $250k-$500k",\n'
            '        "Creator B": "Rarely justifies fee drag below $2M"\n'
            "      },\n"
            '      "root_cause_of_divergence": "Creator A assumes high state income tax brackets, while Creator B assumes low-cost national median tax rates.",\n'
            '      "synthesis_verdict": "Direct indexing is optimal for investors with >$1M and >35% marginal tax rates; low-cost ETFs remain superior for lower brackets."\n'
            "    }\n"
            "  ],\n"
            '  "recommended_playbook": "1. If in top tax bracket with >$500k: consider Direct Indexing. 2. If portfolio is <$500k or tax-advantaged: stick to low-cost broad-market ETFs."\n'
            "}\n"
            "```"
        )

    async def compare_videos(
        self,
        urls: list[str],
        topic: str,
    ) -> CrossVideoComparisonResponse:
        """Compares 2 to 5 videos to construct a cross-creator consensus and disagreement matrix."""
        if len(urls) < 2 or len(urls) > 5:
            raise ToolError(
                ErrorCode.INVALID_INPUT,
                "Please provide between 2 and 5 video URLs to compare.",
            )

        video_metas: list[VideoDetailedMetadata] = []
        video_ids: list[str] = []
        attributions: list[CreatorAttribution] = []

        for u in urls:
            vid = extract_video_id(u)
            if not vid:
                raise ToolError(ErrorCode.INVALID_INPUT, f"Malformed YouTube URL: '{u}'")
            meta = await self.preflight_check(vid)
            video_metas.append(meta)
            video_ids.append(vid)
            attributions.append(
                CreatorAttribution(
                    channel_title=meta.channel_title,
                    channel_id=meta.channel_id,
                    channel_url=f"https://www.youtube.com/channel/{meta.channel_id}",
                    video_url=f"https://www.youtube.com/watch?v={vid}",
                )
            )

        video_titles = [m.title for m in video_metas]
        prompt = self._build_comparison_prompt(topic, video_titles)

        parts: list[Any] = []
        for i, vid in enumerate(video_ids):
            canonical_url = f"https://www.youtube.com/watch?v={vid}"
            parts.append(types.Part.from_text(text=f"--- Video {i+1}: {video_titles[i]} ---"))
            parts.append(types.Part.from_uri(file_uri=canonical_url, mime_type="video/mp4"))
        parts.append(types.Part.from_text(text=prompt))

        contents = [types.Content(role="user", parts=parts)]
        generate_config = types.GenerateContentConfig(
            max_output_tokens=65536,
            temperature=0.1,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        try:
            raw_text, usage, latency, actual_model = await self._generate_with_fallback(
                contents=contents,
                config=generate_config,
                timeout_seconds=360.0,
            )
        except Exception as exc:
            if not isinstance(exc, ToolError):
                logger.error(f"Error comparing videos: {exc}")
            raise

        if not raw_text.strip():
            raise ToolError(ErrorCode.TRANSCRIPTION_FAILED, "Model returned empty response for video comparison.")

        json_obj = _extract_json_block(raw_text)
        consensus: list[str] = []
        disagreements: list[DisagreementDimension] = []
        playbook = "Cross-video comparison completed."

        if json_obj:
            consensus = [str(x) for x in json_obj.get("points_of_consensus", [])]
            playbook = str(json_obj.get("recommended_playbook", playbook))
            for d in json_obj.get("disagreements", []):
                pos_dict = {
                    str(k): str(v) for k, v in d.get("creator_positions", {}).items()
                }
                disagreements.append(
                    DisagreementDimension(
                        dimension=str(d.get("dimension", "Core Topic")),
                        creator_positions=pos_dict,
                        root_cause_of_divergence=str(d.get("root_cause_of_divergence", "")),
                        synthesis_verdict=str(d.get("synthesis_verdict", "")),
                    )
                )

        prompt_tokens, cand_tokens, cost_usd = compute_gemini_cost(usage)

        return CrossVideoComparisonResponse(
            topic=topic,
            video_ids=video_ids,
            points_of_consensus=consensus,
            disagreements=disagreements,
            recommended_playbook=playbook,
            attributions=attributions,
            tokens_input=prompt_tokens,
            tokens_output=cand_tokens,
            cost_usd=cost_usd,
        )


# Global singleton instance
transcription_engine = TranscriptionEngine()
