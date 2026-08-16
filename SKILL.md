---
name: tubelens-oss
description: >
  YouTube research and intelligence MCP server for AI-powered video Q&A, decision extraction,
  claim verification, personalized fit evaluation, and cross-video disagreement analysis.
---

# TubeLens OSS — MCP Server Skill

## Decision Tree

```
User Query Received
       │
       ▼
What is the primary objective?
├── Extract Decisions & Advice ─────► youtube_extract_recommendations(url, focus_category)
│                                     → Structured portfolios, actions, stance, conviction, risks
├── Verify Factual Claims ──────────► youtube_extract_claims(url, category_filter)
│                                     → Falsifiable claims by category with independent verification guidance
├── Personalized Fit Evaluation ────► youtube_evaluate_fit(url, user_profile, constraints)
│                                     → Relevance score (0-100), matching points, caveats, custom action items
├── Cross-Video Synthesis ──────────► youtube_compare_videos(urls=[url1, url2, ...], topic)
│                                     → Consensus scorecard, disagreement matrix with root causes, playbook
├── Specific / Deep Question ───────► youtube_ask_video(url, query)
│                                     → Direct reasoned answer + timestamped evidence citations
└── General Overview / Index ───────► youtube_analyze_video(url)
                                      → AI analytical summary + short excerpts + timestamp index
```

## Golden Rules

1. **Use `youtube_extract_recommendations` for actionable advice.** Extracts entities, stance, conviction, core thesis, and risks.
2. **Use `youtube_extract_claims` for fact-checking.** Isolates factual claims from opinion with verification guidance.
3. **Use `youtube_evaluate_fit` for personalization.** Scores relevance (0-100) and produces user-tailored action items.
4. **Use `youtube_compare_videos` for multi-video research.** Synthesizes 2-5 videos into consensus and root-cause disagreements.
5. **Use `youtube_ask_video` for targeted Q&A.** Delivers direct answers with timestamped evidence without bloating context.
6. **Use `youtube_analyze_video` for comprehensive overviews.** Produces summaries, key topics, and excerpt indexes.
7. **Always attribute.** Every output includes creator name, channel link, and video URL.
8. **Never download media.** All video processing occurs via `Part.from_uri()` on Google's cloud infrastructure.
9. **100% Live & Stateless.** All queries fetch real-time data from YouTube Data API and Gemini in-memory with zero local disk storage.
10. **Format citations.** Always format citations with inline clickable links: `[Title @ HH:MM:SS](URL&t=SECONDS)`.

## Available Tools (10)

| Tool | Category | Description |
|---|---|---|
| `youtube_extract_recommendations` | Intelligence | Extract structured decisions, stance, conviction, risks, alternatives |
| `youtube_extract_claims` | Intelligence | Deconstruct video into verifiable factual claims with verification guidance |
| `youtube_evaluate_fit` | Personalization | Score relevance (0–100) and generate tailored action items for user profile |
| `youtube_compare_videos` | Synthesis | Synthesize 2–5 videos to find consensus and disagreement root causes |
| `youtube_ask_video` | Reasoning | Deep, specific questions answered with citations (ephemeral) |
| `youtube_analyze_video` | Reasoning | AI summary, excerpts, timestamps (default overview) |
| `youtube_resolve_channel` | Discovery | Resolve handle/name/URL to canonical channel |
| `youtube_list_channel_videos` | Catalog | Browse and filter creator's video catalog |
| `youtube_list_playlist_videos` | Discovery | Enumerate videos in a public playlist |
| `youtube_get_videos` | Metadata | Batch metadata for up to 50 video IDs |

## Example Workflows

### Extract decisions & actionable portfolio advice
```
User: "What specific emerging market ETFs does the speaker recommend and why?"
→ youtube_extract_recommendations(url="https://youtube.com/watch?v=...", focus_category="ETFs")
→ Structured RecommendationItems with conviction, risks, and timestamped citations
```

### Personalize a video's advice to user profile
```
User: "I'm a software engineer earning $400k in Washington state. Does this video's direct indexing advice apply to me?"
→ youtube_evaluate_fit(url="https://youtube.com/watch?v=...", user_profile="Software engineer $400k income", constraints=["Zero state income tax"])
→ Relevance: 92/100, must_watch, matching points, caveats (WA 0% state tax), tailored action items
```

### Compare 2-3 videos on the same topic
```
User: "Compare Ben Felix and Rob Berger's takes on Direct Indexing."
→ youtube_compare_videos(urls=[url1, url2], topic="Direct Indexing vs ETFs")
→ Consensus points + Disagreement dimensions with identified root causes (fee assumptions) + Playbook
```
