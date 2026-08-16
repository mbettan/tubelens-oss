# Legal Notice & User Responsibilities

TubeLens OSS is open-source software distributed under the Apache License 2.0.
It is a code library and tool — **not a hosted service**.

## Architectural Separation of Liability

```
[TubeLens OSS Codebase]  →  Presents tools & prompt logic (Apache 2.0)
[End User / AI Agent]     →  Provides API keys & chooses execution parameters
[Google Gemini / Vertex]  →  Processes public YouTube URL via cloud infrastructure
```

The open-source maintainer publishes code. The self-hosting user executes it with
their own Google Cloud credentials and API keys.

## Your Responsibilities as a Self-Hoster

When you run TubeLens, **YOU** are responsible for:

1. **YouTube Terms of Service** — Complying with the [YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service) and the [YouTube Developer Policies](https://developers.google.com/youtube/terms/developer-policies).

2. **Gemini API Terms** — Complying with the [Gemini API Terms of Service](https://ai.google.dev/gemini-api/terms). You must use a **paid** Vertex AI or Gemini API key to ensure your prompts and responses are not used to train Google models.

3. **Copyright Law** — Ensuring your use of video content respects applicable copyright law, including 17 U.S.C. § 107 (Fair Use) and the EU Copyright Directive (Articles 3 & 4 for Text and Data Mining).

4. **Privacy Regulations** — Implementing appropriate privacy controls if your deployment processes personally identifiable information, including compliance with GDPR, CCPA, and applicable local regulations.

5. **API Credentials** — Providing your own YouTube Data API key and Gemini / Vertex AI API credentials. TubeLens does not bundle, distribute, or share API keys.

6. **100% Stateless Execution** — TubeLens does not store or persist YouTube metadata or AI outputs on disk. All queries are live.

7. **Creator Attribution** — Maintaining proper attribution to content creators and linking back to original YouTube content per [YouTube Branding Guidelines](https://developers.google.com/youtube/terms/branding-guidelines).

## What TubeLens Does NOT Do

- ❌ Download, cache, or store YouTube audio or video files
- ❌ Use `yt-dlp`, `youtube-dl`, or any stream-ripping tool
- ❌ Bypass YouTube access controls or DRM
- ❌ Provide unauthorized access to private or unlisted content
- ❌ Host or redistribute copyrighted content
- ❌ Scrape YouTube web pages or internal APIs

## What TubeLens DOES Do

- ✅ Passes public YouTube URLs to Google's Gemini API via `Part.from_uri()`
- ✅ Video is processed entirely on Google's cloud infrastructure in-memory
- ✅ Retrieves fresh metadata live via the official YouTube Data API v3
- ✅ 100% Stateless with zero disk caching overhead
- ✅ Includes mandatory creator attribution in every analysis output

## No Legal Advice

This project and its documentation do not constitute legal advice. If you plan
to deploy TubeLens in a commercial context, consult with an IP/technology attorney
to review your specific use case.
