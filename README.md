# TubeLens OSS

**Search, understand, and cite public YouTube videos** — an AI-powered MCP server for YouTube research.

TubeLens OSS is an open-source [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that helps AI agents analyze public YouTube videos using the official YouTube Data API v3 and Google Gemini multimodal models (`gemini-3.7-flash` with automatic fallback to `gemini-3.6-flash`). It produces AI-generated summaries, key topics, short excerpts with speaker attribution, and timestamped indexes with zero local media storage.

---

## 🌟 Key Features

- ⚡ **100% Live & Stateless** — Every channel catalog, playlist, and video query fetches real-time data directly from YouTube (newly added videos are immediately discoverable).
- ❓ **Ephemeral Deep Video Q&A** — Ask complex questions about any public YouTube video and get direct answers with timestamped evidence.
- 🔍 **Channel & Video Discovery** — Resolve creators, browse catalogs, and enumerate playlists in real time.
- 📊 **AI Video Analysis** — Multi-speaker diarization, summaries, key topics, notable quotes, and timestamp indexes.
- 🏷️ **Creator Attribution** — Mandatory creator attribution in every analysis response.
- 🔒 **Zero Media Downloads & Zero Storage Overhead** — All video processing runs natively via `Part.from_uri()` in Google Cloud with 0 disk persistence.
- 🛡️ **Reliable Model Ladder** — Primary `gemini-3.7-flash` with instant automatic failover to `gemini-3.6-flash` on rate limits or service degradation.

---

## 🏛️ Architecture

```
Your AI Agent (Claude, Cursor, Antigravity, ChatGPT)
         │  (Streamable HTTP / SSE Transport)
         ▼
┌─────────────────────────────────────────────────────────────┐
│  TubeLens OSS (Google Cloud Run / Local Server)             │
│  FastMCP 1.29+ & Starlette Async Engine                     │
├──────────────────────────────┬──────────────────────────────┤
│  YouTube Data API v3         │  Google Gemini / Vertex AI   │
│  (metadata, catalogs, live)  │  (multimodal video reasoning)│
└──────────────────────────────┴──────────────────────────────┘
```

**TubeLens never downloads audio or video.** Public YouTube URLs are passed directly to Google's Gemini API via `Part.from_uri()`, which processes the video entirely on Google's cloud infrastructure in-memory.

---

## 🧠 Intelligence & Research Modes

| Mode | Tool | Purpose | Output |
|---|---|---|---|
| **Decisions & Recommendations** | `youtube_extract_recommendations` | Extract actionable advice & portfolio moves | Stance, conviction, core thesis, risks, alternatives, and timestamped citations |
| **Factual Claims & Verification** | `youtube_extract_claims` | Isolate falsifiable claims vs opinion | Claims by category with independent verification guidance and evidence |
| **Personalized Fit Evaluation** | `youtube_evaluate_fit` | Score relevance for user context | Score (0–100), verdict, matching points, caveats, and custom action items |
| **Cross-Video Comparison** | `youtube_compare_videos` | Synthesize 2–5 videos on a topic | Consensus points, dimensional disagreements with root cause analysis, decision playbook |
| **Deep Video Q&A** | `youtube_ask_video` | Focused questions & answers | Direct answer + timestamped evidence citations (ephemeral) |
| **Analysis Overview** | `youtube_analyze_video` | Analytical summary & key topics | Summary, key topics, short excerpts, timestamp index |

---

## 🛠️ Available MCP Tools (10)

| Tool | Category | Description |
|---|---|---|
| `youtube_resolve_channel` | Discovery | Resolve handle, creator name, or URL to canonical YouTube channel ID |
| `youtube_list_channel_videos` | Catalog | Browse and filter channel video catalog in real-time |
| `youtube_list_playlist_videos` | Discovery | Enumerate videos in a public playlist |
| `youtube_get_videos` | Metadata | Batch fetch detailed metadata for up to 50 video IDs |
| `youtube_analyze_video` | Reasoning | AI summary + excerpts + timestamps (default overview) |
| `youtube_ask_video` | Reasoning | Ask deep questions and get direct answers with timestamped citations |
| `youtube_extract_recommendations` | Intelligence | Extract structured decisions, stance, conviction, risks, alternatives |
| `youtube_extract_claims` | Intelligence | Deconstruct video into verifiable factual claims with verification guidance |
| `youtube_evaluate_fit` | Personalization | Score relevance (0–100) and generate tailored action items for user profile |
| `youtube_compare_videos` | Synthesis | Synthesize 2–5 videos to find consensus and disagreement root causes |

---

## ☁️ Deployment on Google Cloud (Project: `tubelens-oss`)

TubeLens OSS is containerized with a production multi-stage Docker build and deployed as a stateless autoscaling service to **Google Cloud Run**.

### 1. Automated Deployment Script

Run the automated deployment script from the repository root:

```bash
# Set execute permissions
chmod +x scripts/deploy.sh

# Deploy to Google Cloud Run (project: tubelens-oss, region: us-central1)
./scripts/deploy.sh
```

The script automatically:
1. Enables required Google Cloud APIs (`run`, `artifactregistry`, `cloudbuild`, `aiplatform`, `youtube`).
2. Creates the Artifact Registry repository (`mcp-servers`).
3. Builds and pushes the container image via Cloud Build.
4. Deploys to Cloud Run with `gemini-3.7-flash` primary and `gemini-3.6-flash` fallback.
5. Returns the live service URL.

### 2. Manual Cloud Run Deployment

```bash
# Build container with Cloud Build
gcloud builds submit . \
  --project=tubelens-oss \
  --tag=us-central1-docker.pkg.dev/tubelens-oss/mcp-servers/tubelens:latest

# Deploy to Google Cloud Run
gcloud run deploy tubelens-oss \
  --image=us-central1-docker.pkg.dev/tubelens-oss/mcp-servers/tubelens:latest \
  --project=tubelens-oss \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --concurrency=80 \
  --timeout=300 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=tubelens-oss,GOOGLE_CLOUD_LOCATION=global,GEMINI_PRIMARY_MODEL=gemini-3.7-flash,GEMINI_FALLBACK_MODEL=gemini-3.6-flash,AUTH_MODE=none"
```

---

## 🔌 Connecting AI Agents

### 1. Remote SSE Connection (Cloud Run)

Add the deployed SSE endpoint to your MCP client config (Claude Desktop, Cursor, Antigravity, or Cline):

```json
{
  "mcpServers": {
    "tubelens": {
      "url": "https://tubelens-oss-q5hvpgfsaa-uc.a.run.app/sse"
    }
  }
}
```

### 2. Local Process Connection (uv)

```json
{
  "mcpServers": {
    "tubelens": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.server"],
      "cwd": "/path/to/tubelens-oss",
      "env": {
        "GOOGLE_CLOUD_PROJECT": "tubelens-oss",
        "GOOGLE_CLOUD_LOCATION": "global",
        "GEMINI_PRIMARY_MODEL": "gemini-3.7-flash",
        "GEMINI_FALLBACK_MODEL": "gemini-3.6-flash",
        "AUTH_MODE": "none"
      }
    }
  }
}
```

---

## 💻 Local Development & Testing

```bash
# Install dependencies
uv sync

# Run test suite (77 tests, 100% pass)
uv run pytest

# Check code formatting & types
uv run ruff check .
uv run mypy src tests

# Run local MCP server
uv run python -m src.server

# Run end-to-end MCP client test
uv run python scratch/test_mcp_client.py
```

---

## 📜 Legal & Compliance

TubeLens OSS is designed with built-in compliance:
- **YouTube API Terms of Service**: 100% Stateless & Zero-Persistence — no media or cached responses stored on disk.
- **Copyright & Fair Use**: Video analysis produces transformative summaries, fact checks, and timestamped citations (17 U.S.C. § 107).
- **Gemini API**: Prompts and responses processed on Google Cloud enterprise infrastructure.
- **Zero Media Downloads**: Compliant with YouTube Developer Policies §III.E.2.

See [LEGAL.md](LEGAL.md) for full architectural separation of liability details.

---

## 📄 License

Distributed under the [Apache License 2.0](LICENSE).
