"""
Configuration and Environment Management for TubeLens OSS
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google Cloud Project & Location
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    gemini_primary_model: str = "gemini-3.7-flash"
    gemini_fallback_model: str = "gemini-3.6-flash"
    gemini_fallback_model_2: str = ""

    # YouTube Data API Key (optional - used for metadata and handle searches)
    youtube_api_key: str | None = None

    # MCP Server & Network Settings
    port: int = 8080
    host: str = "0.0.0.0"
    auth_mode: str = "none"  # Options: none (local open access), api_key (enforces X-MCP-API-Key or Bearer token)
    mcp_api_key: str | None = None

    # OAuth 2.0 Settings (RFC 6749, RFC 7591, RFC 8414)
    oauth_jwt_secret: str = ""
    oauth_issuer_url: str = ""
    oauth_token_ttl_seconds: int = 2592000  # 30 days
    oauth_registration_token: str = ""  # When set, /oauth/register requires Bearer <token> auth
    oauth_admin_password: str = ""  # When set, /oauth/authorize consent page requires this password

    # Research & Duration Limits
    max_catalog_videos: int = 500
    max_video_duration_seconds: int = 14400  # 4 hours max

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://mbettan.github.io",
    ]

    # Cost Estimation Defaults (USD per million tokens)
    prompt_token_cost_per_million: float = 0.15
    candidate_token_cost_per_million: float = 0.60

    # Analysis mode defaults
    max_excerpt_words: int = 50          # ~2 sentences per excerpt
    max_excerpts_per_analysis: int = 10  # Notable quotes cap
    analysis_summary_words: int = 800    # Target summary length

    @property
    def model_ladder(self) -> list[str]:
        """Primary model followed by fallback models, de-duplicated."""
        ladder = [
            self.gemini_primary_model,
            self.gemini_fallback_model,
            self.gemini_fallback_model_2,
        ]
        seen: set[str] = set()
        out: list[str] = []
        for m in ladder:
            m = m.strip()
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        if not out:
            raise ValueError("At least one Gemini model must be configured in settings.")
        return out


# Global singleton settings instance
Settings.model_rebuild()
settings = Settings()
