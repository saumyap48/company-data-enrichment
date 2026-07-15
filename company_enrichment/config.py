"""
config.py
=========
Centralised configuration for the Company Data Enrichment System.

All runtime values are loaded from environment variables (via a .env file).
No secrets or keys are hard-coded here.

Usage:
    from config import settings
    print(settings.openai_api_key)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Resolve project root so that relative paths always work regardless of where
# the script is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.

    Priority order (highest → lowest):
        1. Real environment variables already set in the shell
        2. Values in the .env file
        3. Default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,       # OPENAI_API_KEY == openai_api_key
        extra="ignore",             # Silently ignore unknown env vars
    )

    # ------------------------------------------------------------------
    # LLM / OpenAI
    # ------------------------------------------------------------------
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (sk-...).",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model used for structured extraction.",
    )
    openai_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = deterministic output.",
    )
    openai_max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Maximum tokens in the LLM response.",
    )

    # ------------------------------------------------------------------
    # File Paths
    # ------------------------------------------------------------------
    input_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "input",
        description="Directory containing raw input CSV files.",
    )
    output_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "output",
        description="Directory where enriched output files are written.",
    )
    log_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="Directory for log files.",
    )
    log_file: Path = Field(
        default=PROJECT_ROOT / "logs" / "app.log",
        description="Full path to the main application log file.",
    )
    cache_dir: Path = Field(
        default=PROJECT_ROOT / ".cache",
        description="Persistent disk cache directory (domain lookups).",
    )
    prompts_dir: Path = Field(
        default=PROJECT_ROOT / "prompts",
        description="Directory containing LLM prompt template files.",
    )
    enriched_output_file: Path = Field(
        default=PROJECT_ROOT / "data" / "output" / "enriched_companies.csv",
        description="Path to the final enriched CSV.",
    )
    validation_report_file: Path = Field(
        default=PROJECT_ROOT / "data" / "output" / "validation_report.json",
        description="Path to the JSON validation report.",
    )

    # ------------------------------------------------------------------
    # HTTP / Scraping
    # ------------------------------------------------------------------
    request_timeout: int = Field(
        default=15,
        ge=1,
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of HTTP retry attempts.",
    )
    retry_backoff_factor: float = Field(
        default=1.5,
        ge=0.0,
        description="Exponential back-off multiplier between retries.",
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Whether to honour robots.txt during scraping.",
    )
    scraper_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (compatible; CompanyEnrichmentBot/1.0; "
            "+https://example.com/bot)"
        ),
        description="User-Agent header sent with every HTTP request.",
    )

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------
    requests_per_second: float = Field(
        default=2.0,
        ge=0.1,
        description="Maximum outbound HTTP requests per second (global).",
    )
    llm_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum OpenAI API calls per minute.",
    )

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------
    max_workers: int = Field(
        default=10,
        ge=1,
        description="ThreadPoolExecutor max worker threads for enrichment.",
    )
    batch_size: int = Field(
        default=50,
        ge=1,
        description="Number of companies to process per batch.",
    )

    # ------------------------------------------------------------------
    # Enrichment Behaviour
    # ------------------------------------------------------------------
    min_confidence_threshold: int = Field(
        default=30,
        ge=0,
        le=100,
        description=(
            "Records with a confidence score below this threshold are "
            "flagged as NEEDS_REVIEW."
        ),
    )
    skip_already_enriched: bool = Field(
        default=True,
        description=(
            "If True, skip rows whose status is already FOUND so the "
            "pipeline can resume without re-processing."
        ),
    )
    max_text_chars_for_llm: int = Field(
        default=8000,
        ge=500,
        description=(
            "Maximum number of characters of scraped website text to "
            "send to the LLM (to manage token costs)."
        ),
    )

    # ------------------------------------------------------------------
    # Fuzzy Deduplication
    # ------------------------------------------------------------------
    fuzzy_name_threshold: int = Field(
        default=90,
        ge=0,
        le=100,
        description=(
            "RapidFuzz score (0-100) above which two company names are "
            "considered duplicates."
        ),
    )

    # ------------------------------------------------------------------
    # Optional Third-Party API Keys
    # (reserved for future enrichment sources — leave blank to skip)
    # ------------------------------------------------------------------
    clearbit_api_key: Optional[str] = Field(
        default=None,
        description="Clearbit enrichment API key (optional).",
    )
    hunter_api_key: Optional[str] = Field(
        default=None,
        description="Hunter.io email finder API key (optional).",
    )
    serpapi_key: Optional[str] = Field(
        default=None,
        description="SerpAPI key for web search (optional).",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("input_dir", "output_dir", "log_dir", "cache_dir", "prompts_dir", mode="after")
    @classmethod
    def ensure_directory_exists(cls, v: Path) -> Path:
        """Auto-create required directories on startup."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("openai_api_key", mode="after")
    @classmethod
    def warn_if_key_missing(cls, v: str) -> str:
        """Emit a warning (not an error) if the OpenAI key is absent."""
        if not v:
            import warnings
            warnings.warn(
                "OPENAI_API_KEY is not set. LLM extraction will be disabled. "
                "Set it in your .env file to enable AI enrichment.",
                stacklevel=2,
            )
        return v

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        """True if an OpenAI API key has been provided."""
        return bool(self.openai_api_key)

    def as_dict(self) -> dict:
        """Return a serialisable snapshot (masks the API key)."""
        data = self.model_dump()
        # Mask sensitive fields
        for sensitive in ("openai_api_key", "clearbit_api_key", "hunter_api_key", "serpapi_key"):
            if data.get(sensitive):
                data[sensitive] = "***REDACTED***"
        return data


# ---------------------------------------------------------------------------
# Singleton — import `settings` everywhere in the project.
# ---------------------------------------------------------------------------
settings: Settings = Settings()
