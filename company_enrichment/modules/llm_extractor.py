"""
modules/llm_extractor.py
========================
GenAI extraction layer for the Company Data Enrichment System.

Uses OpenAI's structured output (JSON mode + Pydantic) to extract
company facts from scraped website text.

Key design choices:
    - Pydantic models enforce the output schema and types — no raw JSON parsing
    - Token budget guard — text is truncated before sending to avoid cost spikes
    - System prompt loaded from prompts/extraction_prompt.txt
    - Exponential-backoff retry on rate-limit / transient API errors
    - Fully disabled (returns empty result) when OPENAI_API_KEY is not set
    - Thread-safe — one shared OpenAI client (the SDK is thread-safe)

Public API:
    extractor = LLMExtractor()
    result    = extractor.extract(website_text, company_name="Acme Corp")
    print(result.description, result.ceo, result.confidence_score)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import tiktoken
except Exception as e:
    logger.warning("tiktoken import failed: %s. Falling back to simple token count.", e)
    tiktoken = None

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the prompt template (relative to project root)
_PROMPT_FILE: Path = settings.prompts_dir / "extraction_prompt.txt"

# Hard token limits to stay well within model context windows
_MAX_PROMPT_TOKENS = 6_000        # tokens reserved for the prompt
_MAX_RESPONSE_TOKENS = settings.openai_max_tokens

# Company-size bands — must match the prompt instruction exactly
_VALID_SIZE_BANDS: set[str] = {
    "1-10", "11-50", "51-200", "201-500",
    "501-1000", "1001-5000", "5001-10000", "10000+",
}


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class CompanyExtraction(BaseModel):
    """
    Structured data extracted from website text by the LLM.

    Every field is optional — empty string means "not found".
    confidence_score ranges 0–100 and is set by the model itself.
    """

    description: str = Field(
        default="",
        description="1–3 sentence factual description of what the company does.",
    )
    industry: str = Field(
        default="",
        description="Primary industry or sector (e.g. 'Enterprise Software', 'FinTech').",
    )
    company_size: str = Field(
        default="",
        description="Employee count band: '1-10', '11-50', …, '10000+', or ''.",
    )
    ceo: str = Field(
        default="",
        description="Full name of the current CEO / Managing Director.",
    )
    founder: str = Field(
        default="",
        description=(
            "Full name(s) of the founder(s). "
            "Multiple founders separated by commas."
        ),
    )
    confidence_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="0–100 score reflecting evidence quality (set by the model).",
    )

    # ------------------------------------------------------------------
    # Validators — clean up model output before it reaches the pipeline
    # ------------------------------------------------------------------

    @field_validator("company_size", mode="before")
    @classmethod
    def validate_size_band(cls, v: str) -> str:
        """Accept only valid size bands; blank anything else."""
        if not v:
            return ""
        v = str(v).strip()
        return v if v in _VALID_SIZE_BANDS else ""

    @field_validator("description", "industry", "ceo", "founder", mode="before")
    @classmethod
    def clean_string(cls, v) -> str:
        """Strip whitespace; coerce None → ''."""
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("confidence_score", mode="before")
    @classmethod
    def coerce_score(cls, v) -> int:
        """Coerce string or float to int, clamp to [0, 100]."""
        try:
            score = int(float(str(v)))
        except (ValueError, TypeError):
            return 0
        return max(0, min(100, score))

    def is_empty(self) -> bool:
        """True if the LLM found nothing useful."""
        return not any([
            self.description, self.industry, self.company_size,
            self.ceo, self.founder,
        ])

    def to_dict(self) -> dict:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

class _PromptLoader:
    """Loads and caches the prompt template from disk."""

    _template: Optional[str] = None

    @classmethod
    def get(cls) -> str:
        if cls._template is None:
            try:
                cls._template = _PROMPT_FILE.read_text(encoding="utf-8")
                logger.debug("Loaded prompt template from '%s'.", _PROMPT_FILE)
            except FileNotFoundError:
                logger.error(
                    "Prompt file not found at '%s'. Using built-in fallback.", _PROMPT_FILE
                )
                cls._template = _BUILTIN_FALLBACK_PROMPT
        return cls._template

    @classmethod
    def render(cls, website_text: str, company_name: str = "") -> str:
        """Fill the template placeholders."""
        template = cls.get()
        return template.replace("{website_text}", website_text) \
                       .replace("{company_name}", company_name)


# Built-in fallback in case the prompts/ file is missing
_BUILTIN_FALLBACK_PROMPT = """Extract company information from the text below.
Return ONLY valid JSON with keys: description, industry, company_size, ceo, founder, confidence_score.
NEVER hallucinate. If a field is not found in the text, use an empty string. DO NOT guess.
confidence_score is an integer 0-100 based on evidence quality.

WEBSITE TEXT:
{website_text}

COMPANY NAME: {company_name}
"""


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------

def _count_tokens(text: str, model: str) -> int:
    """Return approximate token count for ``text`` using tiktoken, fallback to simple whitespace count."""
    if tiktoken is None:
        # Simple whitespace token count as fallback
        return len(text.split())
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _truncate_to_token_budget(text: str, max_tokens: int, model: str) -> str:
    """Truncate ``text`` to fit within ``max_tokens`` using tiktoken if available, otherwise fallback."""
    if tiktoken is None:
        # Approximate truncation by word count
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = enc.decode(tokens[:max_tokens])
    return truncated


# ---------------------------------------------------------------------------
# Retry decorator — handles RateLimitError and transient API failures
# ---------------------------------------------------------------------------

_API_RETRY = retry(
    reraise=True,
    stop=stop_after_attempt(settings.max_retries + 1),
    wait=wait_exponential(multiplier=settings.retry_backoff_factor, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    before_sleep=lambda rs: logger.warning(
        "OpenAI API retry %d/%d — waiting before next attempt…",
        rs.attempt_number, settings.max_retries + 1,
    ),
)


# ---------------------------------------------------------------------------
# LLMExtractor
# ---------------------------------------------------------------------------

class LLMExtractor:
    """
    Extracts structured company data from website text using an LLM.

    Thread-safe: the OpenAI client is safe to share across threads.

    Parameters
    ----------
    model : str, optional
        OpenAI model name. Defaults to settings.openai_model.
    temperature : float, optional
        Sampling temperature. Defaults to settings.openai_temperature.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self._model: str = model or settings.openai_model
        self._temperature: float = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self._enabled: bool = settings.llm_enabled
        self._client: Optional[OpenAI] = None

        if self._enabled:
            self._client = OpenAI(api_key=settings.openai_api_key)
            logger.info(
                "LLMExtractor initialised — model: %s, temperature: %.1f",
                self._model, self._temperature,
            )
        else:
            logger.warning(
                "LLMExtractor: OPENAI_API_KEY not set — "
                "LLM extraction is DISABLED. Set the key in .env to enable it."
            )

    # ------------------------------------------------------------------
    # Public extract method
    # ------------------------------------------------------------------

    def extract(
        self,
        website_text: str,
        company_name: str = "",
        about_text: str = "",
        contact_text: str = "",
        team_text: str = "",
    ) -> CompanyExtraction:
        """
        Extract company information from website text.

        Parameters
        ----------
        website_text : str
            Primary page text (homepage visible text).
        company_name : str, optional
            Company name — passed as context only (never used to invent data).
        about_text : str, optional
            Text from the /about page (appended to improve extraction).
        contact_text : str, optional
            Text from the /contact page.
        team_text : str, optional
            Text from the /leadership or /team page.

        Returns
        -------
        CompanyExtraction
            Pydantic model with extracted fields.
            All fields are empty strings / 0 if the LLM is disabled or
            extraction fails.
        """
        if not self._enabled or self._client is None:
            return CompanyExtraction()

        # -- Combine text sources, prioritise About page and Team page --
        combined_text = "\n\n".join(
            filter(None, [about_text, team_text, website_text, contact_text])
        )

        if not combined_text.strip():
            logger.debug("No text to extract from for '%s'.", company_name)
            return CompanyExtraction()

        # -- Truncate to token budget --
        truncated = _truncate_to_token_budget(
            combined_text,
            max_tokens=_MAX_PROMPT_TOKENS,
            model=self._model,
        )

        # -- Render prompt --
        prompt = _PromptLoader.render(
            website_text=truncated,
            company_name=company_name,
        )

        logger.debug(
            "Sending %d chars to LLM for '%s'.", len(truncated), company_name
        )

        # -- Call OpenAI --
        try:
            result = self._call_api(prompt)
        except Exception as exc:
            logger.error(
                "LLM extraction failed for '%s': %s", company_name, exc, exc_info=True
            )
            return CompanyExtraction()

        logger.info(
            "LLM extracted for '%s' — industry: '%s', ceo: '%s', confidence: %d",
            company_name, result.industry, result.ceo, result.confidence_score,
        )
        return result

    # ------------------------------------------------------------------
    # OpenAI API call (with retry)
    # ------------------------------------------------------------------

    @_API_RETRY
    def _call_api(self, prompt: str) -> CompanyExtraction:
        """
        Call the OpenAI API with JSON mode and parse the Pydantic model.

        Uses the `response_format={"type": "json_object"}` parameter
        to guarantee valid JSON output regardless of the model version.
        """
        assert self._client is not None  # guarded by __init__

        start = time.monotonic()

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=_MAX_RESPONSE_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise business analyst that extracts "
                        "ONLY verifiable facts from text. "
                        "You return ONLY valid JSON. "
                        "CRITICAL: NEVER hallucinate. If information cannot be verified from the text, leave the field blank. DO NOT guess."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        elapsed = time.monotonic() - start
        logger.debug("OpenAI API call took %.2fs.", elapsed)

        # Extract the JSON string from the response
        raw_json: str = response.choices[0].message.content or "{}"

        # Parse with Pydantic (raises ValidationError on bad output)
        try:
            return CompanyExtraction.model_validate_json(raw_json)
        except Exception as parse_exc:
            logger.warning(
                "Failed to parse LLM JSON output: %s\nRaw: %s",
                parse_exc, raw_json[:300],
            )
            # Attempt manual JSON parse as fallback
            return self._parse_fallback(raw_json)

    # ------------------------------------------------------------------
    # Fallback parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fallback(raw_json: str) -> CompanyExtraction:
        """
        Attempt a lenient parse when Pydantic model_validate_json fails.
        Extracts only the fields we know about and ignores anything extra.
        """
        import json
        try:
            data = json.loads(raw_json)
            return CompanyExtraction(
                description=data.get("description", ""),
                industry=data.get("industry", ""),
                company_size=data.get("company_size", ""),
                ceo=data.get("ceo", ""),
                founder=data.get("founder", ""),
                confidence_score=int(data.get("confidence_score", 0)),
            )
        except Exception:
            return CompanyExtraction()

    # ------------------------------------------------------------------
    # Batch extraction convenience method
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        records: list[dict],
        delay_between: float = 1.0,
    ) -> list[CompanyExtraction]:
        """
        Extract from a list of dicts, each having keys:
            website_text, company_name, about_text, contact_text, team_text (all optional)

        Applies a small delay between calls to respect rate limits.
        """
        results = []
        for i, record in enumerate(records):
            logger.debug("Batch extraction %d/%d…", i + 1, len(records))
            result = self.extract(
                website_text=record.get("website_text", ""),
                company_name=record.get("company_name", ""),
                about_text=record.get("about_text", ""),
                contact_text=record.get("contact_text", ""),
                team_text=record.get("team_text", ""),
            )
            results.append(result)
            if i < len(records) - 1:
                time.sleep(delay_between)
        return results
