"""
modules/confidence.py
=====================
Confidence scoring and status assignment engine for the Company Data
Enrichment System.

Every enriched company record receives:
    - confidence_score : int  (0–100)
    - status           : str  one of FOUND | PARTIALLY_FOUND | NOT_FOUND | NEEDS_REVIEW

Scoring philosophy
------------------
The score reflects HOW CERTAIN we are that the data is correct, based on:
    1. The quality / authority of the source that supplied the data
    2. The completeness of the critical fields in the record
    3. Whether the LLM or a scraper contributed (lower trust than verified sources)

Score bands (mirrors the spec):
    95–100  Official company website (direct scrape, self-reported data)
    90–94   Verified LinkedIn page
    80–89   Reliable public source (Clearbit, Wikipedia, etc.)
    60–79   LLM-inferred from strong contextual clues
    30–59   Weak evidence, mostly inferred or low-quality scrape
    0–29    Not found / no usable source

Status rules:
    FOUND           score >= 80  AND  all critical fields present
    PARTIALLY_FOUND score >= 30  AND  at least one critical field present
    NOT_FOUND       score < 30   OR   no critical fields present at all
    NEEDS_REVIEW    score < threshold set in settings  OR  flag conditions met
                    (e.g. company/domain mismatch, suspiciously short description)

Public API:
    scorer = ConfidenceScorer()
    score, status = scorer.score(record)       # score a single dict
    df = scorer.score_dataframe(df)            # score an entire DataFrame
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnrichmentStatus(str, Enum):
    """Possible pipeline statuses for a company record."""
    FOUND           = "FOUND"
    PARTIALLY_FOUND = "PARTIALLY_FOUND"
    NOT_FOUND       = "NOT_FOUND"
    NEEDS_REVIEW    = "NEEDS_REVIEW"


class DataSource(str, Enum):
    """
    Known data sources that contribute to a record.
    The base scores are starting points; field completeness adjusts them.
    """
    OFFICIAL_WEBSITE = "official_website"    # 95–100
    LINKEDIN         = "linkedin"            # 90
    PUBLIC_DATABASE  = "public_database"     # 80
    LLM_INFERRED     = "llm_inferred"        # 60
    WEB_SCRAPE       = "web_scrape"          # 50
    WEAK_SIGNAL      = "weak_signal"         # 30
    NOT_FOUND        = "not_found"           # 0


# ---------------------------------------------------------------------------
# Score tables
# ---------------------------------------------------------------------------

# Base score awarded per source type (before completeness adjustments)
_SOURCE_BASE_SCORE: dict[str, int] = {
    DataSource.OFFICIAL_WEBSITE: 95,
    DataSource.LINKEDIN:         80,
    DataSource.PUBLIC_DATABASE:  80,
    DataSource.LLM_INFERRED:     60,
    DataSource.WEB_SCRAPE:       60,
    DataSource.WEAK_SIGNAL:      30,
    DataSource.NOT_FOUND:         0,
}

# Critical fields — record is only FOUND if ALL of these are present
_CRITICAL_FIELDS: list[str] = ["name", "website", "description", "industry"]

# Important fields — contribute to completeness bonus
_IMPORTANT_FIELDS: list[str] = ["ceo", "founder", "size", "email", "phone"]

# Social fields — small bonus for each present
_SOCIAL_FIELDS: list[str] = ["linkedin", "facebook", "twitter", "instagram"]

# Max bonus points awarded for field completeness (on top of base score)
_MAX_COMPLETENESS_BONUS: int = 5

# Penalty applied when a flag condition is triggered (per flag)
_FLAG_PENALTY: int = 10

# Score below which a record is auto-flagged as NEEDS_REVIEW
_NEEDS_REVIEW_THRESHOLD: int = settings.min_confidence_threshold


# ---------------------------------------------------------------------------
# Flag conditions
# ---------------------------------------------------------------------------

class ReviewFlag(str, Enum):
    """Human-readable flags attached to records that need manual review."""
    SCORE_TOO_LOW           = "score_below_threshold"
    MISSING_CRITICAL_FIELD  = "missing_critical_field"
    NAME_DOMAIN_MISMATCH    = "name_domain_mismatch"
    DESCRIPTION_TOO_SHORT   = "description_too_short"
    SUSPICIOUS_CEO          = "suspicious_ceo_value"
    SUSPICIOUS_FOUNDER      = "suspicious_founder_value"
    DUPLICATE_DETECTED      = "duplicate_detected"
    STALE_DATA              = "stale_data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _present(value) -> bool:
    """True if the value is a non-empty, non-NA string."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    s = str(value).strip()
    return bool(s) and s.lower() not in {"nan", "none", "null", "n/a", "na", ""}


def _count_present(record: dict, fields: list[str]) -> int:
    """Count how many of the given fields are populated."""
    return sum(1 for f in fields if _present(record.get(f)))


def _infer_source(record: dict) -> DataSource:
    """
    Infer the best data source from the record's ``source_used`` field
    or from which fields are populated.

    ``source_used`` is a free-text tag set by the enrichment module.
    """
    source_used: str = str(record.get("source_used", "") or "").lower()

    if "official" in source_used or "website" in source_used:
        return DataSource.OFFICIAL_WEBSITE
    if "linkedin" in source_used:
        return DataSource.LINKEDIN
    if any(tag in source_used for tag in ("clearbit", "wikipedia", "crunchbase", "database")):
        return DataSource.PUBLIC_DATABASE
    if "llm" in source_used or "gpt" in source_used or "openai" in source_used:
        return DataSource.LLM_INFERRED
    if "scrape" in source_used or "scraper" in source_used:
        return DataSource.WEB_SCRAPE
    if _present(record.get("website")) or _present(record.get("description")):
        # We got *something* but source isn't labelled — treat as web scrape
        return DataSource.WEB_SCRAPE
    return DataSource.NOT_FOUND


def _completeness_bonus(record: dict) -> int:
    """
    Calculate a small completeness bonus (0 to _MAX_COMPLETENESS_BONUS).

    Rewards records that have many important / social fields populated.
    """
    important_filled = _count_present(record, _IMPORTANT_FIELDS)
    social_filled    = _count_present(record, _SOCIAL_FIELDS)

    total_possible = len(_IMPORTANT_FIELDS) + len(_SOCIAL_FIELDS)
    total_filled   = important_filled + social_filled

    ratio = total_filled / total_possible if total_possible else 0
    return round(ratio * _MAX_COMPLETENESS_BONUS)


def _detect_flags(record: dict, raw_score: int) -> list[ReviewFlag]:
    """
    Detect conditions that warrant manual review.

    Returns a list of triggered ReviewFlags.
    """
    flags: list[ReviewFlag] = []

    # Score too low
    if raw_score < _NEEDS_REVIEW_THRESHOLD:
        flags.append(ReviewFlag.SCORE_TOO_LOW)

    # Missing critical fields
    for field in _CRITICAL_FIELDS:
        if not _present(record.get(field)):
            flags.append(ReviewFlag.MISSING_CRITICAL_FIELD)
            break   # one flag is enough for this category

    # Description too short (likely a scrape artefact, not real content)
    desc = str(record.get("description", "") or "").strip()
    if desc and len(desc) < 30:
        flags.append(ReviewFlag.DESCRIPTION_TOO_SHORT)

    # Suspicious CEO / Founder values (numbers, URLs, single chars)
    for field_name, flag in (
        ("ceo",     ReviewFlag.SUSPICIOUS_CEO),
        ("founder", ReviewFlag.SUSPICIOUS_FOUNDER),
    ):
        val = str(record.get(field_name, "") or "").strip()
        if val:
            # Red flags: contains digits only, looks like a URL, or single char
            if re.fullmatch(r"[\d\s\-+().]+", val):
                flags.append(flag)
            elif "http" in val.lower() or "@" in val:
                flags.append(flag)
            elif len(val) < 3:
                flags.append(flag)

    # Name / domain mismatch heuristic
    name  = str(record.get("name", "") or "").lower()
    domain = str(record.get("website", "") or "").lower()
    if name and domain:
        # Extract first meaningful word of company name
        name_word = re.sub(r"[^a-z0-9]", "", name.split()[0]) if name.split() else ""
        domain_root = domain.split(".")[0] if "." in domain else domain
        domain_root = re.sub(r"[^a-z0-9]", "", domain_root)
        # Only flag if name word is >= 4 chars and not present in domain at all
        if (
            len(name_word) >= 4
            and name_word not in domain_root
            and domain_root not in name_word
        ):
            flags.append(ReviewFlag.NAME_DOMAIN_MISMATCH)

    return flags


def _determine_status(
    record: dict,
    final_score: int,
    flags: list[ReviewFlag],
) -> EnrichmentStatus:
    """
    Map a score + flag list to a human-readable EnrichmentStatus.

    Decision tree:
        1. If any NEEDS_REVIEW trigger flag present → NEEDS_REVIEW
        2. If score >= 80 and all critical fields present → FOUND
        3. If score >= 30 and at least one critical field present → PARTIALLY_FOUND
        4. Otherwise → NOT_FOUND
    """
    needs_review_flags = {
        ReviewFlag.SCORE_TOO_LOW,
        ReviewFlag.NAME_DOMAIN_MISMATCH,
        ReviewFlag.SUSPICIOUS_CEO,
        ReviewFlag.SUSPICIOUS_FOUNDER,
        ReviewFlag.DESCRIPTION_TOO_SHORT,
    }

    if any(f in needs_review_flags for f in flags):
        return EnrichmentStatus.NEEDS_REVIEW

    critical_present = _count_present(record, _CRITICAL_FIELDS)
    all_critical = critical_present == len(_CRITICAL_FIELDS)

    if final_score >= 80 and all_critical:
        return EnrichmentStatus.FOUND

    if final_score >= 60 and critical_present >= 1:
        return EnrichmentStatus.PARTIALLY_FOUND

    return EnrichmentStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """
    Assigns a confidence score and status to each company record.

    Usage:
        scorer = ConfidenceScorer()

        # Single record
        score, status, flags = scorer.score(record_dict)

        # Full DataFrame
        enriched_df = scorer.score_dataframe(df)
    """

    def score(
        self,
        record: dict,
    ) -> tuple[int, EnrichmentStatus, list[ReviewFlag]]:
        """
        Score a single company record.

        Parameters
        ----------
        record : dict
            A company record (keys matching the standard schema).

        Returns
        -------
        tuple[int, EnrichmentStatus, list[ReviewFlag]]
            (confidence_score, status, flags)
        """
        # 1. Determine source and base score
        source   = _infer_source(record)
        base     = _SOURCE_BASE_SCORE[source]

        # 2. Add completeness bonus
        bonus    = _completeness_bonus(record)

        # 3. Penalise if critical fields are missing
        missing_critical = len(_CRITICAL_FIELDS) - _count_present(record, _CRITICAL_FIELDS)
        penalty = missing_critical * 8   # -8 per missing critical field

        # 4. Raw score (clamped 0–100)
        raw_score = max(0, min(100, base + bonus - penalty))

        # 5. Detect review flags and apply additional penalties
        flags = _detect_flags(record, raw_score)
        flag_penalty = len(flags) * _FLAG_PENALTY
        final_score = max(0, min(100, raw_score - flag_penalty))

        # 6. Determine status
        status = _determine_status(record, final_score, flags)

        logger.debug(
            "Scored '%s': source=%s base=%d bonus=%d penalty=%d "
            "flag_penalty=%d final=%d status=%s flags=%s",
            record.get("name", "?"),
            source.value, base, bonus, penalty,
            flag_penalty, final_score, status.value,
            [f.value for f in flags],
        )

        return final_score, status, flags

    # ------------------------------------------------------------------
    # DataFrame-level scoring
    # ------------------------------------------------------------------

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scoring to every row in a DataFrame.

        Adds / overwrites columns:
            confidence_score  (int)
            status            (str)
            _review_flags     (str, comma-separated flag names)

        Returns the modified DataFrame (does NOT mutate the original).
        """
        df = df.copy()
        scores: list[int]              = []
        statuses: list[str]            = []
        flag_lists: list[str]          = []

        for _, row in df.iterrows():
            record = row.to_dict()
            score, status, flags = self.score(record)
            scores.append(score)
            statuses.append(status.value)
            flag_lists.append(", ".join(f.value for f in flags) if flags else "")

        df["confidence_score"] = scores
        df["status"]           = statuses
        df["_review_flags"]    = flag_lists

        # Summary log
        found          = statuses.count(EnrichmentStatus.FOUND.value)
        partially      = statuses.count(EnrichmentStatus.PARTIALLY_FOUND.value)
        not_found      = statuses.count(EnrichmentStatus.NOT_FOUND.value)
        needs_review   = statuses.count(EnrichmentStatus.NEEDS_REVIEW.value)
        avg_score      = sum(scores) / len(scores) if scores else 0

        logger.info(
            "Scoring complete — FOUND: %d | PARTIALLY_FOUND: %d | "
            "NOT_FOUND: %d | NEEDS_REVIEW: %d | avg_score: %.1f",
            found, partially, not_found, needs_review, avg_score,
        )

        return df

    # ------------------------------------------------------------------
    # Convenience class-methods for direct score lookups
    # ------------------------------------------------------------------

    @staticmethod
    def score_for_source(source: DataSource) -> int:
        """Return the base score for a given DataSource enum value."""
        return _SOURCE_BASE_SCORE.get(source, 0)

    @staticmethod
    def status_label(score: int, all_critical_present: bool) -> EnrichmentStatus:
        """
        Quick status lookup without flag evaluation — useful for unit tests
        and simple heuristics during enrichment.
        """
        if score >= 80 and all_critical_present:
            return EnrichmentStatus.FOUND
        if score >= 60:
            return EnrichmentStatus.PARTIALLY_FOUND
        return EnrichmentStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def score_record(record: dict) -> tuple[int, str, list[str]]:
    """
    Module-level shortcut to score a single dict without instantiating
    ConfidenceScorer explicitly.

    Returns (confidence_score, status_str, flag_names_list)
    """
    scorer = ConfidenceScorer()
    score, status, flags = scorer.score(record)
    return score, status.value, [f.value for f in flags]
