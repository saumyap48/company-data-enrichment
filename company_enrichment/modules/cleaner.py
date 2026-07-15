"""
modules/cleaner.py
==================
Data cleaning pipeline for the Company Data Enrichment System.

Responsibilities:
    1. Trim whitespace from all string fields
    2. Normalise company names (title-case, remove legal suffixes noise, etc.)
    3. Canonicalise domains  (strip https://, http://, www., trailing slashes)
    4. Normalise LinkedIn URLs to a consistent format
    5. Remove / nullify invalid URLs (email, phone, social links)
    6. Replace blank strings and NaN with pd.NA uniformly
    7. Deduplicate rows — exact match first, then fuzzy company-name match
    8. Merge duplicate companies (union their non-null fields)

Usage:
    from modules.cleaner import DataCleaner

    cleaner = DataCleaner()
    clean_df = cleaner.clean(raw_df)
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compile heavy regexes once at module load time
# ---------------------------------------------------------------------------

# Matches any scheme + optional www prefix before the real domain
_SCHEME_RE = re.compile(
    r"^(?:https?://)?(?:www\d?\.)?",
    re.IGNORECASE,
)

# Trailing slashes, query strings, and fragments to strip from domains
_DOMAIN_TAIL_RE = re.compile(r"[/?#].*$")

# LinkedIn company URL normaliser — captures the slug only
_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/(?:company|in|school)/([^/?#\s]+)",
    re.IGNORECASE,
)

# Email-like strings that may mistakenly land in URL columns
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# Very loose URL validity check — must have a dot and valid TLD-ish ending
_URL_VALID_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?[a-z0-9\-]+(?:\.[a-z0-9\-]+)+",
    re.IGNORECASE,
)

# Characters that are never valid inside a bare domain
_DOMAIN_INVALID_CHARS_RE = re.compile(r"[\s,;|\"'<>()]")

# Legal / corporate suffixes to normalise (lowercase → canonical form)
# Used to avoid matching "Microsoft Corp" ≠ "Microsoft Corporation"
_LEGAL_SUFFIX_MAP: dict[str, str] = {
    r"\bcorp\b\.?": "Corporation",
    r"\binc\b\.?": "Inc",
    r"\bltd\b\.?": "Ltd",
    r"\bllc\b\.?": "LLC",
    r"\bllp\b\.?": "LLP",
    r"\bplc\b\.?": "PLC",
    r"\bgmbh\b\.?": "GmbH",
    r"\bag\b\.?": "AG",
    r"\bpte\b\.?": "Pte",
    r"\bsas\b\.?": "SAS",
    r"\bsa\b\.?": "SA",
    r"\bnv\b\.?": "NV",
    r"\bbv\b\.?": "BV",
}

# Social platform helpers
_FACEBOOK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?facebook\.com/([^/?#\s]+)",
    re.IGNORECASE,
)
_TWITTER_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:twitter|x)\.com/([^/?#\s]+)",
    re.IGNORECASE,
)
_INSTAGRAM_RE = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/([^/?#\s]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Low-level string helpers
# ---------------------------------------------------------------------------

def _to_str(value) -> str:
    """Coerce any value to a stripped string, returning '' for NA/None/NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", "null", "n/a", "na", "#n/a", "-"} else s


def _blank_to_na(value) -> object:
    """Return pd.NA for blank / NA-like values, otherwise return the value."""
    s = _to_str(value)
    return pd.NA if s == "" else s


def _normalise_name(name: str) -> str:
    """
    Clean and normalise a company name:
      - Strip whitespace
      - Collapse internal whitespace
      - Title-case
      - Expand abbreviated legal suffixes to their canonical form
    """
    name = " ".join(name.split())   # collapse runs of whitespace
    name = name.strip()

    # Apply legal suffix normalisations (case-insensitive)
    for pattern, replacement in _LEGAL_SUFFIX_MAP.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)

    # Title-case but preserve ALL-CAPS acronyms (e.g. "IBM", "SAP")
    words = name.split()
    title_words = []
    for word in words:
        if word.isupper() and len(word) > 1:
            title_words.append(word)       # preserve acronym
        else:
            title_words.append(word.title())
    return " ".join(title_words)


def _canonicalise_domain(raw: str) -> str:
    """
    Convert any URL / domain string into a bare canonical domain.

    Examples:
        https://www.microsoft.com/en-us/  →  microsoft.com
        http://www2.apple.com/            →  apple.com
        www.openai.com                    →  openai.com
        openai.com/about                  →  openai.com
    """
    if not raw:
        return ""

    raw = raw.strip()

    # Email addresses should not be treated as domains
    if _EMAIL_RE.match(raw):
        return ""

    # Must look vaguely like a URL
    if not _URL_VALID_RE.match(raw):
        return ""

    # Strip scheme + www
    domain = _SCHEME_RE.sub("", raw)
    # Strip path, query, fragment
    domain = _DOMAIN_TAIL_RE.sub("", domain)
    # Strip invalid characters
    if _DOMAIN_INVALID_CHARS_RE.search(domain):
        return ""

    return domain.lower()


def _normalise_linkedin(raw: str) -> str:
    """
    Normalise a LinkedIn URL to:
        https://www.linkedin.com/company/<slug>

    Returns '' if the value is not a recognisable LinkedIn URL.
    """
    if not raw:
        return ""
    m = _LINKEDIN_RE.search(raw)
    if m:
        slug = m.group(1).rstrip("/")
        return f"https://www.linkedin.com/company/{slug}"
    # Could be just a slug like "microsoft" — skip it (too ambiguous)
    return ""


def _normalise_social(raw: str, platform_re: re.Pattern, base_url: str) -> str:
    """Generic social URL normaliser (Facebook, Twitter, Instagram)."""
    if not raw:
        return ""
    m = platform_re.search(raw)
    if m:
        handle = m.group(1).rstrip("/")
        return f"{base_url}/{handle}"
    # If it already looks like a URL for the right platform, return clean
    if platform_re.pattern.split(r"\.com")[0].split("?:")[-1].split(r"\.")[-1] in raw.lower():
        return raw.strip()
    return ""


def _normalise_email(raw: str) -> str:
    """Return the email if valid, else ''."""
    raw = raw.strip().lower()
    return raw if _EMAIL_RE.match(raw) else ""


def _normalise_phone(raw: str) -> str:
    """
    Strip formatting characters and return a normalised phone number.
    Accepts digits, +, -, (, ), spaces. Returns '' if too short.
    """
    cleaned = re.sub(r"[^\d+\-()\s]", "", raw).strip()
    digits_only = re.sub(r"\D", "", cleaned)
    return cleaned if len(digits_only) >= 7 else ""


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------

class DataCleaner:
    """
    Stateless cleaning pipeline.  Call ``clean(df)`` to get a cleaned copy.

    Parameters
    ----------
    fuzzy_threshold : int
        RapidFuzz score (0-100) above which two company names are considered
        duplicates (default pulled from settings).
    """

    def __init__(self, fuzzy_threshold: Optional[int] = None) -> None:
        self.fuzzy_threshold: int = (
            fuzzy_threshold if fuzzy_threshold is not None
            else settings.fuzzy_name_threshold
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline on a DataFrame.

        Steps executed in order:
            1.  Coerce all values to strings / pd.NA
            2.  Trim whitespace on every column
            3.  Normalise company names
            4.  Canonicalise domains (website column)
            5.  Normalise LinkedIn URLs
            6.  Normalise social URLs (Facebook, Twitter, Instagram)
            7.  Normalise emails and phone numbers
            8.  Remove exact-duplicate rows
            9.  Fuzzy-deduplicate by company name
            10. Merge duplicate company records
        """
        logger.info("DataCleaner: starting clean on %d rows.", len(df))

        df = df.copy()   # never mutate the caller's DataFrame

        # Step 1 & 2 — coerce and trim
        df = self._coerce_and_trim(df)

        # Step 3 — normalise company names
        df = self._normalise_names(df)

        # Step 4 — canonicalise domains
        df = self._normalise_domains(df)

        # Step 5 — normalise LinkedIn
        df = self._normalise_linkedin_col(df)

        # Step 6 — normalise social URLs
        df = self._normalise_socials(df)

        # Step 7 — normalise emails and phones
        df = self._normalise_contacts(df)

        # Step 8 — remove exact duplicates
        before = len(df)
        df = self._remove_exact_duplicates(df)
        logger.info(
            "Exact dedup: removed %d rows → %d remain.", before - len(df), len(df)
        )

        # Step 9 & 10 — fuzzy dedup + merge
        before = len(df)
        df = self._fuzzy_deduplicate_and_merge(df)
        logger.info(
            "Fuzzy dedup: merged %d groups → %d remain.", before - len(df), len(df)
        )

        # Re-index
        df = df.reset_index(drop=True)
        logger.info("DataCleaner: finished. Output rows: %d.", len(df))
        return df

    # ------------------------------------------------------------------
    # Step 1 & 2 — coerce and trim
    # ------------------------------------------------------------------

    def _coerce_and_trim(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace all blank / NA-like values with pd.NA, trim strings."""
        for col in df.columns:
            df[col] = df[col].apply(_blank_to_na)
        return df

    # ------------------------------------------------------------------
    # Step 3 — company name normalisation
    # ------------------------------------------------------------------

    def _normalise_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise the 'name' (and 'company_name') columns."""
        for col in ("name", "company_name"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: _normalise_name(_to_str(v)) if pd.notna(v) else pd.NA
                )
                # Re-blank rows that normalised to empty string
                df[col] = df[col].replace("", pd.NA)
        return df

    # ------------------------------------------------------------------
    # Step 4 — domain canonicalisation
    # ------------------------------------------------------------------

    def _normalise_domains(self, df: pd.DataFrame) -> pd.DataFrame:
        """Canonicalise the 'website' column to bare domains."""
        if "website" not in df.columns:
            return df

        def _clean_domain(v) -> object:
            raw = _to_str(v)
            if not raw:
                return pd.NA
            canon = _canonicalise_domain(raw)
            return canon if canon else pd.NA

        df["website"] = df["website"].apply(_clean_domain)
        return df

    # ------------------------------------------------------------------
    # Step 5 — LinkedIn normalisation
    # ------------------------------------------------------------------

    def _normalise_linkedin_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise the 'linkedin' column."""
        if "linkedin" not in df.columns:
            return df

        def _clean_li(v) -> object:
            raw = _to_str(v)
            if not raw:
                return pd.NA
            normalised = _normalise_linkedin(raw)
            return normalised if normalised else pd.NA

        df["linkedin"] = df["linkedin"].apply(_clean_li)
        return df

    # ------------------------------------------------------------------
    # Step 6 — social URL normalisation
    # ------------------------------------------------------------------

    def _normalise_socials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise Facebook, Twitter, and Instagram URL columns."""
        social_configs = [
            ("facebook", _FACEBOOK_RE, "https://www.facebook.com"),
            ("twitter",  _TWITTER_RE,  "https://www.twitter.com"),
            ("instagram", _INSTAGRAM_RE, "https://www.instagram.com"),
        ]
        for col, pattern, base in social_configs:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v, p=pattern, b=base: (
                        _normalise_social(_to_str(v), p, b) or pd.NA
                    )
                    if pd.notna(v) else pd.NA
                )
        return df

    # ------------------------------------------------------------------
    # Step 7 — contact normalisation (email, phone)
    # ------------------------------------------------------------------

    def _normalise_contacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalise email and phone fields."""
        if "email" in df.columns:
            df["email"] = df["email"].apply(
                lambda v: _normalise_email(_to_str(v)) or pd.NA
                if pd.notna(v) else pd.NA
            )
        if "phone" in df.columns:
            df["phone"] = df["phone"].apply(
                lambda v: _normalise_phone(_to_str(v)) or pd.NA
                if pd.notna(v) else pd.NA
            )
        return df

    # ------------------------------------------------------------------
    # Step 8 — exact deduplication
    # ------------------------------------------------------------------

    def _remove_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that are exact duplicates on the subset:
            (name, website, linkedin)

        For each duplicate group the row with the most non-null fields
        is kept.
        """
        key_cols = [c for c in ("name", "website", "linkedin") if c in df.columns]
        if not key_cols:
            return df

        # Sort so that the richest row (most non-NA) is first in each group
        df["_non_null_count"] = df.notna().sum(axis=1)
        df = df.sort_values("_non_null_count", ascending=False)
        df = df.drop_duplicates(subset=key_cols, keep="first")
        df = df.drop(columns=["_non_null_count"])
        return df

    # ------------------------------------------------------------------
    # Step 9 & 10 — fuzzy deduplication + merge
    # ------------------------------------------------------------------

    def _fuzzy_deduplicate_and_merge(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group companies whose names are similar enough (above the fuzzy
        threshold) and merge each group into a single canonical row.

        Strategy:
          - Extract company names as the match key.
          - Build clusters using a greedy union-find approach via RapidFuzz.
          - For each cluster, merge rows: prefer non-null values from the
            row with the highest confidence_score / most filled fields.
        """
        if "name" not in df.columns:
            return df

        names: list[str] = [
            _to_str(v) for v in df["name"].tolist()
        ]
        n = len(names)

        if n <= 1:
            return df

        # -- Union-Find --
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        # Only compare non-empty names
        valid_indices = [i for i, n_val in enumerate(names) if n_val]

        # 1. Exact domain match
        domain_to_idx: dict[str, int] = {}
        if "website" in df.columns:
            for idx in range(n):
                domain = _to_str(df.at[idx, "website"])
                if domain:
                    if domain in domain_to_idx:
                        union(idx, domain_to_idx[domain])
                    else:
                        domain_to_idx[domain] = idx

        # 2. Fuzzy name match
        for i, idx_i in enumerate(valid_indices):
            name_i = names[idx_i]
            if not name_i:
                continue
            # RapidFuzz process.extract returns top matches efficiently
            candidates = process.extract(
                name_i,
                [names[j] for j in valid_indices[i + 1:]],
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.fuzzy_threshold,
                limit=None,
            )
            for _matched_name, _score, rel_idx in candidates:
                abs_idx = valid_indices[i + 1 + rel_idx]
                union(idx_i, abs_idx)

        # -- Group rows by cluster root --
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        # -- Merge each cluster --
        merged_rows: list[dict] = []
        for root, indices in clusters.items():
            if len(indices) == 1:
                merged_rows.append(df.iloc[indices[0]].to_dict())
            else:
                merged_rows.append(self._merge_rows(df.iloc[indices]))

        result = pd.DataFrame(merged_rows, columns=df.columns)
        return result

    # ------------------------------------------------------------------
    # Row-level merge helper
    # ------------------------------------------------------------------

    def _merge_rows(self, group: pd.DataFrame) -> dict:
        """
        Merge a group of duplicate rows into one canonical row.

        Rules:
          - Numeric confidence_score: take the maximum.
          - All other fields: prefer the first non-NA value from the row
            with the highest confidence_score (ties broken by most-filled row).
        """
        # Sort by confidence (desc) then filled fields (desc)
        try:
            group = group.copy()
            group["_cs"] = pd.to_numeric(
                group.get("confidence_score", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0)
            group["_fill"] = group.notna().sum(axis=1)
            group = group.sort_values(["_cs", "_fill"], ascending=False)
        except Exception:
            pass  # If sorting fails, proceed with original order

        merged: dict = {}
        for col in group.columns:
            if col.startswith("_"):
                continue  # skip temp columns
            # Take the first non-NA value across the group
            for val in group[col]:
                if pd.notna(val) and _to_str(val):
                    merged[col] = val
                    break
            else:
                merged[col] = pd.NA

        # For confidence_score, take the max across the group
        if "confidence_score" in group.columns:
            scores = pd.to_numeric(group["confidence_score"], errors="coerce")
            if scores.notna().any():
                merged["confidence_score"] = str(int(scores.max()))

        return merged


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Shortcut: instantiate DataCleaner with defaults and clean ``df``."""
    return DataCleaner().clean(df)
