"""
modules/data_loader.py
======================
Responsible for reading all CSV files from the input directory,
auto-detecting non-standard column names, mapping them to the
standard schema, merging all files into a single DataFrame, and
handling malformed/empty files gracefully without halting the pipeline.

Standard output columns (always present after loading):
    company_id, name, company_name, website, linkedin, description,
    industry, size, email, phone, facebook, twitter, instagram,
    ceo, founder, source_used, confidence_score, status, last_updated
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard schema — every output DataFrame will have exactly these columns.
# ---------------------------------------------------------------------------
STANDARD_COLUMNS: list[str] = [
    "company_id",
    "name",
    "company_name",
    "website",
    "linkedin",
    "description",
    "industry",
    "size",
    "email",
    "phone",
    "facebook",
    "twitter",
    "instagram",
    "ceo",
    "founder",
    "source_used",
    "confidence_score",
    "status",
    "last_updated",
]

# ---------------------------------------------------------------------------
# Column alias map
# Keys   = canonical field name (our standard schema)
# Values = list of raw column names found in the wild (all lowercase)
#
# The matcher is case-insensitive and also performs partial-word matching
# as a fallback (see _detect_column).
# ---------------------------------------------------------------------------
COLUMN_ALIASES: dict[str, list[str]] = {
    "name": [
        "company_name",
        "company name",
        "input-company-name",
        "input_company_name",
        "account_name",
        "account name",
        "organization",
        "organisation",
        "firm",
        "name",
    ],
    "website": [
        "website",
        "domain",
        "company_domain",
        "company domain",
        "input-company-domain",
        "input_company_domain",
        "web",
        "url",
        "homepage",
        "company_url",
        "site",
        "web_address",
    ],
    "linkedin": [
        "linkedin",
        "linkedin_url",
        "linkedin url",
        "linkedin_profile",
        "li_url",
        "linked_in",
    ],
    "description": [
        "description",
        "company_description",
        "about",
        "overview",
        "summary",
        "bio",
    ],
    "industry": [
        "industry",
        "sector",
        "vertical",
        "market",
        "industry_type",
    ],
    "size": [
        "size",
        "company_size",
        "employee_count",
        "employees",
        "headcount",
        "staff_count",
        "num_employees",
        "number_of_employees",
    ],
    "email": [
        "email",
        "email_address",
        "contact_email",
        "company_email",
        "e_mail",
    ],
    "phone": [
        "phone",
        "phone_number",
        "telephone",
        "contact_phone",
        "tel",
        "mobile",
        "contact_number",
    ],
    "facebook": [
        "facebook",
        "facebook_url",
        "fb",
        "facebook_page",
    ],
    "twitter": [
        "twitter",
        "twitter_url",
        "twitter_handle",
        "x_url",
        "x_handle",
    ],
    "instagram": [
        "instagram",
        "instagram_url",
        "ig",
        "instagram_handle",
    ],
    "ceo": [
        "ceo",
        "chief_executive_officer",
        "chief executive",
        "ceo_name",
        "executive",
    ],
    "founder": [
        "founder",
        "co_founder",
        "co-founder",
        "founders",
        "founding_member",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_col(col: str) -> str:
    """Lowercase, strip, replace hyphens/spaces with underscores."""
    return col.strip().lower().replace("-", "_").replace(" ", "_")


def _detect_column(
    raw_columns: list[str],
    canonical: str,
    aliases: list[str],
) -> Optional[str]:
    """
    Find the first raw column that matches any alias for a canonical field.

    Matching order (most-specific first):
      1. Exact match (after normalisation)
      2. Partial / substring match as a last resort

    Returns the *original* (un-normalised) raw column name so we can use
    it to rename the DataFrame column, or None if no match found.
    """
    norm_map: dict[str, str] = {_normalise_col(c): c for c in raw_columns}
    norm_aliases: list[str] = [_normalise_col(a) for a in aliases]

    # 1. Exact match
    for alias in norm_aliases:
        if alias in norm_map:
            return norm_map[alias]

    # 2. Substring match (e.g. "company_website_url" contains "website")
    for norm_raw, orig_raw in norm_map.items():
        for alias in norm_aliases:
            if alias in norm_raw:
                return orig_raw

    return None


def _build_column_mapping(raw_columns: list[str]) -> dict[str, str]:
    """
    Build a {raw_col_name: canonical_name} mapping for a single CSV's columns.

    Only columns that successfully map to a canonical field are included.
    """
    mapping: dict[str, str] = {}
    already_mapped: set[str] = set()  # avoid mapping two raw cols → same canonical

    for canonical, aliases in COLUMN_ALIASES.items():
        raw_match = _detect_column(raw_columns, canonical, aliases)
        if raw_match and canonical not in already_mapped:
            mapping[raw_match] = canonical
            already_mapped.add(canonical)
            logger.debug(
                "Mapped raw column '%s' → canonical '%s'", raw_match, canonical
            )

    return mapping


def _generate_company_id(name: str, website: str) -> str:
    """
    Deterministic UUID-like ID derived from company name + website.
    Ensures the same company always gets the same ID across runs.
    """
    seed = f"{str(name).strip().lower()}|{str(website).strip().lower()}"
    hex_digest = hashlib.md5(seed.encode()).hexdigest()  # noqa: S324 (non-crypto use)
    return str(uuid.UUID(hex=hex_digest))


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add any missing standard columns (filled with pd.NA) and drop
    any columns that are not part of the standard schema.
    We keep extra source columns by *not* dropping them here — they are
    simply retained alongside the standard schema for traceability.
    """
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _assign_company_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a deterministic company_id for every row that lacks one.
    Rows that already have a company_id are left untouched (resume support).
    """
    mask = df["company_id"].isna() | (df["company_id"].astype(str).str.strip() == "")
    df.loc[mask, "company_id"] = df.loc[mask].apply(
        lambda row: _generate_company_id(
            row.get("name", "") or "",
            row.get("website", "") or "",
        ),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Loads and unifies all CSV files from the configured input directory.

    Usage:
        loader = DataLoader()
        df = loader.load_all()
    """

    def __init__(self, input_dir: Optional[Path] = None) -> None:
        self.input_dir: Path = input_dir or settings.input_dir
        self._loaded_files: list[str] = []
        self._failed_files: list[str] = []

    # ------------------------------------------------------------------
    # Single-file loader
    # ------------------------------------------------------------------

    def load_single(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Load one CSV, detect its column names, and return a DataFrame
        that conforms to the standard schema.

        Returns None if the file cannot be read or is empty.
        """
        logger.info("Loading file: %s", csv_path.name)

        # -- Try multiple encodings in order of likelihood --
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        df: Optional[pd.DataFrame] = None

        for encoding in encodings:
            try:
                df = pd.read_csv(
                    csv_path,
                    encoding=encoding,
                    dtype=str,             # Read everything as string initially
                    keep_default_na=False,  # We handle NA ourselves
                    on_bad_lines="warn",   # Skip bad lines, don't crash
                    engine="python",       # More forgiving than C engine
                )
                logger.debug(
                    "Loaded '%s' with encoding '%s' (%d rows, %d cols)",
                    csv_path.name, encoding, len(df), len(df.columns),
                )
                break  # Successfully read — stop trying encodings
            except pd.errors.EmptyDataError:
                logger.warning("File '%s' is empty — skipping.", csv_path.name)
                return None
            except Exception:
                # Try next encoding silently
                continue

        if df is None:
            logger.error(
                "Could not read '%s' with any supported encoding.", csv_path.name
            )
            return None

        # -- Guard: must have at least one column --
        if df.empty or len(df.columns) == 0:
            logger.warning("File '%s' has no usable columns — skipping.", csv_path.name)
            return None

        # -- Detect and rename columns to canonical names --
        raw_columns: list[str] = df.columns.tolist()
        col_mapping = _build_column_mapping(raw_columns)

        if not col_mapping:
            logger.warning(
                "No recognisable columns found in '%s'. "
                "Columns present: %s",
                csv_path.name, raw_columns,
            )
            # Still include the file but all standard fields will be NA
        else:
            df = df.rename(columns=col_mapping)
            logger.debug(
                "Column mapping applied for '%s': %s",
                csv_path.name, col_mapping,
            )

        # -- Add standard columns that are missing --
        df = _enforce_schema(df)

        # -- Tag source file for traceability --
        df["_source_file"] = csv_path.name

        # -- Initialise pipeline status fields for new rows --
        df["status"] = df["status"].where(
            df["status"].notna() & (df["status"].astype(str).str.strip() != ""),
            other="NOT_FOUND",
        )
        df["confidence_score"] = df["confidence_score"].where(
            df["confidence_score"].notna()
            & (df["confidence_score"].astype(str).str.strip() != ""),
            other="0",
        )

        self._loaded_files.append(csv_path.name)
        return df

    # ------------------------------------------------------------------
    # Bulk loader
    # ------------------------------------------------------------------

    def load_all(self) -> pd.DataFrame:
        """
        Discover and load every *.csv file under ``input_dir``.

        Returns a single unified DataFrame with the standard schema.
        Continues processing even when individual files fail.
        """
        csv_files: list[Path] = sorted(self.input_dir.glob("**/*.csv"))

        if not csv_files:
            logger.warning(
                "No CSV files found in '%s'. "
                "Please place your input CSVs in that directory.",
                self.input_dir,
            )
            return self._empty_dataframe()

        logger.info(
            "Discovered %d CSV file(s) in '%s'.", len(csv_files), self.input_dir
        )

        frames: list[pd.DataFrame] = []

        for csv_path in csv_files:
            try:
                df_single = self.load_single(csv_path)
                if df_single is not None and not df_single.empty:
                    frames.append(df_single)
            except Exception as exc:
                logger.error(
                    "Unexpected error loading '%s': %s", csv_path.name, exc, exc_info=True
                )
                self._failed_files.append(csv_path.name)

        if not frames:
            logger.error("No data could be loaded from any CSV file.")
            return self._empty_dataframe()

        # -- Concatenate all DataFrames --
        logger.info("Concatenating %d file(s) into a single DataFrame…", len(frames))
        combined: pd.DataFrame = pd.concat(frames, ignore_index=True, sort=False)

        # -- Replace empty strings with pd.NA for uniform NA handling --
        combined = combined.replace(r"^\s*$", pd.NA, regex=True)

        # -- Assign deterministic company IDs --
        combined = _assign_company_ids(combined)

        logger.info(
            "DataLoader complete. Loaded: %d file(s), Failed: %d file(s). "
            "Total rows: %d.",
            len(self._loaded_files),
            len(self._failed_files),
            len(combined),
        )

        if self._failed_files:
            logger.warning("Failed files: %s", self._failed_files)

        return combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_dataframe(self) -> pd.DataFrame:
        """Return an empty DataFrame with the standard schema columns."""
        return pd.DataFrame(columns=STANDARD_COLUMNS + ["_source_file"])

    @property
    def loaded_files(self) -> list[str]:
        """Names of files that were successfully loaded."""
        return self._loaded_files.copy()

    @property
    def failed_files(self) -> list[str]:
        """Names of files that failed to load."""
        return self._failed_files.copy()

    def summary(self) -> dict:
        """Return a human-readable load summary dict."""
        return {
            "loaded": len(self._loaded_files),
            "failed": len(self._failed_files),
            "loaded_files": self._loaded_files,
            "failed_files": self._failed_files,
        }
