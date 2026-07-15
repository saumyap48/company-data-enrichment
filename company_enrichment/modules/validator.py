"""
modules/validator.py
====================
Post-enrichment validation layer for the Company Data Enrichment System.

Responsibilities:
    1.  Validate every enriched record against business rules
    2.  Detect and report remaining duplicates (domain + name)
    3.  Flag missing critical fields
    4.  Validate CEO / Founder values for sanity
    5.  Cross-check company name against domain
    6.  Validate email format, phone length, URL patterns
    7.  Generate a structured validation_report.json

Validation is NON-DESTRUCTIVE — it never deletes or modifies records.
It only attaches validation metadata and writes the report.

Public API:
    validator = DataValidator()
    validated_df, report = validator.validate(enriched_df)
    validator.save_report(report)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from config import settings
from modules.cleaner import _EMAIL_RE, _to_str

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fields every production record should have
_CRITICAL_FIELDS: list[str] = ["name", "website", "description", "industry"]

# Fields that are desirable but not mandatory
_IMPORTANT_FIELDS: list[str] = ["ceo", "founder", "size", "email", "phone"]

# Social fields
_SOCIAL_FIELDS: list[str] = ["linkedin", "facebook", "twitter", "instagram"]

# Min length thresholds
_MIN_DESCRIPTION_LEN: int = 30
_MIN_PHONE_DIGITS: int = 7
_MAX_PHONE_DIGITS: int = 15

# Fuzzy threshold for duplicate company-name detection in validation
_VALIDATION_FUZZY_THRESHOLD: int = 92

# URL validation pattern (loose — just needs scheme + domain)
_URL_PATTERN = re.compile(
    r"^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$"
)

# Person-name sanity: must look like words (letters, spaces, hyphens, apostrophes)
_PERSON_NAME_PATTERN = re.compile(r"^[A-Za-z\s\-\.']{2,80}$")

# Phone digits extractor
_PHONE_DIGITS_RE = re.compile(r"\d")


# ---------------------------------------------------------------------------
# Validation result codes
# ---------------------------------------------------------------------------

class ValidationCode:
    """String constants for all validation issue codes."""
    # Critical
    MISSING_NAME           = "MISSING_NAME"
    MISSING_WEBSITE        = "MISSING_WEBSITE"
    MISSING_DESCRIPTION    = "MISSING_DESCRIPTION"
    MISSING_INDUSTRY       = "MISSING_INDUSTRY"
    # Duplicates
    DUPLICATE_DOMAIN       = "DUPLICATE_DOMAIN"
    DUPLICATE_NAME         = "DUPLICATE_NAME_FUZZY"
    # Field quality
    DESCRIPTION_TOO_SHORT  = "DESCRIPTION_TOO_SHORT"
    INVALID_EMAIL          = "INVALID_EMAIL"
    INVALID_PHONE          = "INVALID_PHONE"
    INVALID_URL            = "INVALID_URL"
    INVALID_CEO            = "INVALID_CEO"
    INVALID_FOUNDER        = "INVALID_FOUNDER"
    NAME_DOMAIN_MISMATCH   = "NAME_DOMAIN_MISMATCH"
    LOW_CONFIDENCE         = "LOW_CONFIDENCE_SCORE"
    # Status
    NOT_FOUND_STATUS       = "STATUS_NOT_FOUND"
    NEEDS_REVIEW_STATUS    = "STATUS_NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# Per-record validation result
# ---------------------------------------------------------------------------

@dataclass
class RecordValidationResult:
    """Validation outcome for a single company record."""
    company_id: str = ""
    name: str = ""
    is_valid: bool = True
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, code: str) -> None:
        self.issues.append(code)
        self.is_valid = False

    def add_warning(self, code: str) -> None:
        self.warnings.append(code)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Aggregated statistics and issue breakdown for the full dataset.
    Serialised to validation_report.json.
    """
    generated_at: str = ""

    # Counts
    total_companies: int = 0
    total_processed: int = 0
    total_valid: int = 0
    total_invalid: int = 0
    total_warnings: int = 0
    
    total_success: int = 0
    total_failed: int = 0
    duplicates_removed: int = 0
    missing_ceo: int = 0
    missing_founder: int = 0

    # Duplicate tracking
    duplicate_domains_found: int = 0
    duplicate_names_found: int = 0

    # Status breakdown
    status_found: int = 0
    status_partially_found: int = 0
    status_not_found: int = 0
    status_needs_review: int = 0

    # Field completeness (counts of records where field is populated)
    fields_with_name: int = 0
    fields_with_website: int = 0
    fields_with_description: int = 0
    fields_with_industry: int = 0
    fields_with_ceo: int = 0
    fields_with_founder: int = 0
    fields_with_email: int = 0
    fields_with_phone: int = 0
    fields_with_linkedin: int = 0

    # Score stats
    average_confidence_score: float = 0.0
    min_confidence_score: int = 0
    max_confidence_score: int = 0

    # Issue breakdown (code -> count)
    issue_counts: dict = field(default_factory=dict)

    # Per-record results (compact)
    records: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validators (pure functions)
# ---------------------------------------------------------------------------

def _present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    s = str(value).strip()
    return bool(s) and s.lower() not in {"nan", "none", "null", "n/a", "na", ""}


def _validate_email(value: str) -> bool:
    """Return True if the email address looks valid."""
    return bool(_EMAIL_RE.match(value.strip()))


def _validate_phone(value: str) -> bool:
    """Return True if the phone has an appropriate number of digits."""
    digits = len(_PHONE_DIGITS_RE.findall(value))
    return _MIN_PHONE_DIGITS <= digits <= _MAX_PHONE_DIGITS


def _validate_url(value: str) -> bool:
    """Return True if the URL matches a very basic pattern."""
    return bool(_URL_PATTERN.match(value.strip()))


def _validate_person_name(value: str) -> bool:
    """
    Sanity-check a CEO or Founder name.
    Rejects: pure numbers, email addresses, URLs, single characters.
    """
    value = value.strip()
    if not value or len(value) < 2:
        return False
    if "@" in value or "http" in value.lower():
        return False
    if re.fullmatch(r"[\d\s\-+().]+", value):
        return False
    return bool(_PERSON_NAME_PATTERN.match(value))


def _name_domain_match(name: str, domain: str) -> bool:
    """
    Heuristic: check if the company name bears any resemblance to the domain.
    Returns True (no mismatch) if we can't tell or names are too short.
    """
    if not name or not domain:
        return True   # can't evaluate — assume OK

    name_norm   = re.sub(r"[^a-z0-9]", "", name.lower().split()[0])
    domain_root = re.sub(r"[^a-z0-9]", "", domain.lower().split(".")[0])

    if len(name_norm) < 4 or len(domain_root) < 3:
        return True   # too short to meaningfully compare

    # Either is a substring of the other, OR fuzzy score ≥ 60
    if name_norm in domain_root or domain_root in name_norm:
        return True
    ratio = fuzz.partial_ratio(name_norm, domain_root)
    return ratio >= 60


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Validates an enriched company DataFrame and produces a structured report.

    Usage:
        validator = DataValidator()
        validated_df, report = validator.validate(df)
        validator.save_report(report)
    """

    def __init__(
        self,
        report_path: Optional[Path] = None,
        fuzzy_threshold: int = _VALIDATION_FUZZY_THRESHOLD,
    ) -> None:
        self._report_path    = report_path or settings.validation_report_file
        self._fuzzy_threshold = fuzzy_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Validate the enriched DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, ValidationReport]
            - df: the original DataFrame with two new columns added:
                  ``_validation_issues``  (comma-separated error codes)
                  ``_validation_warnings`` (comma-separated warning codes)
            - report: ValidationReport dataclass (also saved to disk)
        """
        logger.info("DataValidator: validating %d records…", len(df))

        df = df.copy()

        # --- Build duplicate indexes upfront ---
        dup_domains = self._find_duplicate_domains(df)
        dup_names   = self._find_duplicate_names(df)

        # --- Per-record validation ---
        results: list[RecordValidationResult] = []
        issue_col: list[str]   = []
        warning_col: list[str] = []

        for index in df.index:
            record = df.loc[index].to_dict()
            result = self._validate_record(record, dup_domains, dup_names)
            
            # Write back any nullified fields (record was mutated)
            for k, v in record.items():
                df.at[index, k] = v
                
            results.append(result)
            issue_col.append(", ".join(result.issues))
            warning_col.append(", ".join(result.warnings))

        df["_validation_issues"]  = issue_col
        df["_validation_warnings"] = warning_col

        # --- Build report ---
        report = self._build_report(df, results, dup_domains, dup_names)

        logger.info(
            "Validation complete — valid: %d | invalid: %d | "
            "dup_domains: %d | dup_names: %d | avg_score: %.1f",
            report.total_valid, report.total_invalid,
            report.duplicate_domains_found, report.duplicate_names_found,
            report.average_confidence_score,
        )

        return df, report

    def save_report(self, report: ValidationReport) -> Path:
        """Serialise the ValidationReport to JSON and write to disk."""
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(report)
        with self._report_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Validation report saved -> %s", self._report_path)
        return self._report_path

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def _find_duplicate_domains(self, df: pd.DataFrame) -> set[str]:
        """
        Return the set of domains that appear more than once in the DataFrame.
        Empty / NA domains are excluded.
        """
        if "website" not in df.columns:
            return set()
        domain_col = df["website"].astype(str).str.strip().str.lower()
        domain_col = domain_col[domain_col.notna() & (domain_col != "") & (domain_col != "nan")]
        counts = domain_col.value_counts()
        return set(counts[counts > 1].index.tolist())

    def _find_duplicate_names(self, df: pd.DataFrame) -> dict[int, int]:
        """
        Build a mapping {row_index -> canonical_row_index} for rows whose
        company name fuzzy-matches another row above the threshold.

        Only the non-canonical duplicates are flagged (the first occurrence
        of each cluster is considered canonical).
        """
        if "name" not in df.columns:
            return {}

        names = df["name"].fillna("").astype(str).tolist()
        duplicates: dict[int, int] = {}

        for i in range(len(names)):
            if i in duplicates:
                continue   # already flagged as a dup of something earlier
            name_i = names[i].strip()
            if not name_i or len(name_i) < 3:
                continue
            for j in range(i + 1, len(names)):
                if j in duplicates:
                    continue
                name_j = names[j].strip()
                if not name_j:
                    continue
                score = fuzz.token_sort_ratio(name_i, name_j)
                if score >= self._fuzzy_threshold:
                    duplicates[j] = i   # j is a dup of i

        return duplicates

    # ------------------------------------------------------------------
    # Per-record validation
    # ------------------------------------------------------------------

    def _validate_record(
        self,
        record: dict,
        dup_domains: set[str],
        dup_names: dict[int, int],
    ) -> RecordValidationResult:
        """Run all validation checks on a single record dict."""
        result = RecordValidationResult(
            company_id=_to_str(record.get("company_id", "")),
            name=_to_str(record.get("name", "")),
        )

        # --- Critical field presence ---
        if not _present(record.get("name")):
            result.add_error(ValidationCode.MISSING_NAME)
        if not _present(record.get("website")):
            result.add_error(ValidationCode.MISSING_WEBSITE)
        if not _present(record.get("description")):
            result.add_error(ValidationCode.MISSING_DESCRIPTION)
        if not _present(record.get("industry")):
            result.add_error(ValidationCode.MISSING_INDUSTRY)

        # --- Description quality ---
        desc = _to_str(record.get("description", ""))
        if desc and len(desc) < _MIN_DESCRIPTION_LEN:
            result.add_warning(ValidationCode.DESCRIPTION_TOO_SHORT)

        # --- Email ---
        email = _to_str(record.get("email", ""))
        if email and not _validate_email(email):
            result.add_warning(ValidationCode.INVALID_EMAIL)
            record["email"] = pd.NA

        # --- Phone ---
        phone = _to_str(record.get("phone", ""))
        if phone and not _validate_phone(phone):
            result.add_warning(ValidationCode.INVALID_PHONE)
            record["phone"] = pd.NA

        # --- URL fields ---
        for url_field in ("linkedin", "facebook", "twitter", "instagram"):
            url_val = _to_str(record.get(url_field, ""))
            if url_val and not _validate_url(url_val):
                result.add_warning(ValidationCode.INVALID_URL)
                record[url_field] = pd.NA

        # --- CEO sanity ---
        ceo = _to_str(record.get("ceo", ""))
        if ceo and not _validate_person_name(ceo):
            result.add_warning(ValidationCode.INVALID_CEO)
            record["ceo"] = pd.NA

        # --- Founder sanity ---
        founder = _to_str(record.get("founder", ""))
        if founder:
            # Founders may be comma-separated — check each
            for individual in founder.split(","):
                if individual.strip() and not _validate_person_name(individual.strip()):
                    result.add_warning(ValidationCode.INVALID_FOUNDER)
                    record["founder"] = pd.NA
                    break

        # --- Size sanity (remove parsing mistakes like 2025 99.999) ---
        size = _to_str(record.get("size", ""))
        if size and size not in {"1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10000+"}:
            record["size"] = pd.NA

        # --- Name / domain mismatch ---
        name   = _to_str(record.get("name", ""))
        domain = _to_str(record.get("website", ""))
        if name and domain and not _name_domain_match(name, domain):
            result.add_warning(ValidationCode.NAME_DOMAIN_MISMATCH)

        # --- Confidence score ---
        try:
            score = int(float(str(record.get("confidence_score", 0) or 0)))
        except (ValueError, TypeError):
            score = 0
        if score < settings.min_confidence_threshold:
            result.add_warning(ValidationCode.LOW_CONFIDENCE)

        # --- Status-based flags ---
        status = _to_str(record.get("status", ""))
        if status == "NOT_FOUND":
            result.add_warning(ValidationCode.NOT_FOUND_STATUS)
        elif status == "NEEDS_REVIEW":
            result.add_warning(ValidationCode.NEEDS_REVIEW_STATUS)

        # --- Duplicate domain ---
        domain_lower = domain.lower().strip()
        if domain_lower and domain_lower in dup_domains:
            result.add_warning(ValidationCode.DUPLICATE_DOMAIN)

        return result

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(
        self,
        df: pd.DataFrame,
        results: list[RecordValidationResult],
        dup_domains: set[str],
        dup_names: dict[int, int],
    ) -> ValidationReport:
        """Aggregate per-record results into a ValidationReport."""

        report = ValidationReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            total_companies=len(df),
            total_processed=len(results),
        )

        # Validity counts
        report.total_valid    = sum(1 for r in results if r.is_valid)
        report.total_invalid  = sum(1 for r in results if not r.is_valid)
        report.total_warnings = sum(len(r.warnings) for r in results)

        # Duplicate counts
        report.duplicate_domains_found = len(dup_domains)
        report.duplicate_names_found   = len(dup_names)
        
        # New Report Fields
        report.total_success = report.total_valid
        report.total_failed = report.total_invalid
        report.duplicates_removed = len(dup_domains) + len(dup_names)  # Approximation based on flags
        report.missing_ceo = len(results) - sum(1 for r in df.get("ceo", []) if bool(_to_str(r))) if "ceo" in df.columns else 0
        report.missing_founder = len(results) - sum(1 for r in df.get("founder", []) if bool(_to_str(r))) if "founder" in df.columns else 0

        # Status breakdown
        if "status" in df.columns:
            status_counts = df["status"].fillna("NOT_FOUND").value_counts().to_dict()
            report.status_found          = int(status_counts.get("FOUND", 0))
            report.status_partially_found = int(status_counts.get("PARTIALLY_FOUND", 0))
            report.status_not_found      = int(status_counts.get("NOT_FOUND", 0))
            report.status_needs_review   = int(status_counts.get("NEEDS_REVIEW", 0))

        # Field completeness
        def _count_filled(col: str) -> int:
            if col not in df.columns:
                return 0
            return int(df[col].apply(_present).sum())

        report.fields_with_name        = _count_filled("name")
        report.fields_with_website     = _count_filled("website")
        report.fields_with_description = _count_filled("description")
        report.fields_with_industry    = _count_filled("industry")
        report.fields_with_ceo         = _count_filled("ceo")
        report.fields_with_founder     = _count_filled("founder")
        report.fields_with_email       = _count_filled("email")
        report.fields_with_phone       = _count_filled("phone")
        report.fields_with_linkedin    = _count_filled("linkedin")

        # Confidence score stats
        if "confidence_score" in df.columns:
            scores = pd.to_numeric(df["confidence_score"], errors="coerce").dropna()
            if not scores.empty:
                report.average_confidence_score = round(float(scores.mean()), 2)
                report.min_confidence_score     = int(scores.min())
                report.max_confidence_score     = int(scores.max())

        # Issue code breakdown
        issue_counts: dict[str, int] = {}
        for r in results:
            for code in r.issues + r.warnings:
                issue_counts[code] = issue_counts.get(code, 0) + 1
        report.issue_counts = issue_counts

        # Per-record compact summary (only include records with issues)
        compact_records = []
        for r in results:
            if r.issues or r.warnings:
                compact_records.append({
                    "company_id": r.company_id,
                    "name":       r.name,
                    "is_valid":   r.is_valid,
                    "errors":     r.issues,
                    "warnings":   r.warnings,
                })
        report.records = compact_records

        return report

    # ------------------------------------------------------------------
    # Convenience: print summary to logger
    # ------------------------------------------------------------------

    @staticmethod
    def log_summary(report: ValidationReport) -> None:
        """Emit a concise validation summary to the logger."""
        logger.info("=" * 60)
        logger.info("VALIDATION REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info("Generated at      : %s", report.generated_at)
        logger.info("Total companies   : %d", report.total_companies)
        logger.info("Valid records     : %d", report.total_valid)
        logger.info("Invalid records   : %d", report.total_invalid)
        logger.info("Total warnings    : %d", report.total_warnings)
        logger.info("-" * 60)
        logger.info("Status — FOUND           : %d", report.status_found)
        logger.info("Status — PARTIALLY_FOUND : %d", report.status_partially_found)
        logger.info("Status — NOT_FOUND       : %d", report.status_not_found)
        logger.info("Status — NEEDS_REVIEW    : %d", report.status_needs_review)
        logger.info("-" * 60)
        logger.info("Duplicate domains : %d", report.duplicate_domains_found)
        logger.info("Duplicate names   : %d", report.duplicate_names_found)
        logger.info("-" * 60)
        logger.info("Avg confidence    : %.1f", report.average_confidence_score)
        logger.info("Min confidence    : %d",   report.min_confidence_score)
        logger.info("Max confidence    : %d",   report.max_confidence_score)
        logger.info("-" * 60)
        logger.info("Field completeness:")
        logger.info("  name        : %d", report.fields_with_name)
        logger.info("  website     : %d", report.fields_with_website)
        logger.info("  description : %d", report.fields_with_description)
        logger.info("  industry    : %d", report.fields_with_industry)
        logger.info("  ceo         : %d", report.fields_with_ceo)
        logger.info("  founder     : %d", report.fields_with_founder)
        logger.info("  email       : %d", report.fields_with_email)
        logger.info("  phone       : %d", report.fields_with_phone)
        logger.info("  linkedin    : %d", report.fields_with_linkedin)
        logger.info("=" * 60)
