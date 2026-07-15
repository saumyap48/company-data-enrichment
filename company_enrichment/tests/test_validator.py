"""
tests/test_validator.py
=======================
Unit tests for modules/validator.py

Tests:
    - Per-record validation (missing fields, bad email, bad phone, bad URLs)
    - CEO / Founder name sanity checks
    - Name / domain mismatch heuristic
    - Duplicate domain detection
    - Fuzzy duplicate name detection
    - ValidationReport generation (counts, completeness, issue codes)
    - Report serialisation (save_report)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from modules.validator import (
    DataValidator,
    RecordValidationResult,
    ValidationCode,
    ValidationReport,
    _name_domain_match,
    _present,
    _validate_email,
    _validate_person_name,
    _validate_phone,
    _validate_url,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a test DataFrame, filling missing standard columns with pd.NA."""
    from modules.data_loader import STANDARD_COLUMNS
    df = pd.DataFrame(rows)
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _good_record(**overrides) -> dict:
    """Return a complete, valid company record dict."""
    base = {
        "company_id": "test-id-001",
        "name": "Acme Corporation",
        "website": "acme.com",
        "description": "A global leader in industrial solutions and manufacturing products.",
        "industry": "Manufacturing",
        "ceo": "Jane Smith",
        "founder": "John Smith",
        "email": "info@acme.com",
        "phone": "+1 (555) 123-4567",
        "linkedin": "https://www.linkedin.com/company/acme",
        "facebook": "https://www.facebook.com/acme",
        "twitter": "https://www.twitter.com/acme",
        "instagram": "https://www.instagram.com/acme",
        "confidence_score": "85",
        "status": "FOUND",
        "source_used": "official_website",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Low-level validators
# ---------------------------------------------------------------------------

class TestValidateEmail:
    def test_valid(self):
        assert _validate_email("info@example.com") is True

    def test_valid_subdomain(self):
        assert _validate_email("user@mail.example.co.uk") is True

    def test_no_at_sign(self):
        assert _validate_email("notanemail.com") is False

    def test_no_domain(self):
        assert _validate_email("info@") is False

    def test_empty(self):
        assert _validate_email("") is False


class TestValidatePhone:
    def test_valid_us(self):
        assert _validate_phone("+1 555 123 4567") is True

    def test_valid_uk(self):
        assert _validate_phone("+44 20 7946 0958") is True

    def test_too_short(self):
        assert _validate_phone("12345") is False

    def test_too_long(self):
        assert _validate_phone("1" * 20) is False

    def test_empty(self):
        assert _validate_phone("") is False


class TestValidateUrl:
    def test_valid_https(self):
        assert _validate_url("https://www.linkedin.com/company/acme") is True

    def test_valid_http(self):
        assert _validate_url("http://facebook.com/acme") is True

    def test_no_scheme(self):
        assert _validate_url("linkedin.com/company/acme") is False

    def test_empty(self):
        assert _validate_url("") is False


class TestValidatePersonName:
    def test_valid_name(self):
        assert _validate_person_name("Jane Smith") is True

    def test_valid_hyphenated(self):
        assert _validate_person_name("Mary-Jane Watson") is True

    def test_digits_only(self):
        assert _validate_person_name("1234567") is False

    def test_url(self):
        assert _validate_person_name("https://example.com") is False

    def test_email(self):
        assert _validate_person_name("ceo@example.com") is False

    def test_too_short(self):
        assert _validate_person_name("A") is False

    def test_empty(self):
        assert _validate_person_name("") is False


class TestNameDomainMatch:
    def test_match(self):
        assert _name_domain_match("Microsoft Corporation", "microsoft.com") is True

    def test_mismatch(self):
        assert _name_domain_match("Microsoft Corporation", "apple.com") is False

    def test_partial_match(self):
        assert _name_domain_match("Google LLC", "google.io") is True

    def test_empty_name(self):
        assert _name_domain_match("", "example.com") is True  # can't evaluate → OK

    def test_empty_domain(self):
        assert _name_domain_match("Acme", "") is True  # can't evaluate → OK

    def test_short_name(self):
        assert _name_domain_match("IBM", "ibm.com") is True  # short → skip check


# ---------------------------------------------------------------------------
# DataValidator._validate_record
# ---------------------------------------------------------------------------

class TestValidateRecord:
    def setup_method(self):
        self.validator = DataValidator(report_path=Path("/tmp/test_report.json"))

    def test_good_record_is_valid(self):
        result = self.validator._validate_record(_good_record(), set(), {})
        assert result.is_valid is True
        assert result.issues == []

    def test_missing_name_is_error(self):
        record = _good_record(name=None)
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.MISSING_NAME in result.issues

    def test_missing_website_is_error(self):
        record = _good_record(website=None)
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.MISSING_WEBSITE in result.issues

    def test_missing_description_is_error(self):
        record = _good_record(description=None)
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.MISSING_DESCRIPTION in result.issues

    def test_missing_industry_is_error(self):
        record = _good_record(industry=None)
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.MISSING_INDUSTRY in result.issues

    def test_short_description_is_warning(self):
        record = _good_record(description="Too short")
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.DESCRIPTION_TOO_SHORT in result.warnings

    def test_invalid_email_is_warning(self):
        record = _good_record(email="notanemail")
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.INVALID_EMAIL in result.warnings

    def test_invalid_phone_is_warning(self):
        record = _good_record(phone="12")
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.INVALID_PHONE in result.warnings

    def test_invalid_ceo_is_warning(self):
        record = _good_record(ceo="12345678")
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.INVALID_CEO in result.warnings

    def test_invalid_founder_is_warning(self):
        record = _good_record(founder="https://ceo.com")
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.INVALID_FOUNDER in result.warnings

    def test_low_confidence_is_warning(self):
        record = _good_record(confidence_score="5")
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.LOW_CONFIDENCE in result.warnings

    def test_duplicate_domain_is_warning(self):
        record = _good_record()
        dup_domains = {"acme.com"}
        result = self.validator._validate_record(record, dup_domains, {})
        assert ValidationCode.DUPLICATE_DOMAIN in result.warnings

    def test_not_found_status_is_warning(self):
        record = _good_record(status="NOT_FOUND")
        result = self.validator._validate_record(record, set(), {})
        assert ValidationCode.NOT_FOUND_STATUS in result.warnings


# ---------------------------------------------------------------------------
# DataValidator.validate (integration)
# ---------------------------------------------------------------------------

class TestDataValidatorValidate:
    def setup_method(self, tmp_path_factory):
        self.validator = DataValidator(report_path=Path("/tmp/test_report.json"))

    def test_adds_validation_columns(self):
        df = _make_df([_good_record()])
        validated_df, report = self.validator.validate(df)
        assert "_validation_issues" in validated_df.columns
        assert "_validation_warnings" in validated_df.columns

    def test_report_counts_are_correct(self):
        rows = [
            _good_record(),                          # valid
            _good_record(name=None, website=None),   # invalid
        ]
        df = _make_df(rows)
        _, report = self.validator.validate(df)
        assert report.total_companies == 2
        assert report.total_valid == 1
        assert report.total_invalid == 1

    def test_duplicate_domain_detected(self):
        rows = [_good_record(), _good_record()]  # same domain: acme.com
        df = _make_df(rows)
        _, report = self.validator.validate(df)
        assert report.duplicate_domains_found >= 1

    def test_field_completeness_counted(self):
        df = _make_df([_good_record()])
        _, report = self.validator.validate(df)
        assert report.fields_with_name == 1
        assert report.fields_with_website == 1
        assert report.fields_with_ceo == 1

    def test_confidence_stats(self):
        rows = [_good_record(confidence_score="80"), _good_record(confidence_score="60")]
        df = _make_df(rows)
        _, report = self.validator.validate(df)
        assert report.average_confidence_score == 70.0
        assert report.min_confidence_score == 60
        assert report.max_confidence_score == 80


# ---------------------------------------------------------------------------
# ValidationReport serialisation
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_saves_json(self, tmp_path):
        report_path = tmp_path / "validation_report.json"
        validator = DataValidator(report_path=report_path)
        df = _make_df([_good_record()])
        _, report = validator.validate(df)
        saved_path = validator.save_report(report)
        assert saved_path.exists()
        with saved_path.open() as f:
            data = json.load(f)
        assert "total_companies" in data
        assert data["total_companies"] == 1

    def test_report_contains_issue_counts(self, tmp_path):
        report_path = tmp_path / "report.json"
        validator = DataValidator(report_path=report_path)
        df = _make_df([_good_record(name=None)])  # will have MISSING_NAME
        _, report = validator.save_report if False else (lambda: (None, None))()  # type: ignore
        _, report = validator.validate(df)
        validator.save_report(report)
        with report_path.open() as f:
            data = json.load(f)
        assert "issue_counts" in data
        assert ValidationCode.MISSING_NAME in data["issue_counts"]
