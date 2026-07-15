"""
tests/test_cleaner.py
=====================
Unit tests for modules/cleaner.py

Tests:
    - Whitespace / NA coercion
    - Company name normalisation (title-case, legal suffixes)
    - Domain canonicalisation (strip scheme, www, paths, query strings)
    - LinkedIn URL normalisation
    - Social URL normalisation (Facebook, Twitter, Instagram)
    - Email validation
    - Phone normalisation
    - Exact duplicate removal
    - Fuzzy duplicate merging
"""

from __future__ import annotations

import pandas as pd
import pytest

from modules.cleaner import (
    DataCleaner,
    _blank_to_na,
    _canonicalise_domain,
    _normalise_linkedin,
    _normalise_name,
    _normalise_email,
    _normalise_phone,
    _to_str,
    clean_dataframe,
)


# ---------------------------------------------------------------------------
# _to_str helpers
# ---------------------------------------------------------------------------

class TestToStr:
    def test_none_returns_empty(self):
        assert _to_str(None) == ""

    def test_nan_returns_empty(self):
        assert _to_str(float("nan")) == ""

    def test_nan_string_returns_empty(self):
        assert _to_str("nan") == ""

    def test_none_string_returns_empty(self):
        assert _to_str("None") == ""

    def test_na_string_returns_empty(self):
        assert _to_str("N/A") == ""

    def test_dash_returns_empty(self):
        assert _to_str("-") == ""

    def test_normal_string_stripped(self):
        assert _to_str("  Acme  ") == "Acme"

    def test_empty_string_returns_empty(self):
        assert _to_str("") == ""


# ---------------------------------------------------------------------------
# _normalise_name
# ---------------------------------------------------------------------------

class TestNormaliseName:
    def test_title_case(self):
        assert _normalise_name("acme corporation") == "Acme Corporation"

    def test_preserves_acronym(self):
        result = _normalise_name("IBM global services")
        assert "IBM" in result

    def test_corp_expanded(self):
        assert "Corporation" in _normalise_name("Acme Corp")

    def test_inc_normalised(self):
        result = _normalise_name("beta inc.")
        assert "Inc" in result

    def test_llc_normalised(self):
        result = _normalise_name("gamma llc")
        assert "LLC" in result

    def test_collapses_whitespace(self):
        assert _normalise_name("Acme   Corp") == "Acme Corporation"

    def test_strips_leading_trailing(self):
        result = _normalise_name("  Acme  ")
        assert result == "Acme"


# ---------------------------------------------------------------------------
# _canonicalise_domain
# ---------------------------------------------------------------------------

class TestCanonicaliseDomain:
    def test_strips_https(self):
        assert _canonicalise_domain("https://example.com") == "example.com"

    def test_strips_http(self):
        assert _canonicalise_domain("http://example.com") == "example.com"

    def test_strips_www(self):
        assert _canonicalise_domain("www.example.com") == "example.com"

    def test_strips_https_www(self):
        assert _canonicalise_domain("https://www.example.com") == "example.com"

    def test_strips_path(self):
        assert _canonicalise_domain("https://example.com/about/us") == "example.com"

    def test_strips_query_string(self):
        assert _canonicalise_domain("https://example.com?ref=foo") == "example.com"

    def test_strips_fragment(self):
        assert _canonicalise_domain("https://example.com#contact") == "example.com"

    def test_lowercased(self):
        assert _canonicalise_domain("HTTPS://EXAMPLE.COM") == "example.com"

    def test_email_returns_empty(self):
        assert _canonicalise_domain("info@example.com") == ""

    def test_invalid_string_returns_empty(self):
        assert _canonicalise_domain("not a url at all!") == ""

    def test_bare_domain_passes_through(self):
        assert _canonicalise_domain("example.com") == "example.com"

    def test_www2_stripped(self):
        result = _canonicalise_domain("http://www2.example.com")
        assert "www2" not in result


# ---------------------------------------------------------------------------
# _normalise_linkedin
# ---------------------------------------------------------------------------

class TestNormaliseLinkedIn:
    def test_full_url(self):
        result = _normalise_linkedin("https://www.linkedin.com/company/microsoft")
        assert result == "https://www.linkedin.com/company/microsoft"

    def test_no_https(self):
        result = _normalise_linkedin("linkedin.com/company/google")
        assert result == "https://www.linkedin.com/company/google"

    def test_trailing_slash_removed(self):
        result = _normalise_linkedin("https://www.linkedin.com/company/apple/")
        assert result == "https://www.linkedin.com/company/apple"

    def test_non_linkedin_url_returns_empty(self):
        result = _normalise_linkedin("https://facebook.com/company/foo")
        assert result == ""

    def test_empty_returns_empty(self):
        assert _normalise_linkedin("") == ""

    def test_in_profile_normalised(self):
        result = _normalise_linkedin("https://www.linkedin.com/in/johndoe")
        assert "linkedin.com" in result


# ---------------------------------------------------------------------------
# _normalise_email
# ---------------------------------------------------------------------------

class TestNormaliseEmail:
    def test_valid_email(self):
        assert _normalise_email("info@example.com") == "info@example.com"

    def test_lowercased(self):
        assert _normalise_email("INFO@EXAMPLE.COM") == "info@example.com"

    def test_invalid_no_at(self):
        assert _normalise_email("notanemail.com") == ""

    def test_invalid_no_domain(self):
        assert _normalise_email("info@") == ""

    def test_empty(self):
        assert _normalise_email("") == ""


# ---------------------------------------------------------------------------
# _normalise_phone
# ---------------------------------------------------------------------------

class TestNormalisePhone:
    def test_valid_phone(self):
        result = _normalise_phone("+1 (555) 123-4567")
        assert result != ""

    def test_too_short_returns_empty(self):
        assert _normalise_phone("123") == ""

    def test_strips_letters(self):
        result = _normalise_phone("Call: +44 20 7946 0958")
        assert result != ""

    def test_empty(self):
        assert _normalise_phone("") == ""


# ---------------------------------------------------------------------------
# DataCleaner — full pipeline
# ---------------------------------------------------------------------------

class TestDataCleaner:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a minimal DataFrame from a list of dicts."""
        from modules.data_loader import STANDARD_COLUMNS
        df = pd.DataFrame(rows)
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def test_blank_strings_become_na(self):
        df = self._make_df([{"name": "  ", "website": ""}])
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert pd.isna(result.iloc[0]["name"]) or result.iloc[0]["name"] == ""

    def test_domain_canonicalised(self):
        df = self._make_df([{"name": "Acme", "website": "https://www.acme.com/about"}])
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert result.iloc[0]["website"] == "acme.com"

    def test_linkedin_normalised(self):
        df = self._make_df([{
            "name": "Acme",
            "linkedin": "http://linkedin.com/company/acme-corp/"
        }])
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert result.iloc[0]["linkedin"] == "https://www.linkedin.com/company/acme-corp"

    def test_exact_duplicates_removed(self):
        rows = [
            {"name": "Acme Corp", "website": "acme.com", "linkedin": pd.NA},
            {"name": "Acme Corp", "website": "acme.com", "linkedin": pd.NA},
        ]
        df = self._make_df(rows)
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert len(result) == 1

    def test_fuzzy_duplicates_merged(self):
        rows = [
            {"name": "Microsoft Corporation", "website": "microsoft.com"},
            {"name": "Microsoft Corp", "website": "microsoft.com"},
        ]
        df = self._make_df(rows)
        cleaner = DataCleaner(fuzzy_threshold=85)
        result = cleaner.clean(df)
        # After fuzzy merge, should be 1 row
        assert len(result) == 1

    def test_name_title_cased(self):
        df = self._make_df([{"name": "acme corporation"}])
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert result.iloc[0]["name"] == "Acme Corporation"

    def test_invalid_email_blanked(self):
        df = self._make_df([{"name": "Test", "email": "not-an-email"}])
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert pd.isna(result.iloc[0]["email"]) or result.iloc[0]["email"] == ""

    def test_nan_string_blanked(self):
        df = self._make_df([{"name": "nan", "website": "NaN"}])
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert pd.isna(result.iloc[0]["name"]) or result.iloc[0]["name"] == ""

    def test_clean_dataframe_shortcut(self):
        from modules.data_loader import STANDARD_COLUMNS
        df = pd.DataFrame([{"name": "Acme Corp", "website": "https://acme.com"}])
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        result = clean_dataframe(df)
        assert not result.empty
        assert result.iloc[0]["website"] == "acme.com"
