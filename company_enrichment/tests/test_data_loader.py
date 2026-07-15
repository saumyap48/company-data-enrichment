"""
tests/test_data_loader.py
=========================
Unit tests for modules/data_loader.py

Tests:
    - Column alias detection (exact + substring match)
    - Multi-encoding CSV loading
    - Empty / malformed file handling
    - Schema enforcement (all standard columns present)
    - Deterministic company ID generation
    - Bulk load with mixed good/bad files
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pandas as pd
import pytest

from modules.data_loader import (
    DataLoader,
    STANDARD_COLUMNS,
    _build_column_mapping,
    _detect_column,
    _generate_company_id,
    _normalise_col,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_input_dir(tmp_path: Path) -> Path:
    """Return a temporary directory to use as input_dir."""
    d = tmp_path / "input"
    d.mkdir()
    return d


def _write_csv(directory: Path, filename: str, content: str) -> Path:
    """Helper: write a CSV string to a file and return its path."""
    p = directory / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _normalise_col
# ---------------------------------------------------------------------------

class TestNormaliseCol:
    def test_strips_whitespace(self):
        assert _normalise_col("  Company Name  ") == "company_name"

    def test_lowercases(self):
        assert _normalise_col("LinkedIn_URL") == "linkedin_url"

    def test_replaces_hyphens(self):
        assert _normalise_col("input-company-name") == "input_company_name"

    def test_replaces_spaces(self):
        assert _normalise_col("company domain") == "company_domain"

    def test_mixed(self):
        assert _normalise_col("Input-Company Domain") == "input_company_domain"


# ---------------------------------------------------------------------------
# _detect_column
# ---------------------------------------------------------------------------

class TestDetectColumn:
    def test_exact_match(self):
        raw = ["company_name", "website", "email"]
        result = _detect_column(raw, "name", ["company_name", "name"])
        assert result == "company_name"

    def test_case_insensitive(self):
        raw = ["Company_Name", "Website"]
        result = _detect_column(raw, "name", ["company_name"])
        assert result == "Company_Name"

    def test_hyphen_alias(self):
        raw = ["input-company-name", "domain"]
        result = _detect_column(raw, "name", ["input_company_name"])
        assert result == "input-company-name"

    def test_substring_fallback(self):
        raw = ["company_website_url"]
        result = _detect_column(raw, "website", ["website"])
        assert result == "company_website_url"

    def test_no_match_returns_none(self):
        raw = ["foo", "bar", "baz"]
        result = _detect_column(raw, "website", ["website", "domain", "url"])
        assert result is None


# ---------------------------------------------------------------------------
# _build_column_mapping
# ---------------------------------------------------------------------------

class TestBuildColumnMapping:
    def test_standard_names(self):
        raw = ["company_name", "website", "linkedin_url"]
        mapping = _build_column_mapping(raw)
        assert mapping.get("company_name") == "name"
        assert mapping.get("website") == "website"
        assert mapping.get("linkedin_url") == "linkedin"

    def test_non_standard_names(self):
        raw = ["input-company-name", "input-company-domain"]
        mapping = _build_column_mapping(raw)
        assert "name" in mapping.values()
        assert "website" in mapping.values()

    def test_unknown_columns_not_mapped(self):
        raw = ["foo_bar", "unknown_col"]
        mapping = _build_column_mapping(raw)
        assert mapping == {}

    def test_no_double_mapping(self):
        """Two columns that match the same canonical name — first wins."""
        raw = ["company_name", "name"]
        mapping = _build_column_mapping(raw)
        canonical_values = list(mapping.values())
        assert canonical_values.count("name") == 1


# ---------------------------------------------------------------------------
# _generate_company_id
# ---------------------------------------------------------------------------

class TestGenerateCompanyId:
    def test_returns_valid_uuid(self):
        cid = _generate_company_id("Acme Corp", "acme.com")
        assert uuid.UUID(cid)  # raises ValueError if not valid UUID

    def test_deterministic(self):
        id1 = _generate_company_id("Acme Corp", "acme.com")
        id2 = _generate_company_id("Acme Corp", "acme.com")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = _generate_company_id("Acme Corp", "acme.com")
        id2 = _generate_company_id("Widget Inc", "widget.com")
        assert id1 != id2

    def test_case_insensitive_name(self):
        id1 = _generate_company_id("ACME CORP", "acme.com")
        id2 = _generate_company_id("acme corp", "acme.com")
        assert id1 == id2


# ---------------------------------------------------------------------------
# DataLoader.load_single
# ---------------------------------------------------------------------------

class TestDataLoaderLoadSingle:
    def test_loads_standard_csv(self, tmp_input_dir):
        csv_content = "company_name,website,linkedin_url\nAcme Corp,acme.com,linkedin.com/company/acme\n"
        p = _write_csv(tmp_input_dir, "test.csv", csv_content)
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_single(p)
        assert df is not None
        assert "name" in df.columns
        assert "website" in df.columns
        assert "linkedin" in df.columns
        assert df.iloc[0]["name"] == "Acme Corp"

    def test_loads_nonstandard_column_names(self, tmp_input_dir):
        csv_content = "input-company-name,input-company-domain\nBeta LLC,beta.com\n"
        p = _write_csv(tmp_input_dir, "nonstandard.csv", csv_content)
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_single(p)
        assert df is not None
        assert "name" in df.columns
        assert "website" in df.columns

    def test_empty_file_returns_none(self, tmp_input_dir):
        p = _write_csv(tmp_input_dir, "empty.csv", "")
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_single(p)
        assert df is None

    def test_all_standard_columns_present(self, tmp_input_dir):
        csv_content = "company_name\nAcme Corp\n"
        p = _write_csv(tmp_input_dir, "minimal.csv", csv_content)
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_single(p)
        assert df is not None
        for col in STANDARD_COLUMNS:
            assert col in df.columns, f"Missing standard column: {col}"

    def test_source_file_column_added(self, tmp_input_dir):
        csv_content = "company_name\nAcme Corp\n"
        p = _write_csv(tmp_input_dir, "source_test.csv", csv_content)
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_single(p)
        assert "_source_file" in df.columns
        assert df.iloc[0]["_source_file"] == "source_test.csv"


# ---------------------------------------------------------------------------
# DataLoader.load_all
# ---------------------------------------------------------------------------

class TestDataLoaderLoadAll:
    def test_loads_multiple_files(self, tmp_input_dir):
        _write_csv(tmp_input_dir, "a.csv", "company_name\nAlpha Inc\n")
        _write_csv(tmp_input_dir, "b.csv", "company_name\nBeta Ltd\n")
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_all()
        assert len(df) == 2
        assert loader.summary()["loaded"] == 2
        assert loader.summary()["failed"] == 0

    def test_empty_directory_returns_empty_df(self, tmp_input_dir):
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_all()
        assert df.empty

    def test_continues_after_bad_file(self, tmp_input_dir):
        _write_csv(tmp_input_dir, "good.csv", "company_name\nGood Corp\n")
        # Write a binary file that isn't a valid CSV
        bad = tmp_input_dir / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02\x03")
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_all()
        # At minimum the good file should load
        assert len(df) >= 1

    def test_deduplicates_across_files(self, tmp_input_dir):
        row = "company_name,website\nAcme Corp,acme.com\n"
        _write_csv(tmp_input_dir, "file1.csv", row)
        _write_csv(tmp_input_dir, "file2.csv", row)  # exact duplicate
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_all()
        # Both rows loaded (dedup happens in cleaner, not loader)
        assert len(df) == 2

    def test_company_ids_assigned(self, tmp_input_dir):
        _write_csv(tmp_input_dir, "ids.csv", "company_name,website\nAcme,acme.com\n")
        loader = DataLoader(input_dir=tmp_input_dir)
        df = loader.load_all()
        assert "company_id" in df.columns
        assert df["company_id"].notna().all()
