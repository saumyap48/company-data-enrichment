"""
tests/test_confidence.py
========================
Unit tests for modules/confidence.py

Tests:
    - Source inference from source_used field
    - Base score per data source
    - Completeness bonus calculation
    - Flag detection (name/domain mismatch, short description, suspicious CEO)
    - Status assignment (FOUND, PARTIALLY_FOUND, NOT_FOUND, NEEDS_REVIEW)
    - DataFrame-level scoring
    - Edge cases (all empty record, perfect record)
"""

from __future__ import annotations

import pandas as pd
import pytest

from modules.confidence import (
    ConfidenceScorer,
    DataSource,
    EnrichmentStatus,
    ReviewFlag,
    _completeness_bonus,
    _count_present,
    _detect_flags,
    _determine_status,
    _infer_source,
    _present,
    score_record,
)


# ---------------------------------------------------------------------------
# _present
# ---------------------------------------------------------------------------

class TestPresent:
    def test_none_is_not_present(self):
        assert not _present(None)

    def test_nan_is_not_present(self):
        assert not _present(float("nan"))

    def test_empty_string_is_not_present(self):
        assert not _present("")

    def test_whitespace_is_not_present(self):
        assert not _present("   ")

    def test_na_string_is_not_present(self):
        assert not _present("N/A")

    def test_dash_is_not_present(self):
        assert not _present("-")

    def test_value_is_present(self):
        assert _present("Microsoft")

    def test_zero_is_present(self):
        assert _present("0")


# ---------------------------------------------------------------------------
# _infer_source
# ---------------------------------------------------------------------------

class TestInferSource:
    def test_official_website(self):
        record = {"source_used": "official_website"}
        assert _infer_source(record) == DataSource.OFFICIAL_WEBSITE

    def test_linkedin(self):
        record = {"source_used": "linkedin"}
        assert _infer_source(record) == DataSource.LINKEDIN

    def test_llm(self):
        record = {"source_used": "llm_inferred"}
        assert _infer_source(record) == DataSource.LLM_INFERRED

    def test_scrape(self):
        record = {"source_used": "web_scrape"}
        assert _infer_source(record) == DataSource.WEB_SCRAPE

    def test_empty_source_with_website(self):
        record = {"source_used": "", "website": "acme.com", "description": "We make things."}
        assert _infer_source(record) == DataSource.WEB_SCRAPE

    def test_empty_record_returns_not_found(self):
        record = {"source_used": ""}
        assert _infer_source(record) == DataSource.NOT_FOUND


# ---------------------------------------------------------------------------
# _completeness_bonus
# ---------------------------------------------------------------------------

class TestCompletenessBonus:
    def test_all_empty_gives_zero(self):
        record = {}
        bonus = _completeness_bonus(record)
        assert bonus == 0

    def test_all_filled_gives_max(self):
        record = {
            "ceo": "Jane Doe", "founder": "John Doe", "size": "100-500",
            "email": "a@b.com", "phone": "+1 555 1234",
            "linkedin": "https://linkedin.com/company/x",
            "facebook": "https://facebook.com/x",
            "twitter": "https://twitter.com/x",
            "instagram": "https://instagram.com/x",
        }
        bonus = _completeness_bonus(record)
        assert bonus > 0

    def test_partial_fill_gives_partial_bonus(self):
        record = {"ceo": "Jane Doe", "email": "a@b.com"}
        bonus = _completeness_bonus(record)
        assert 0 < bonus < 5


# ---------------------------------------------------------------------------
# _detect_flags
# ---------------------------------------------------------------------------

class TestDetectFlags:
    def test_score_too_low_flag(self):
        record = {}
        flags = _detect_flags(record, raw_score=10)
        assert ReviewFlag.SCORE_TOO_LOW in flags

    def test_description_too_short_flag(self):
        record = {"description": "Short desc"}
        flags = _detect_flags(record, raw_score=80)
        assert ReviewFlag.DESCRIPTION_TOO_SHORT in flags

    def test_suspicious_ceo_digit_string(self):
        record = {"ceo": "1234567890"}
        flags = _detect_flags(record, raw_score=80)
        assert ReviewFlag.SUSPICIOUS_CEO in flags

    def test_suspicious_ceo_url(self):
        record = {"ceo": "https://ceo.com"}
        flags = _detect_flags(record, raw_score=80)
        assert ReviewFlag.SUSPICIOUS_CEO in flags

    def test_valid_ceo_no_flag(self):
        record = {"ceo": "Jane Smith"}
        flags = _detect_flags(record, raw_score=80)
        assert ReviewFlag.SUSPICIOUS_CEO not in flags

    def test_name_domain_mismatch_flag(self):
        record = {"name": "Microsoft Corporation", "website": "apple.com"}
        flags = _detect_flags(record, raw_score=80)
        assert ReviewFlag.NAME_DOMAIN_MISMATCH in flags

    def test_name_domain_match_no_flag(self):
        record = {"name": "Microsoft Corporation", "website": "microsoft.com"}
        flags = _detect_flags(record, raw_score=80)
        assert ReviewFlag.NAME_DOMAIN_MISMATCH not in flags

    def test_good_record_no_flags(self):
        record = {
            "name": "Acme Corp",
            "website": "acme.com",
            "description": "A great company that makes industrial products worldwide.",
            "ceo": "Jane Doe",
        }
        flags = _detect_flags(record, raw_score=90)
        assert not flags


# ---------------------------------------------------------------------------
# _determine_status
# ---------------------------------------------------------------------------

class TestDetermineStatus:
    def test_found_when_score_high_and_all_critical(self):
        record = {
            "name": "Acme", "website": "acme.com",
            "description": "Makes things.", "industry": "Manufacturing"
        }
        status = _determine_status(record, final_score=85, flags=[])
        assert status == EnrichmentStatus.FOUND

    def test_not_found_when_score_zero(self):
        record = {}
        status = _determine_status(record, final_score=0, flags=[])
        assert status == EnrichmentStatus.NOT_FOUND

    def test_partially_found_when_score_medium(self):
        record = {"name": "Acme", "website": "acme.com"}
        status = _determine_status(record, final_score=50, flags=[])
        assert status == EnrichmentStatus.PARTIALLY_FOUND

    def test_needs_review_when_flag_present(self):
        record = {"name": "Acme", "website": "acme.com"}
        status = _determine_status(
            record, final_score=85, flags=[ReviewFlag.NAME_DOMAIN_MISMATCH]
        )
        assert status == EnrichmentStatus.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# ConfidenceScorer.score (integration)
# ---------------------------------------------------------------------------

class TestConfidenceScorerScore:
    def setup_method(self):
        self.scorer = ConfidenceScorer()

    def test_perfect_record_gives_found(self):
        record = {
            "name": "Acme Corp",
            "website": "acme.com",
            "description": "A leader in industrial solutions for global markets.",
            "industry": "Manufacturing",
            "ceo": "Jane Smith",
            "founder": "John Smith",
            "source_used": "official_website",
            "confidence_score": "0",
            "status": "",
        }
        score, status, flags = self.scorer.score(record)
        assert score >= 80
        assert status in (EnrichmentStatus.FOUND, EnrichmentStatus.PARTIALLY_FOUND)

    def test_empty_record_gives_not_found(self):
        record = {}
        score, status, flags = self.scorer.score(record)
        assert status == EnrichmentStatus.NOT_FOUND
        assert score == 0

    def test_llm_source_gives_lower_base(self):
        record_llm = {"source_used": "llm_inferred", "name": "X", "website": "x.com",
                      "description": "A description here", "industry": "Tech"}
        record_web = {"source_used": "official_website", "name": "X", "website": "x.com",
                      "description": "A description here", "industry": "Tech"}
        score_llm, _, _ = self.scorer.score(record_llm)
        score_web, _, _ = self.scorer.score(record_web)
        assert score_web >= score_llm


# ---------------------------------------------------------------------------
# ConfidenceScorer.score_dataframe
# ---------------------------------------------------------------------------

class TestConfidenceScorerDataframe:
    def test_adds_score_and_status_columns(self):
        df = pd.DataFrame([
            {"name": "A", "website": "a.com", "description": "Desc A Industry B",
             "industry": "Tech", "source_used": "official_website",
             "confidence_score": pd.NA, "status": pd.NA},
        ])
        scorer = ConfidenceScorer()
        result = scorer.score_dataframe(df)
        assert "confidence_score" in result.columns
        assert "status" in result.columns
        assert "_review_flags" in result.columns

    def test_scores_all_rows(self):
        rows = [
            {"name": f"Company {i}", "website": f"company{i}.com",
             "source_used": "web_scrape",
             "confidence_score": pd.NA, "status": pd.NA}
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        scorer = ConfidenceScorer()
        result = scorer.score_dataframe(df)
        assert len(result) == 5
        assert result["confidence_score"].notna().all()


# ---------------------------------------------------------------------------
# Module-level score_record convenience function
# ---------------------------------------------------------------------------

class TestScoreRecord:
    def test_returns_tuple_of_three(self):
        record = {"name": "Acme", "source_used": "official_website"}
        result = score_record(record)
        assert len(result) == 3
        score, status_str, flags_list = result
        assert isinstance(score, int)
        assert isinstance(status_str, str)
        assert isinstance(flags_list, list)
