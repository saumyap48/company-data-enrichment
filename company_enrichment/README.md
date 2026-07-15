# Company Data Enrichment System

A production-quality, highly scalable AI-powered pipeline to clean, enrich, validate, and export company data from incomplete CSV files.

## Features
- **Auto-Detection & Cleaning**: Automatically maps messy column names (e.g. `input-company-name` to `name`), deduplicates exact/fuzzy matches, and canonicalises domains.
- **Web Scraping**: Built-in scraper with `robots.txt` compliance, user-agent rotation, retry/backoff, redirect detection, and connection pooling.
- **AI Extraction**: Uses OpenAI (GPT-4o) with strict Pydantic structured output to extract `industry`, `description`, `ceo`, `founder`, and `size` directly from scraped text without hallucinations.
- **Confidence Scoring**: Assigns a `0-100` score and status (`FOUND`, `NEEDS_REVIEW`, etc.) based on data sources and critical field presence.
- **Post-Validation**: Checks for missing fields, domain mismatches, and data sanity, producing a comprehensive `validation_report.json`.

## Quick Start
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY (optional, falls back to scraping only if omitted)
   ```
3. Add your CSVs to `data/input/`.
4. Run the interactive CLI:
   ```bash
   python main.py
   ```

## Pipeline Architecture
`DataLoader` ➔ `DataCleaner` ➔ `WebScraper` & `LLMExtractor` ➔ `ConfidenceScorer` ➔ `DataValidator`

## Output
Results are saved to `data/output/enriched_companies.csv` and `data/output/validation_report.json`.
