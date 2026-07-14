# Company Data Enrichment System

## Overview

The **Company Data Enrichment System** is a production-oriented Python application that automatically cleans, standardizes, enriches, validates, and exports company information from multiple CSV files containing incomplete business records.

The system processes hundreds of company records, identifies duplicate organizations, enriches missing information using official websites and AI-assisted extraction, validates the results, and exports a clean dataset with confidence scores.

---

# Features

* Read multiple CSV files from a single input directory
* Automatically map different column names to a unified schema
* Merge duplicate companies using company name and domain matching
* Normalize company names, websites, and LinkedIn URLs
* Extract company information from official websites
* AI-powered extraction using OpenAI structured outputs
* Detect CEO and Founder when available
* Extract company description, industry, company size, contact details, and social links
* Confidence scoring based on source reliability
* Validation of URLs, phone numbers, and email addresses
* Generate validation reports
* Detailed logging and error handling
* Modular and scalable architecture

---

# Tech Stack

* Python 3.11+
* Pandas
* Requests
* BeautifulSoup4
* Pydantic
* OpenAI API
* python-dotenv
* Tenacity
* Logging

---

# Project Structure

```text
company_enrichment/
│
├── data/
│   ├── input/
│   └── output/
│
├── modules/
│   ├── cleaner.py
│   ├── confidence.py
│   ├── data_loader.py
│   ├── enrichment.py
│   ├── llm_extractor.py
│   ├── scraper.py
│   ├── validator.py
│   └── utils.py
│
├── prompts/
│   └── extraction_prompt.txt
│
├── tests/
│
├── config.py
├── main.py
├── requirements.txt
├── README.md
└── .env.example
```

---

# Workflow

```text
Input CSV Files
        │
        ▼
Data Loading
        │
        ▼
Data Cleaning & Normalization
        │
        ▼
Duplicate Detection
        │
        ▼
Website Scraping
        │
        ▼
LLM Information Extraction
        │
        ▼
Confidence Scoring
        │
        ▼
Validation
        │
        ▼
Export CSV + Validation Report
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/saumyap48/company-data-enrichment.git
cd company-data-enrichment/company_enrichment
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

---

# Usage

Place your CSV files inside:

```text
data/input/
```

Run the application:

```bash
python main.py
```

CLI options include:

* Process input files
* Run enrichment
* Validate data
* Export reports

---

# Input Schema

The application accepts incomplete company records.

Example:

| company_name | website        | linkedin                                |
| ------------ | -------------- | --------------------------------------- |
| Salesforce   | salesforce.com | https://linkedin.com/company/salesforce |
| Stripe       | stripe.com     |                                         |

Different input column names are automatically mapped to a standard schema.

---

# Output Schema

The enriched dataset contains fields such as:

* company_id
* name
* company_name
* website
* linkedin
* description
* industry
* size
* email
* phone
* facebook
* twitter
* instagram
* ceo
* founder
* confidence_score
* status
* last_updated

Output files are generated in:

```text
data/output/
```

Example:

* `enriched_companies.csv`
* `validation_report.json`

---

# Confidence Scoring

|  Score | Meaning                                   |
| -----: | ----------------------------------------- |
| 95–100 | Official website and verified information |
|  80–94 | Reliable public sources                   |
|  60–79 | Partial information available             |
|   0–59 | Limited or unverifiable information       |

---

# Status Values

* FOUND
* PARTIALLY_FOUND
* NEEDS_REVIEW
* NOT_FOUND

---

# Error Handling

The system is designed to continue processing even if individual companies fail.

It includes handling for:

* Invalid URLs
* Missing data
* Network failures
* Website redirects
* Rate limiting
* Parsing failures
* API failures

---

# Testing

Run the test suite:

```bash
pytest
```

---

# Future Improvements

* LinkedIn API integration
* Additional business data providers
* Async web scraping
* Incremental processing
* Docker support
* REST API interface
* Dashboard for enrichment statistics

---

# Disclaimer

This project only uses publicly available information and does not intentionally generate or fabricate company data. When information cannot be verified, the corresponding fields are left blank.

---

# Author

**Saumya Pandey**

GitHub: https://github.com/saumyap48
