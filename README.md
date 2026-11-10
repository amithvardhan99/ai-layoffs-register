# AI Layoffs Register — FastAPI Dashboard

Interactive web application for exploring the **AI Layoffs Register (2023–2026)** — every row sourced.

The dashboard surfaces the exploratory analysis from `notebook.ipynb` with live filters so every chart and KPI updates to the subset you select.

## Features

- **KPI strip** — events, companies, roles counted / not counted, share of explicit AI attribution, date span
- **Six insight sections** aligned with the notebook roadmap:
  1. Overall layoff statistics (timeline + top companies)
  2. Industry analysis (activity & affected headcount by sector)
  3. Geography (country activity + attribution by sector)
  4. AI attribution mix + job families
  5. Workforce impact (% reductions + execution status)
  6. Evidence quality (tiers and tier × attribution)
- **Filter panel** — Year, Sector, Country, Attribution, Evidence tier, Execution, Kind (multi-select)
- **Recent events table** — latest 25 matching rows with source links
- Clean, restrained UI (teal accent, Inter type, Plotly charts)

## Quick start

```bash
cd app
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Data is read from `../ai-layoffs-with-workforce-numbers.csv` (the cleaned register produced by the notebook).

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/insights` | JSON: stats, Plotly figures, event rows. Accepts query params: `sector`, `country`, `attribution`, `evidence_tier`, `execution`, `kind`, `year` (all repeatable) |
| `GET /api/health` | Health check |

## Project data

Uses the cleaned register columns documented in `DATASET_README.md` and the column glossary in the notebook / project style guide.
