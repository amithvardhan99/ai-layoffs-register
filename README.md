# AI Layoffs Register — Streamlit Dashboard

Interactive dashboard for the **AI Layoffs Register (2023–2026)**. Charts and KPIs update from the live filter panel.

## Project layout

```
artifacts/                          ← project root (CSV files live here)
├── ai-layoffs-with-workforce-numbers.csv
├── notebook.ipynb
└── app/
    ├── streamlit_app.py            ← main application
    ├── requirements.txt
    └── README.md
```

## Run locally

```bash
cd path/to/artifacts/app

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Deploy on Render

1. Push this project to a Git repo.
2. On [Render](https://render.com): **New → Web Service**, connect the repo.
3. Settings:
   - **Root Directory:** `app` (or leave blank if the repo root is `app`)
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
4. Ensure `ai-layoffs-with-workforce-numbers.csv` is one level above `app/` **or** next to `streamlit_app.py` (the app checks both paths).

If the CSV sits at the repo root next to `app/`, Root Directory = `app` still works because the code resolves `../ai-layoffs-with-workforce-numbers.csv`.

## Features

- KPI strip — events, companies, roles counted / not counted, % explicit AI, date span
- Six insight sections aligned with the notebook roadmap
- Filter panel — Year, Sector, Country, Attribution, Evidence tier, Execution, Kind
- Recent events table with source links
- Restrained UI (teal accent, Inter, Plotly)

## Data

Uses `ai-layoffs-with-workforce-numbers.csv` (cleaned register from the notebook).
