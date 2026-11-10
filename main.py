"""
AI Layoffs Register — Interactive Dashboard
FastAPI application surfacing EDA insights from the sourced register (2023–2026).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.utils import PlotlyJSONEncoder
import json

# ---------------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "ai-layoffs-with-workforce-numbers.csv"

app = FastAPI(
    title="AI Layoffs Register",
    description="Interactive dashboard for AI-attributed workforce reductions (2023–2026)",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    # Normalise country for filtering (strip parenthetical notes)
    df["Country_clean"] = (
        df["Country"]
        .astype(str)
        .str.split("(")
        .str[0]
        .str.strip()
        .str.replace(r"\s+and\s+", ", ", regex=True)
    )
    # Roles counted as numeric
    df["Roles counted"] = pd.to_numeric(df["Roles counted"], errors="coerce")
    df["% of workforce"] = pd.to_numeric(df["% of workforce"], errors="coerce").fillna(0)
    df["Roles reported (not counted)"] = pd.to_numeric(
        df["Roles reported (not counted)"], errors="coerce"
    )
    return df


DF = load_data()

# Pre-compute filter options
FILTER_OPTIONS = {
    "sectors": sorted(DF["Sector"].dropna().unique().tolist()),
    "countries": sorted(
        {c.strip() for row in DF["Country_clean"].dropna() for c in str(row).split(",") if c.strip()}
    ),
    "attributions": sorted(DF["Attribution"].dropna().unique().tolist()),
    "evidence_tiers": sorted(DF["Evidence tier"].dropna().unique().tolist()),
    "executions": sorted(DF["Execution"].dropna().unique().tolist()),
    "kinds": sorted(DF["Kind"].dropna().unique().tolist()),
    "years": sorted(DF["Year"].dropna().unique().astype(int).tolist()),
}


# ---------------------------------------------------------------------------
# Filtering helper
# ---------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    sector: Optional[list[str]] = None,
    country: Optional[list[str]] = None,
    attribution: Optional[list[str]] = None,
    evidence_tier: Optional[list[str]] = None,
    execution: Optional[list[str]] = None,
    kind: Optional[list[str]] = None,
    year: Optional[list[int]] = None,
) -> pd.DataFrame:
    out = df.copy()
    if sector:
        out = out[out["Sector"].isin(sector)]
    if attribution:
        out = out[out["Attribution"].isin(attribution)]
    if evidence_tier:
        out = out[out["Evidence tier"].isin(evidence_tier)]
    if execution:
        out = out[out["Execution"].isin(execution)]
    if kind:
        out = out[out["Kind"].isin(kind)]
    if year:
        out = out[out["Year"].isin(year)]
    if country:
        mask = out["Country_clean"].apply(
            lambda x: any(c.strip() in [p.strip() for p in str(x).split(",")] for c in country)
        )
        out = out[mask]
    return out


def _empty_fig(msg: str = "No data for current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=280,
    )
    return fig


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
    margin=dict(l=48, r=24, t=40, b=48),
    colorway=[
        "#0f766e",
        "#0369a1",
        "#7c3aed",
        "#c2410c",
        "#be123c",
        "#a16207",
        "#4338ca",
        "#0e7490",
    ],
)


def fig_to_json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


# ---------------------------------------------------------------------------
# Insight builders (mirror notebook EDA)
# ---------------------------------------------------------------------------

def summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    counted = df["Roles counted"].fillna(0).sum()
    reported_nc = df["Roles reported (not counted)"].fillna(0).sum()
    events = len(df)
    companies = df["Company"].nunique()
    explicit = int((df["Attribution"] == "explicit").sum())
    return {
        "events": events,
        "companies": companies,
        "roles_counted": int(counted),
        "roles_reported_not_counted": int(reported_nc),
        "explicit_pct": round(100 * explicit / events, 1) if events else 0,
        "date_min": df["Date"].min().strftime("%Y-%m-%d") if events and df["Date"].notna().any() else "—",
        "date_max": df["Date"].max().strftime("%Y-%m-%d") if events and df["Date"].notna().any() else "—",
    }


def chart_timeline(df: pd.DataFrame) -> go.Figure:
    if df.empty or df["Date"].isna().all():
        return _empty_fig()
    daily = df.groupby(df["Date"].dt.to_period("M")).size().reset_index(name="Events")
    daily["Date"] = daily["Date"].astype(str)
    daily = daily.sort_values("Date")
    daily["Cumulative"] = daily["Events"].cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["Date"],
            y=daily["Events"],
            name="Events / month",
            marker_color="#0f766e",
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Cumulative"],
            name="Cumulative",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#7c3aed", width=2),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Layoff events over time", font=dict(size=14)),
        yaxis=dict(title="Events", gridcolor="#e2e8f0"),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="", tickangle=-40, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=340,
        barmode="overlay",
    )
    return fig


def chart_top_companies(df: pd.DataFrame, n: int = 12) -> go.Figure:
    if df.empty:
        return _empty_fig()
    top = (
        df.groupby("Company")["Roles counted"]
        .sum()
        .fillna(0)
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top = top[top["Roles counted"] > 0]
    if top.empty:
        return _empty_fig("No counted roles for current filters")
    fig = px.bar(
        top,
        x="Roles counted",
        y="Company",
        orientation="h",
        text="Roles counted",
        color_discrete_sequence=["#0f766e"],
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Companies with highest counted roles", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Roles counted", gridcolor="#e2e8f0"),
        height=max(280, 28 * len(top) + 80),
        showlegend=False,
    )
    return fig


def chart_sector_activity(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    counts = df["Sector"].value_counts().reset_index()
    counts.columns = ["Sector", "Events"]
    fig = px.bar(
        counts,
        x="Events",
        y="Sector",
        orientation="h",
        text="Events",
        color_discrete_sequence=["#0369a1"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Layoff activity by sector", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Number of events", gridcolor="#e2e8f0"),
        height=max(280, 26 * len(counts) + 80),
        showlegend=False,
    )
    return fig


def chart_sector_roles(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    roles = (
        df.groupby("Sector")["Roles counted"]
        .sum()
        .fillna(0)
        .sort_values(ascending=False)
        .reset_index()
    )
    roles = roles[roles["Roles counted"] > 0]
    if roles.empty:
        return _empty_fig("No counted roles")
    fig = px.bar(
        roles,
        x="Roles counted",
        y="Sector",
        orientation="h",
        text="Roles counted",
        color_discrete_sequence=["#7c3aed"],
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Affected employees by sector (counted)", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Roles counted", gridcolor="#e2e8f0"),
        height=max(280, 26 * len(roles) + 80),
        showlegend=False,
    )
    return fig


def chart_country_activity(df: pd.DataFrame, n: int = 15) -> go.Figure:
    if df.empty:
        return _empty_fig()
    # Expand multi-country rows
    rows = []
    for _, r in df.iterrows():
        countries = [c.strip() for c in str(r["Country_clean"]).split(",") if c.strip()]
        for c in countries:
            rows.append(c)
    if not rows:
        return _empty_fig()
    counts = pd.Series(rows).value_counts().head(n).reset_index()
    counts.columns = ["Country", "Events"]
    fig = px.bar(
        counts,
        x="Events",
        y="Country",
        orientation="h",
        text="Events",
        color_discrete_sequence=["#0e7490"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Layoff activity by country", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Number of events", gridcolor="#e2e8f0"),
        height=max(280, 26 * len(counts) + 80),
        showlegend=False,
    )
    return fig


def chart_attribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    counts = df["Attribution"].value_counts().reset_index()
    counts.columns = ["Attribution", "Events"]
    colors = {"explicit": "#0f766e", "mixed": "#0369a1", "blamed": "#c2410c"}
    fig = px.pie(
        counts,
        names="Attribution",
        values="Events",
        color="Attribution",
        color_discrete_map=colors,
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="AI attribution mix", font=dict(size=14)),
        height=320,
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def chart_attribution_by_sector(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    ct = pd.crosstab(df["Sector"], df["Attribution"])
    if ct.empty:
        return _empty_fig()
    # Prefer explicit share
    if "explicit" in ct.columns:
        order = (ct["explicit"] / ct.sum(axis=1)).sort_values(ascending=False).index
        ct = ct.reindex(order)
    fig = go.Figure()
    palette = {"explicit": "#0f766e", "mixed": "#0369a1", "blamed": "#c2410c"}
    for col in ["explicit", "mixed", "blamed"]:
        if col in ct.columns:
            fig.add_trace(
                go.Bar(
                    name=col,
                    y=ct.index,
                    x=ct[col],
                    orientation="h",
                    marker_color=palette.get(col, "#64748b"),
                )
            )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Attribution by sector", font=dict(size=14)),
        barmode="stack",
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Events", gridcolor="#e2e8f0"),
        height=max(300, 26 * len(ct) + 90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def chart_workforce_pct(df: pd.DataFrame, n: int = 10) -> go.Figure:
    if df.empty:
        return _empty_fig()
    top = (
        df[df["% of workforce"] > 0]
        .sort_values("% of workforce", ascending=False)
        .drop_duplicates(subset=["Company"])
        .head(n)[["Company", "% of workforce"]]
    )
    if top.empty:
        return _empty_fig("No workforce % figures for current filters")
    fig = px.bar(
        top,
        x="% of workforce",
        y="Company",
        orientation="h",
        text="% of workforce",
        color_discrete_sequence=["#be123c"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Highest workforce reduction %", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="% of workforce", gridcolor="#e2e8f0"),
        height=max(280, 28 * len(top) + 80),
        showlegend=False,
    )
    return fig


def chart_execution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    counts = df["Execution"].value_counts().reset_index()
    counts.columns = ["Execution", "Events"]
    fig = px.bar(
        counts,
        x="Events",
        y="Execution",
        orientation="h",
        text="Events",
        color_discrete_sequence=["#a16207"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Execution status", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Events", gridcolor="#e2e8f0"),
        height=280,
        showlegend=False,
    )
    return fig


def chart_evidence_tier(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    counts = df["Evidence tier"].value_counts().reindex(["Tier 1", "Tier 2", "Tier 3"]).dropna().reset_index()
    counts.columns = ["Evidence tier", "Events"]
    fig = px.bar(
        counts,
        x="Events",
        y="Evidence tier",
        orientation="h",
        text="Events",
        color_discrete_sequence=["#4338ca"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Evidence tier distribution", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Events", gridcolor="#e2e8f0"),
        height=260,
        showlegend=False,
    )
    return fig


def chart_evidence_vs_attribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    ct = pd.crosstab(df["Evidence tier"], df["Attribution"])
    order = [t for t in ["Tier 1", "Tier 2", "Tier 3"] if t in ct.index]
    ct = ct.reindex(order)
    fig = go.Figure()
    palette = {"explicit": "#0f766e", "mixed": "#0369a1", "blamed": "#c2410c"}
    for col in ["explicit", "mixed", "blamed"]:
        if col in ct.columns:
            fig.add_trace(
                go.Bar(
                    name=col,
                    y=ct.index,
                    x=ct[col],
                    orientation="h",
                    marker_color=palette.get(col, "#64748b"),
                )
            )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Evidence quality × AI attribution", font=dict(size=14)),
        barmode="group",
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Events", gridcolor="#e2e8f0"),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def chart_job_families(df: pd.DataFrame, n: int = 12) -> go.Figure:
    if df.empty or "Common professions" not in df.columns:
        return _empty_fig()
    items = []
    for val in df["Common professions"].dropna():
        for p in str(val).split(";"):
            p = p.strip()
            if p and p.lower() != "not specified":
                items.append(p)
    if not items:
        return _empty_fig("No job-family detail for current filters")
    counts = pd.Series(items).value_counts().head(n).reset_index()
    counts.columns = ["Profession", "Count"]
    fig = px.bar(
        counts,
        x="Count",
        y="Profession",
        orientation="h",
        text="Count",
        color_discrete_sequence=["#0f766e"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Most frequently affected job families", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Mentions across events", gridcolor="#e2e8f0"),
        height=max(280, 28 * len(counts) + 80),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "filters": FILTER_OPTIONS,
            "stats": summary_stats(DF),
        },
    )


@app.get("/api/insights")
async def api_insights(
    sector: Optional[list[str]] = Query(None),
    country: Optional[list[str]] = Query(None),
    attribution: Optional[list[str]] = Query(None),
    evidence_tier: Optional[list[str]] = Query(None),
    execution: Optional[list[str]] = Query(None),
    kind: Optional[list[str]] = Query(None),
    year: Optional[list[int]] = Query(None),
):
    df = apply_filters(
        DF,
        sector=sector,
        country=country,
        attribution=attribution,
        evidence_tier=evidence_tier,
        execution=execution,
        kind=kind,
        year=year,
    )
    stats = summary_stats(df)

    charts = {
        "timeline": fig_to_json(chart_timeline(df)),
        "top_companies": fig_to_json(chart_top_companies(df)),
        "sector_activity": fig_to_json(chart_sector_activity(df)),
        "sector_roles": fig_to_json(chart_sector_roles(df)),
        "country": fig_to_json(chart_country_activity(df)),
        "attribution": fig_to_json(chart_attribution(df)),
        "attribution_sector": fig_to_json(chart_attribution_by_sector(df)),
        "workforce_pct": fig_to_json(chart_workforce_pct(df)),
        "execution": fig_to_json(chart_execution(df)),
        "evidence_tier": fig_to_json(chart_evidence_tier(df)),
        "evidence_attr": fig_to_json(chart_evidence_vs_attribution(df)),
        "job_families": fig_to_json(chart_job_families(df)),
    }

    # Recent events table (top 25 by date)
    table_df = df.sort_values("Date", ascending=False).head(25)
    events = []
    for _, r in table_df.iterrows():
        events.append(
            {
                "company": r["Company"],
                "date": r["Date"].strftime("%Y-%m-%d") if pd.notna(r["Date"]) else "",
                "sector": r.get("Sector", ""),
                "country": r.get("Country", ""),
                "roles_counted": None if pd.isna(r["Roles counted"]) else int(r["Roles counted"]),
                "attribution": r.get("Attribution", ""),
                "evidence_tier": r.get("Evidence tier", ""),
                "execution": r.get("Execution", ""),
                "kind": r.get("Kind", ""),
                "claim": (r.get("Claim") or "")[:220],
                "source_url": r.get("Source URL", ""),
                "event_id": r.get("Event ID", ""),
            }
        )

    return JSONResponse({"stats": stats, "charts": charts, "events": events})


@app.get("/api/health")
async def health():
    return {"status": "ok", "rows": len(DF)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
