"""
AI Layoffs Register — Streamlit Dashboard
Interactive exploration of the sourced AI-attributed workforce reductions register (2023–2026).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Layoffs Register",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional light theme styling — high contrast for senior analytics audiences
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: Inter, system-ui, -apple-system, sans-serif;
    }

    /* Force readable text on main surface and sidebar */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }

    /* Multiselect / widget labels */
    label[data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px;
        padding: 0.9rem 1rem 0.75rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        line-height: 1.3 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    /* Allow date-span (and any long metric) to wrap inside the card */
    div[data-testid="stMetric"] {
        min-height: 4.4rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word;
    }

    /* Headings — always high contrast */
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 650 !important;
        letter-spacing: -0.02em;
    }
    h1 {
        font-size: 1.65rem !important;
        margin-bottom: 0.15rem !important;
    }
    h2 {
        font-size: 1.15rem !important;
        margin-top: 1.15rem !important;
        margin-bottom: 0.55rem !important;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #e2e8f0;
    }
    h3 {
        font-size: 1.02rem !important;
        color: #0f172a !important;
    }

    /* Captions and secondary text */
    [data-testid="stCaptionContainer"],
    .stCaption {
        color: #64748b !important;
    }
    p, span, li {
        color: #334155;
    }

    hr {
        margin: 0.9rem 0 1.1rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #ffffff;
    }

    /* Buttons */
    div[data-testid="stSidebar"] button {
        background: #0f766e !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSidebar"] button:hover {
        background: #0d9488 !important;
        color: #ffffff !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .footer-note {
        color: #64748b !important;
        font-size: 0.78rem;
        text-align: center;
        margin-top: 1.75rem;
        padding-top: 0.85rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Plotly containers sit on white cards */
    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.35rem 0.25rem 0.15rem 0.25rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Plotly defaults
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
    margin=dict(l=48, r=24, t=44, b=48),
    colorway=[
        "#0f766e", "#0369a1", "#7c3aed", "#c2410c",
        "#be123c", "#a16207", "#4338ca", "#0e7490",
    ],
)

COLOR_TEAL = "#0f766e"
COLOR_BLUE = "#0369a1"
COLOR_VIOLET = "#7c3aed"
COLOR_ROSE = "#be123c"
COLOR_AMBER = "#a16207"
COLOR_INDIGO = "#4338ca"
COLOR_CYAN = "#0e7490"

ATTR_COLORS = {"explicit": "#0f766e", "mixed": "#0369a1", "blamed": "#c2410c"}


def _empty_fig(msg: str = "No data for current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20), height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    path = base.parent / "ai-layoffs-with-workforce-numbers.csv"
    if not path.exists():
        path = base / "ai-layoffs-with-workforce-numbers.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Country_clean"] = (
        df["Country"].astype(str).str.split("(").str[0].str.strip()
        .str.replace(r"\s+and\s+", ", ", regex=True)
    )
    df["Roles counted"] = pd.to_numeric(df["Roles counted"], errors="coerce")
    df["% of workforce"] = pd.to_numeric(df["% of workforce"], errors="coerce").fillna(0)
    df["Roles reported (not counted)"] = pd.to_numeric(
        df["Roles reported (not counted)"], errors="coerce"
    )
    return df


def apply_filters(
    df: pd.DataFrame,
    years: list,
    sectors: list,
    countries: list,
    attributions: list,
    evidence_tiers: list,
    executions: list,
    kinds: list,
) -> pd.DataFrame:
    out = df.copy()
    if years:
        out = out[out["Year"].isin(years)]
    if sectors:
        out = out[out["Sector"].isin(sectors)]
    if attributions:
        out = out[out["Attribution"].isin(attributions)]
    if evidence_tiers:
        out = out[out["Evidence tier"].isin(evidence_tiers)]
    if executions:
        out = out[out["Execution"].isin(executions)]
    if kinds:
        out = out[out["Kind"].isin(kinds)]
    if countries:
        mask = out["Country_clean"].apply(
            lambda x: any(
                c.strip() in [p.strip() for p in str(x).split(",")]
                for c in countries
            )
        )
        out = out[mask]
    return out


def summary_stats(df: pd.DataFrame) -> dict:
    events = len(df)
    counted = int(df["Roles counted"].fillna(0).sum())
    reported_nc = int(df["Roles reported (not counted)"].fillna(0).sum())
    companies = int(df["Company"].nunique())
    explicit = int((df["Attribution"] == "explicit").sum())
    return {
        "events": events,
        "companies": companies,
        "roles_counted": counted,
        "roles_reported_not_counted": reported_nc,
        "explicit_pct": round(100 * explicit / events, 1) if events else 0.0,
        "date_min": df["Date"].min().strftime("%Y-%m-%d") if events and df["Date"].notna().any() else "—",
        "date_max": df["Date"].max().strftime("%Y-%m-%d") if events and df["Date"].notna().any() else "—",
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_timeline(df: pd.DataFrame) -> go.Figure:
    if df.empty or df["Date"].isna().all():
        return _empty_fig()
    daily = df.groupby(df["Date"].dt.to_period("M")).size().reset_index(name="Events")
    daily["Date"] = daily["Date"].astype(str)
    daily = daily.sort_values("Date")
    daily["Cumulative"] = daily["Events"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["Date"], y=daily["Events"], name="Events / month",
        marker_color=COLOR_TEAL, opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Cumulative"], name="Cumulative",
        yaxis="y2", mode="lines+markers",
        line=dict(color=COLOR_VIOLET, width=2), marker=dict(size=4),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Layoff events over time", font=dict(size=14)),
        yaxis=dict(title="Events", gridcolor="#e2e8f0"),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="", tickangle=-40, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=340, barmode="overlay",
    )
    return fig


def chart_top_companies(df: pd.DataFrame, n: int = 12) -> go.Figure:
    if df.empty:
        return _empty_fig()
    top = (
        df.groupby("Company")["Roles counted"].sum().fillna(0)
        .sort_values(ascending=False).head(n).reset_index()
    )
    top = top[top["Roles counted"] > 0]
    if top.empty:
        return _empty_fig("No counted roles for current filters")
    fig = px.bar(
        top, x="Roles counted", y="Company", orientation="h",
        text="Roles counted", color_discrete_sequence=[COLOR_TEAL],
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Companies with highest counted roles", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Roles counted", gridcolor="#e2e8f0"),
        height=max(280, 28 * len(top) + 80), showlegend=False,
    )
    return fig


def chart_sector_activity(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    counts = df["Sector"].value_counts().reset_index()
    counts.columns = ["Sector", "Events"]
    fig = px.bar(
        counts, x="Events", y="Sector", orientation="h",
        text="Events", color_discrete_sequence=[COLOR_BLUE],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Layoff activity by sector", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Number of events", gridcolor="#e2e8f0"),
        height=max(280, 26 * len(counts) + 80), showlegend=False,
    )
    return fig


def chart_sector_roles(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    roles = (
        df.groupby("Sector")["Roles counted"].sum().fillna(0)
        .sort_values(ascending=False).reset_index()
    )
    roles = roles[roles["Roles counted"] > 0]
    if roles.empty:
        return _empty_fig("No counted roles")
    fig = px.bar(
        roles, x="Roles counted", y="Sector", orientation="h",
        text="Roles counted", color_discrete_sequence=[COLOR_VIOLET],
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Affected employees by sector (counted)", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Roles counted", gridcolor="#e2e8f0"),
        height=max(280, 26 * len(roles) + 80), showlegend=False,
    )
    return fig


def chart_country_activity(df: pd.DataFrame, n: int = 15) -> go.Figure:
    if df.empty:
        return _empty_fig()
    rows = []
    for _, r in df.iterrows():
        for c in [p.strip() for p in str(r["Country_clean"]).split(",") if p.strip()]:
            rows.append(c)
    if not rows:
        return _empty_fig()
    counts = pd.Series(rows).value_counts().head(n).reset_index()
    counts.columns = ["Country", "Events"]
    fig = px.bar(
        counts, x="Events", y="Country", orientation="h",
        text="Events", color_discrete_sequence=[COLOR_CYAN],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Layoff activity by country", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Number of events", gridcolor="#e2e8f0"),
        height=max(280, 26 * len(counts) + 80), showlegend=False,
    )
    return fig


def chart_attribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    counts = df["Attribution"].value_counts().reset_index()
    counts.columns = ["Attribution", "Events"]
    fig = px.pie(
        counts, names="Attribution", values="Events",
        color="Attribution", color_discrete_map=ATTR_COLORS, hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="AI attribution mix", font=dict(size=14)),
        height=320, showlegend=True,
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def chart_attribution_by_sector(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    ct = pd.crosstab(df["Sector"], df["Attribution"])
    if ct.empty:
        return _empty_fig()
    if "explicit" in ct.columns:
        order = (ct["explicit"] / ct.sum(axis=1)).sort_values(ascending=False).index
        ct = ct.reindex(order)
    fig = go.Figure()
    for col in ["explicit", "mixed", "blamed"]:
        if col in ct.columns:
            fig.add_trace(go.Bar(
                name=col, y=ct.index, x=ct[col], orientation="h",
                marker_color=ATTR_COLORS.get(col, "#64748b"),
            ))
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
        top, x="% of workforce", y="Company", orientation="h",
        text="% of workforce", color_discrete_sequence=[COLOR_ROSE],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Highest workforce reduction %", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="% of workforce", gridcolor="#e2e8f0"),
        height=max(280, 28 * len(top) + 80), showlegend=False,
    )
    return fig


def chart_execution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    counts = df["Execution"].value_counts().reset_index()
    counts.columns = ["Execution", "Events"]
    fig = px.bar(
        counts, x="Events", y="Execution", orientation="h",
        text="Events", color_discrete_sequence=[COLOR_AMBER],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Execution status", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Events", gridcolor="#e2e8f0"),
        height=280, showlegend=False,
    )
    return fig


def chart_evidence_tier(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    counts = (
        df["Evidence tier"].value_counts()
        .reindex(["Tier 1", "Tier 2", "Tier 3"]).dropna().reset_index()
    )
    counts.columns = ["Evidence tier", "Events"]
    fig = px.bar(
        counts, x="Events", y="Evidence tier", orientation="h",
        text="Events", color_discrete_sequence=[COLOR_INDIGO],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Evidence tier distribution", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Events", gridcolor="#e2e8f0"),
        height=260, showlegend=False,
    )
    return fig


def chart_evidence_vs_attribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig()
    ct = pd.crosstab(df["Evidence tier"], df["Attribution"])
    order = [t for t in ["Tier 1", "Tier 2", "Tier 3"] if t in ct.index]
    ct = ct.reindex(order)
    fig = go.Figure()
    for col in ["explicit", "mixed", "blamed"]:
        if col in ct.columns:
            fig.add_trace(go.Bar(
                name=col, y=ct.index, x=ct[col], orientation="h",
                marker_color=ATTR_COLORS.get(col, "#64748b"),
            ))
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
        counts, x="Count", y="Profession", orientation="h",
        text="Count", color_discrete_sequence=[COLOR_TEAL],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Most frequently affected job families", font=dict(size=14)),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Mentions across events", gridcolor="#e2e8f0"),
        height=max(280, 28 * len(counts) + 80), showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_data()

    # Sidebar filters
    with st.sidebar:
        st.markdown("## Filters")
        st.caption("Leave a filter empty to include all values in that dimension.")

        years = sorted(df["Year"].dropna().unique().astype(int).tolist())
        sectors = sorted(df["Sector"].dropna().unique().tolist())
        countries = sorted({
            c.strip()
            for row in df["Country_clean"].dropna()
            for c in str(row).split(",")
            if c.strip()
        })
        attributions = sorted(df["Attribution"].dropna().unique().tolist())
        evidence_tiers = sorted(df["Evidence tier"].dropna().unique().tolist())
        executions = sorted(df["Execution"].dropna().unique().tolist())
        kinds = sorted(df["Kind"].dropna().unique().tolist())

        f_year = st.multiselect("Year", years, default=[])
        f_sector = st.multiselect("Sector", sectors, default=[])
        f_country = st.multiselect("Country", countries, default=[])
        f_attr = st.multiselect("Attribution", attributions, default=[])
        f_evidence = st.multiselect("Evidence tier", evidence_tiers, default=[])
        f_execution = st.multiselect("Execution", executions, default=[])
        f_kind = st.multiselect("Kind", kinds, default=[])

        st.markdown("---")
        if st.button("Reset filters", use_container_width=True):
            st.rerun()

    filtered = apply_filters(
        df, f_year, f_sector, f_country, f_attr, f_evidence, f_execution, f_kind
    )
    stats = summary_stats(filtered)

    # Header
    st.markdown("# AI Layoffs Register")
    st.markdown(
        '<p style="color:#64748b;font-size:0.92rem;margin-top:-0.35rem;margin-bottom:1rem;">'
        "Every row sourced · 2023–2026 · Interactive insights from the AI-attributed workforce reductions register"
        "</p>",
        unsafe_allow_html=True,
    )

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Events", f"{stats['events']:,}")
    c2.metric("Companies", f"{stats['companies']:,}")
    c3.metric("Roles counted", f"{stats['roles_counted']:,}")
    c4.metric("Not counted", f"{stats['roles_reported_not_counted']:,}")
    c5.metric("Explicit AI", f"{stats['explicit_pct']}%")
    c6.metric("Date span", f"{stats['date_min']} → {stats['date_max']}")

    st.markdown("---")

    # 1. Overall
    st.markdown("## 1 · Overall layoff statistics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_timeline(filtered), use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(chart_top_companies(filtered), use_container_width=True, config={"displayModeBar": False})

    # 2. Industry
    st.markdown("## 2 · Industry analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_sector_activity(filtered), use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(chart_sector_roles(filtered), use_container_width=True, config={"displayModeBar": False})

    # 3. Geography
    st.markdown("## 3 · Geographical analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_country_activity(filtered), use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(chart_attribution_by_sector(filtered), use_container_width=True, config={"displayModeBar": False})

    # 4. Attribution
    st.markdown("## 4 · AI attribution analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_attribution(filtered), use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(chart_job_families(filtered), use_container_width=True, config={"displayModeBar": False})

    # 5. Workforce impact
    st.markdown("## 5 · Workforce impact")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_workforce_pct(filtered), use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(chart_execution(filtered), use_container_width=True, config={"displayModeBar": False})

    # 6. Evidence quality
    st.markdown("## 6 · Evidence quality")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_evidence_tier(filtered), use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(chart_evidence_vs_attribution(filtered), use_container_width=True, config={"displayModeBar": False})

    # Events table
    st.markdown("## Recent events")
    st.caption("Latest 25 matching rows · click Source URL where available")
    table = (
        filtered.sort_values("Date", ascending=False)
        .head(25)[
            [
                "Date", "Company", "Sector", "Country",
                "Roles counted", "Attribution", "Evidence tier",
                "Execution", "Kind", "Claim", "Source URL",
            ]
        ]
        .copy()
    )
    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    table["Roles counted"] = table["Roles counted"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    table["Claim"] = table["Claim"].fillna("").astype(str).str.slice(0, 180)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source URL": st.column_config.LinkColumn("Source", display_text="Open"),
            "Roles counted": st.column_config.NumberColumn("Roles counted", format="%d"),
            "Claim": st.column_config.TextColumn("Claim", width="large"),
        },
    )

    st.markdown(
        '<p class="footer-note">Data from the AI Layoffs Register · Every row is source-cited · '
        'Methodology tiers 1–3 · Built for exploration, not press tallies</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
