"""
5_Driving_Behaviour.py
======================
Driver Pulse — Driving Behaviour Page
Owned by: Saisha (Safety & Sensor Intelligence Lead)

What this page does:
  1. Shows a driver's overall behaviour score with grade & trend
  2. Deep-dives into smoothness, speed, cabin and consistency components
  3. Per-trip behaviour breakdown with quality timeline
  4. Shift pattern heatmap (when do they drive, how do they behave)
  5. Peer benchmarking (percentile vs fleet)
  6. Route & zone analysis
  7. Gamified badges
  8. Personalised coaching tips

Design Aesthetic:
  Clean white editorial — like a premium driver wellness report.
  White background, black typography, sharp black CTAs.
  Data-forward. Every number has context. No decoration for its own sake.

Integration:
  Auth contract: st.session_state["logged_in"] + st.session_state["driver_id"]
  Works in demo mode (dropdown) if not logged in.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_loader import load_all_data, get_driver_data
from utils.behaviour_analytics import (
    build_behaviour_profile,
    get_all_driver_summary_scores,
)


# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
# Page config is handled in app.py



# ─────────────────────────────────────────────────────────────
#  CSS — White editorial theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300;1,9..40,400&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

  /* ── Global reset ── */
  .stApp {
    background: #ffffff;
    color: #0a0a0a;
    font-family: 'DM Sans', sans-serif;
  }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1440px;
  }

  /* ── Page header ── */
  .page-hero {
    border-bottom: 2px solid #0a0a0a;
    padding-bottom: 28px;
    margin-bottom: 36px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }
  .page-hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 800;
    color: #0a0a0a;
    line-height: 1.05;
    margin: 0;
    letter-spacing: -0.02em;
  }
  .page-hero-sub {
    font-size: 0.8rem;
    font-weight: 500;
    color: #6b7280;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 8px;
  }
  .page-hero-meta {
    text-align: right;
    font-size: 0.78rem;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    line-height: 1.8;
  }

  /* ── Driver selector (demo mode) ── */
  .stSelectbox label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .stSelectbox > div > div {
    border: 1.5px solid #0a0a0a !important;
    border-radius: 6px !important;
    background: #fff !important;
    color: #0a0a0a !important;
  }
  .stSelectbox > div > div > div { color: #0a0a0a !important; }
  .stSelectbox svg { fill: #0a0a0a !important; opacity: 1 !important; }
  [data-baseweb="popover"] *, [role="option"], [role="option"] * { color: #0a0a0a !important; }

  /* ── Section label ── */
  .section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 20px;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e5e7eb;
  }

  /* ── Score card — the hero number ── */
  .score-hero-card {
    background: #0a0a0a;
    color: #ffffff;
    border-radius: 20px;
    padding: 40px 36px;
    position: relative;
    overflow: hidden;
  }
  .score-hero-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
  }
  .score-hero-number {
    font-family: 'Playfair Display', serif;
    font-size: 6rem;
    font-weight: 800;
    line-height: 1;
    color: #ffffff;
  }
  .score-hero-grade {
    display: inline-block;
    background: #ffffff;
    color: #0a0a0a;
    font-family: 'DM Mono', monospace;
    font-size: 1.2rem;
    font-weight: 500;
    padding: 6px 18px;
    border-radius: 4px;
    margin-top: 12px;
  }
  .score-hero-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 12px;
  }

  /* ── Trend badge ── */
  .trend-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 16px;
  }
  .trend-improving { background: #dcfce7; color: #15803d; }
  .trend-declining  { background: #fee2e2; color: #b91c1c; }
  .trend-stable     { background: #f3f4f6; color: #4b5563; }

  /* ── KPI strip cards ── */
  .kpi-strip {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 22px 24px;
    height: 100%;
  }
  .kpi-strip-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 8px;
  }
  .kpi-strip-value {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #0a0a0a;
    line-height: 1;
  }
  .kpi-strip-unit {
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 4px;
  }
  .kpi-strip-delta {
    font-size: 0.75rem;
    margin-top: 8px;
    font-weight: 600;
  }
  .delta-pos { color: #16a34a; }
  .delta-neg { color: #dc2626; }
  .delta-neu { color: #6b7280; }

  /* ── Component score bar ── */
  .comp-row {
    margin-bottom: 20px;
  }
  .comp-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
  }
  .comp-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #374151;
  }
  .comp-score-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #0a0a0a;
    font-weight: 500;
  }
  .comp-bar-bg {
    background: #f3f4f6;
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
  }
  .comp-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: #0a0a0a;
    transition: width 0.4s ease;
  }
  .comp-bar-fill.good  { background: #16a34a; }
  .comp-bar-fill.warn  { background: #d97706; }
  .comp-bar-fill.poor  { background: #dc2626; }
  .comp-weight {
    font-size: 0.65rem;
    color: #9ca3af;
    margin-top: 3px;
  }

  /* ── Trip quality table ── */
  .trip-table-header {
    display: grid;
    grid-template-columns: 90px 120px 100px 80px 80px 80px 90px 1fr;
    gap: 10px;
    padding: 10px 16px;
    background: #0a0a0a;
    border-radius: 10px 10px 0 0;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #ffffff;
  }
  .trip-table-row {
    display: grid;
    grid-template-columns: 90px 120px 100px 80px 80px 80px 90px 1fr;
    gap: 10px;
    padding: 12px 16px;
    border-bottom: 1px solid #f3f4f6;
    font-size: 0.8rem;
    align-items: center;
    transition: background 0.15s;
  }
  .trip-table-row:hover { background: #f9fafb; }
  .trip-table-row:last-child { border-bottom: none; }

  /* ── Quality badge ── */
  .q-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .q-excellent { background: #dcfce7; color: #15803d; }
  .q-good      { background: #dbeafe; color: #1d4ed8; }
  .q-fair      { background: #fef9c3; color: #a16207; }
  .q-poor      { background: #fee2e2; color: #b91c1c; }
  .q-unknown   { background: #f3f4f6; color: #6b7280; }

  /* ── Severity dot ── */
  .sev-dot {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .sev-dot::before {
    content: '●';
    font-size: 0.6rem;
  }
  .sev-none   { color: #6b7280; }
  .sev-low    { color: #16a34a; }
  .sev-medium { color: #d97706; }
  .sev-high   { color: #dc2626; }

  /* ── Percentile bar ── */
  .pct-container {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 28px 30px;
  }
  .pct-bar-bg {
    background: #e5e7eb;
    border-radius: 4px;
    height: 10px;
    position: relative;
    margin: 16px 0;
  }
  .pct-bar-fill {
    background: #0a0a0a;
    height: 100%;
    border-radius: 4px;
  }
  .pct-marker {
    position: absolute;
    top: -6px;
    transform: translateX(-50%);
    width: 22px;
    height: 22px;
    background: #0a0a0a;
    border-radius: 50%;
    border: 3px solid #ffffff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }

  /* ── Badge grid ── */
  .badge-card {
    background: #f9fafb;
    border: 1.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .badge-card:hover {
    border-color: #0a0a0a;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
  }
  .badge-icon { font-size: 2rem; margin-bottom: 8px; }
  .badge-name {
    font-size: 0.78rem;
    font-weight: 700;
    color: #0a0a0a;
    margin-bottom: 4px;
  }
  .badge-detail {
    font-size: 0.68rem;
    color: #6b7280;
    line-height: 1.4;
  }
  .badge-card.locked {
    opacity: 0.35;
    filter: grayscale(1);
  }

  /* ── Coaching tips ── */
  .tip-card {
    border: 1.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 14px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
    background: #fff;
  }
  .tip-card.tip-high { border-left: 4px solid #dc2626; }
  .tip-card.tip-medium { border-left: 4px solid #d97706; }
  .tip-card.tip-positive { border-left: 4px solid #16a34a; }
  .tip-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
  .tip-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #0a0a0a;
    margin-bottom: 4px;
  }
  .tip-body {
    font-size: 0.8rem;
    color: #4b5563;
    line-height: 1.6;
  }
  .tip-metric {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 8px;
    background: #f3f4f6;
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
  }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 80px 40px;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    margin-top: 32px;
  }
  .empty-state-icon { font-size: 3.5rem; margin-bottom: 16px; }
  .empty-state-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0a0a0a;
    margin-bottom: 10px;
  }
  .empty-state-body {
    font-size: 0.88rem;
    color: #6b7280;
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.7;
  }

  /* ── Heatmap label ── */
  .hm-hour-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #6b7280;
    text-align: center;
  }

  /* ── Info pill ── */
  .info-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f3f4f6;
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 500;
    color: #374151;
    margin: 2px;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 7px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 8px 20px;
    border: none;
    letter-spacing: 0.02em;
  }
  .stTabs [aria-selected="true"] {
    background: #0a0a0a !important;
    color: #ffffff !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    padding-top: 28px;
  }

  /* ── Chart card ── */
  .chart-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px;
  }
  .chart-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0a0a0a;
    margin-bottom: 18px;
    letter-spacing: -0.01em;
  }

  /* ── Star rating ── */
  .star-display {
    font-size: 1.1rem;
    letter-spacing: 2px;
    margin-top: 4px;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #f9fafb; }
  ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

  /* ── No data info box ── */
  .stInfo { background: #f0f9ff; border-color: #bae6fd; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────────────────────
PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#374151", size=11),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(gridcolor="#f3f4f6", showgrid=True, zeroline=False,
               linecolor="#e5e7eb", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#f3f4f6", showgrid=True, zeroline=False,
               linecolor="#e5e7eb", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#e5e7eb",
                borderwidth=1, font=dict(size=10)),
    hoverlabel=dict(bgcolor="#0a0a0a", font_color="#ffffff",
                    bordercolor="#0a0a0a", font_size=12),
)

C_BLACK  = "#0a0a0a"
C_GRAY   = "#9ca3af"
C_LIGHT  = "#f3f4f6"
C_GREEN  = "#16a34a"
C_AMBER  = "#d97706"
C_RED    = "#dc2626"
C_BLUE   = "#2563eb"


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 75: return C_GREEN
    if score >= 50: return C_AMBER
    return C_RED


def score_bar_class(score: float) -> str:
    if score >= 75: return "good"
    if score >= 50: return "warn"
    return "poor"


def quality_class(q: str) -> str:
    return {"excellent": "q-excellent", "good": "q-good",
            "fair": "q-fair", "poor": "q-poor"}.get(str(q).lower(), "q-unknown")


def severity_class(s: str) -> str:
    return f"sev-{str(s).lower()}"


def stars_html(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def trend_html(trend: str, delta: float) -> str:
    if trend == "improving":
        return f'<span class="trend-badge trend-improving">↑ Improving &nbsp;{delta:+.1f} pts</span>'
    if trend == "declining":
        return f'<span class="trend-badge trend-declining">↓ Declining &nbsp;{delta:+.1f} pts</span>'
    return f'<span class="trend-badge trend-stable">→ Stable</span>'


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading behaviour data…"):
    data = load_all_data()

drivers_df   = data["drivers"]
all_summaries = data["summaries"]


# ─────────────────────────────────────────────────────────────
#  AUTH / SESSION CHECK
# ─────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in", False):
    if "driver_id" not in st.session_state:
        st.session_state["driver_id"] = "SDRV083"
    demo_mode = True
else:
    demo_mode = False


# ─────────────────────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────────────────────
col_title, col_picker = st.columns([3, 1])

with col_title:
    st.markdown("""
    <div class="page-hero">
      <div>
        <div class="page-hero-title">Driving<br>Behaviour</div>
        <div class="page-hero-sub">Performance analysis · Coaching · Benchmarking</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_picker:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if demo_mode:
        # Only show drivers who have trip summaries for a useful demo
        if not all_summaries.empty:
            available_drivers = sorted(all_summaries["driver_id"].unique().tolist())
        else:
            available_drivers = sorted(drivers_df["driver_id"].tolist())

        default_idx = (available_drivers.index(st.session_state["driver_id"])
                       if st.session_state["driver_id"] in available_drivers else 0)
        selected = st.selectbox(
            "View driver",
            options=available_drivers,
            index=default_idx,
            key="behaviour_driver_select",
        )
        st.session_state["driver_id"] = selected
    else:
        selected = st.session_state["driver_id"]

    # Quick driver info pill
    driver_row = drivers_df[drivers_df["driver_id"] == st.session_state["driver_id"]]
    if not driver_row.empty:
        dr = driver_row.iloc[0]
        st.markdown(f"""
        <div style="margin-top:10px; text-align:right;">
          <span class="info-pill">👤 {dr['name']}</span>
          <span class="info-pill">📍 {dr['city']}</span>
          <span class="info-pill">⭐ {dr['rating']}</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LOAD DRIVER DATA & BUILD PROFILE
# ─────────────────────────────────────────────────────────────
driver_id   = st.session_state["driver_id"]
driver_data = get_driver_data(driver_id, data)
driver_info = driver_data["driver"]

driver_name = driver_info.get("name", driver_id) if driver_info else driver_id

with st.spinner("Computing behaviour profile…"):
    profile = build_behaviour_profile(
        driver_id    = driver_id,
        driver_info  = driver_info if driver_info else {},
        trips_df     = driver_data["trips"],
        summaries_df = driver_data["summaries"],
        flags_df     = driver_data["flags"],
        all_drivers_df   = drivers_df,
        all_summaries_df = all_summaries,
    )


# ─────────────────────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────────────────────
if not profile.get("has_data", False):
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-state-icon">🚗</div>
      <div class="empty-state-title">No Trip Data Yet, {driver_name}</div>
      <div class="empty-state-body">
        Your driving behaviour report will appear here once you complete
        your first trip. Every trip builds your profile — smoothness,
        speed discipline, cabin quality, and consistency.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
#  SECTION 1 — HERO SCORE ROW
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">OVERALL BEHAVIOUR SCORE</div>', unsafe_allow_html=True)

col_hero, col_components, col_kpis = st.columns([1.2, 1.4, 1.4])

with col_hero:
    score = profile["overall_behaviour_score"]
    grade = profile["overall_grade"]
    stars = profile["overall_stars"]
    trend = profile["trend"]
    delta = profile["trend_delta"]

    star_html = stars_html(stars)
    trend_badge = trend_html(trend, delta)

    st.markdown(f"""
    <div class="score-hero-card">
      <div class="score-hero-label">BEHAVIOUR SCORE</div>
      <div class="score-hero-number">{score:.0f}</div>
      <div style="margin-top:10px;">
        <span class="score-hero-grade">{grade} — {star_html}</span>
      </div>
      <div style="margin-top:14px;">{trend_badge}</div>
      <div style="margin-top:20px; font-size:0.72rem; color:rgba(255,255,255,0.4);
                  border-top:1px solid rgba(255,255,255,0.1); padding-top:16px;">
        Based on {profile['total_trips']} trip{"s" if profile['total_trips'] != 1 else ""} ·
        {profile['total_distance_km']:.0f} km driven
      </div>
    </div>
    """, unsafe_allow_html=True)


with col_components:
    st.markdown('<div class="chart-card" style="height:100%">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Score Components</div>', unsafe_allow_html=True)

    components = [
        ("Smoothness",   profile["avg_smoothness"],    35, "How gently you accelerate & brake"),
        ("Speed Discipline", profile["avg_speed_score"], 25, "Adherence to safe speed limits"),
        ("Cabin Quality", profile["avg_cabin_score"],  20, "Noise & passenger environment"),
        ("Consistency",  profile["consistency_score"], 20, "Trip-to-trip reliability"),
    ]

    for name, val, weight, desc in components:
        bar_cls = score_bar_class(val)
        st.markdown(f"""
        <div class="comp-row">
          <div class="comp-header">
            <span class="comp-name">{name}</span>
            <span class="comp-score-val">{val:.0f}<span style="font-size:0.7rem;color:#9ca3af">/100</span></span>
          </div>
          <div class="comp-bar-bg">
            <div class="comp-bar-fill {bar_cls}" style="width:{val}%"></div>
          </div>
          <div class="comp-weight">{desc} · {weight}% of total</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


with col_kpis:
    # 4 KPI cards in 2×2
    k1, k2 = st.columns(2)
    k3, k4 = st.columns(2)

    def kpi_card(col, label, value, unit, delta_str=None, delta_positive=True):
        d_class = "delta-pos" if delta_positive else "delta-neg"
        delta_html = f'<div class="kpi-strip-delta {d_class}">{delta_str}</div>' if delta_str else ""
        col.markdown(f"""
        <div class="kpi-strip">
          <div class="kpi-strip-label">{label}</div>
          <div class="kpi-strip-value">{value}</div>
          <div class="kpi-strip-unit">{unit}</div>
          {delta_html}
        </div>
        """, unsafe_allow_html=True)

    excellent = profile["excellent_trips"]
    total     = profile["total_trips"]
    pct_excel = round(excellent / max(total, 1) * 100)

    kpi_card(k1, "TRIPS", str(total), "completed")
    kpi_card(k2, "EXCELLENT", f"{pct_excel}%", "trip quality",
             f"{excellent} of {total} trips", excellent > total // 2)
    kpi_card(k3, "PERCENTILE", f"{profile['percentile']}th",
             "vs fleet", "better than fleet avg" if profile["percentile"] > 50 else "below fleet avg",
             profile["percentile"] > 50)
    kpi_card(k4, "RATING", f"{profile['rating']:.1f}",
             "platform rating",
             f"{profile['experience_months']}mo experience", True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SECTION 2 — TABS
# ─────────────────────────────────────────────────────────────
tab_trips, tab_charts, tab_shifts, tab_bench, tab_coach = st.tabs([
    "  📋  Trip Log  ",
    "  📊  Analytics  ",
    "  🕐  Shift Patterns  ",
    "  🏆  Benchmarking  ",
    "  💡  Coaching  ",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — TRIP LOG
# ══════════════════════════════════════════════════════════════
with tab_trips:
    st.markdown('<div class="section-label">PER-TRIP BEHAVIOUR LOG</div>', unsafe_allow_html=True)

    trip_df = profile["trip_scores_df"]

    if trip_df.empty:
        st.info("No trip data available for this driver.")
    else:
        # ── Filters row ──
        fc1, fc2, fc3, _ = st.columns([1, 1, 1, 2])
        with fc1:
            quality_filter = st.selectbox(
                "Quality filter",
                ["All", "Excellent", "Good", "Fair", "Poor"],
                key="quality_filter"
            )
        with fc2:
            severity_filter = st.selectbox(
                "Severity filter",
                ["All", "None", "Low", "Medium", "High"],
                key="severity_filter"
            )
        with fc3:
            sort_by = st.selectbox(
                "Sort by",
                ["Trip order", "Score ↓", "Score ↑", "Fare ↓", "Distance ↓"],
                key="sort_by"
            )

        # Apply filters
        display_df = trip_df.copy()
        if quality_filter != "All":
            display_df = display_df[display_df["trip_quality_rating"].str.lower() == quality_filter.lower()]
        if severity_filter != "All":
            display_df = display_df[display_df["max_severity"].str.lower() == severity_filter.lower()]

        # Apply sort
        sort_map = {
            "Score ↓": ("total_score", False),
            "Score ↑": ("total_score", True),
            "Fare ↓":  ("fare", False),
            "Distance ↓": ("distance_km", False),
        }
        if sort_by in sort_map:
            col_s, asc = sort_map[sort_by]
            display_df = display_df.sort_values(col_s, ascending=asc)

        # ── Summary stats row ──
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Showing", f"{len(display_df)} trips")
        s2.metric("Avg Score", f"{display_df['total_score'].mean():.1f}" if not display_df.empty else "–")
        s3.metric("Avg Fare", f"₹{display_df['fare'].mean():.0f}" if not display_df.empty else "–")
        s4.metric("Total Distance", f"{display_df['distance_km'].sum():.1f} km" if not display_df.empty else "–")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if display_df.empty:
            st.info("No trips match the selected filters.")
        else:
            # ── Table header ──
            st.markdown("""
            <div style="background:#fff; border:1px solid #e5e7eb; border-radius:12px; overflow:hidden;">
              <div class="trip-table-header">
                <span>TRIP ID</span>
                <span>ROUTE</span>
                <span>TIME</span>
                <span>SCORE</span>
                <span>QUALITY</span>
                <span>SEVERITY</span>
                <span>FARE</span>
                <span>FLAGS</span>
              </div>
            """, unsafe_allow_html=True)

            for _, row in display_df.iterrows():
                score_val = row["total_score"]
                sc_color  = score_color(score_val)
                q_cls     = quality_class(row["trip_quality_rating"])
                s_cls     = severity_class(row["max_severity"])
                route     = f"{str(row['pickup_location'])[:12]}→{str(row['dropoff_location'])[:12]}"
                time_str  = str(row.get("start_time", ""))[:5] if row.get("start_time") else "—"
                flags_str = f"{int(row['flagged_count'])} flag{'s' if row['flagged_count'] != 1 else ''}"
                if row.get("high_flags", 0) > 0:
                    flags_str += f" ({row['high_flags']} high)"

                st.markdown(f"""
                <div class="trip-table-row">
                  <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#6b7280">
                    {str(row['trip_id'])[-6:]}
                  </span>
                  <span style="font-size:0.75rem; color:#374151">{route}</span>
                  <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#6b7280">{time_str}</span>
                  <span style="font-family:'DM Mono',monospace; font-weight:700;
                               color:{sc_color}">{score_val:.0f}</span>
                  <span><span class="q-badge {q_cls}">{row['trip_quality_rating']}</span></span>
                  <span class="sev-dot {s_cls}">{str(row['max_severity']).capitalize()}</span>
                  <span style="font-family:'DM Mono',monospace; font-size:0.8rem">₹{row['fare']:.0f}</span>
                  <span style="font-size:0.75rem; color:#9ca3af">{flags_str}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab_charts:
    trip_df = profile["trip_scores_df"]

    if trip_df.empty:
        st.info("No analytics data available yet.")
    else:
        st.markdown('<div class="section-label">BEHAVIOUR SCORE OVER TRIPS</div>', unsafe_allow_html=True)

        # ── Chart 1: Score timeline ──
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Behaviour Score Timeline</div>', unsafe_allow_html=True)

        fig_timeline = go.Figure()

        # Background severity shading
        for _, row in trip_df.iterrows():
            if row["max_severity"] == "high":
                fig_timeline.add_vrect(
                    x0=row["trip_index"] - 0.4,
                    x1=row["trip_index"] + 0.4,
                    fillcolor="rgba(220,38,38,0.05)",
                    line_width=0,
                )

        # Score line
        fig_timeline.add_trace(go.Scatter(
            x=trip_df["trip_index"],
            y=trip_df["total_score"],
            mode="lines+markers",
            name="Behaviour Score",
            line=dict(color=C_BLACK, width=2.5),
            marker=dict(
                color=[score_color(s) for s in trip_df["total_score"]],
                size=9,
                line=dict(width=2, color=C_BLACK),
            ),
            hovertemplate=(
                "<b>Trip %{x}</b><br>"
                "Score: <b>%{y:.1f}</b><br>"
                "<extra></extra>"
            ),
        ))

        # Trend line (if 3+ trips)
        if len(trip_df) >= 3:
            x_vals = trip_df["trip_index"].values
            y_vals = trip_df["total_score"].values
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            trend_color = C_GREEN if z[0] > 0 else C_RED if z[0] < -0.5 else C_GRAY
            fig_timeline.add_trace(go.Scatter(
                x=x_vals,
                y=p(x_vals),
                mode="lines",
                name="Trend",
                line=dict(color=trend_color, width=1.5, dash="dot"),
                hoverinfo="skip",
            ))

        # Fleet avg reference line
        fleet_avg = profile["fleet_avg"]
        fig_timeline.add_hline(
            y=fleet_avg, line_dash="dash",
            line_color=C_GRAY, line_width=1,
            annotation_text=f"Fleet avg ({fleet_avg:.0f})",
            annotation_font_color=C_GRAY,
            annotation_font_size=10,
        )

        fig_timeline.update_layout(
            **{**PLOTLY, "height": 280,
               "yaxis": dict(range=[0, 105], title="Score", gridcolor="#f3f4f6"),
               "xaxis": dict(title="Trip Number", dtick=1 if len(trip_df) <= 10 else 2),
               "margin": dict(l=10, r=10, t=10, b=30),
               "legend": dict(orientation="h", y=1.1)},
        )
        st.plotly_chart(fig_timeline, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Charts 2 + 3 ──
        c1, c2 = st.columns(2)

        with c1:
            # Component breakdown radar
            st.markdown('<div class="section-label">COMPONENT ANALYSIS</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Behaviour Components Radar</div>', unsafe_allow_html=True)

            radar_labels = ["Smoothness", "Speed\nDiscipline", "Cabin\nQuality", "Consistency"]
            radar_values = [
                profile["avg_smoothness"],
                profile["avg_speed_score"],
                profile["avg_cabin_score"],
                profile["consistency_score"],
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(10,10,10,0.08)",
                line=dict(color=C_BLACK, width=2),
                marker=dict(color=C_BLACK, size=6),
                name="You",
            ))
            # Fleet average
            fa = profile["fleet_avg"]
            fig_radar.add_trace(go.Scatterpolar(
                r=[fa, fa, fa, fa, fa],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(107,114,128,0.05)",
                line=dict(color=C_GRAY, width=1.5, dash="dot"),
                marker=dict(color=C_GRAY, size=4),
                name="Fleet avg",
            ))
            fig_radar.update_layout(
                **{**PLOTLY,
                   "height": 280,
                   "margin": dict(l=30, r=30, t=30, b=30),
                   "polar": dict(
                       bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True, range=[0, 100],
                                       gridcolor="#e5e7eb",
                                       tickfont=dict(color="#9ca3af", size=9)),
                       angularaxis=dict(gridcolor="#e5e7eb",
                                        tickfont=dict(color="#374151", size=10)),
                   )},
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            # Trip quality distribution
            st.markdown('<div class="section-label">QUALITY DISTRIBUTION</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Trip Quality Distribution</div>', unsafe_allow_html=True)

            q_counts = trip_df["trip_quality_rating"].value_counts()
            q_order  = ["excellent", "good", "fair", "poor"]
            q_labels = [q for q in q_order if q in q_counts.index]
            q_values = [q_counts.get(q, 0) for q in q_labels]
            q_colors = {"excellent": "#16a34a", "good": "#2563eb",
                        "fair": "#d97706", "poor": "#dc2626"}

            fig_qual = go.Figure(go.Bar(
                x=q_labels,
                y=q_values,
                marker_color=[q_colors.get(q, C_GRAY) for q in q_labels],
                text=q_values,
                textposition="outside",
                textfont=dict(size=12, color=C_BLACK),
                hovertemplate="<b>%{x}</b><br>%{y} trips<extra></extra>",
            ))
            fig_qual.update_layout(
                **{**PLOTLY, "height": 280,
                   "showlegend": False,
                   "margin": dict(l=10, r=10, t=20, b=30),
                   "xaxis": dict(title=None),
                   "yaxis": dict(title="Trips", dtick=1)},
            )
            st.plotly_chart(fig_qual, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Charts 4 + 5 ──
        c3, c4 = st.columns(2)

        with c3:
            # Stress score over trips
            st.markdown('<div class="section-label">STRESS PROFILE</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Trip Stress Score vs Behaviour Score</div>', unsafe_allow_html=True)

            fig_stress = go.Figure()
            fig_stress.add_trace(go.Bar(
                x=trip_df["trip_index"],
                y=trip_df["stress_score"] * 100,
                name="Stress %",
                marker_color=[
                    "rgba(220,38,38,0.7)" if s > 0.6 else
                    "rgba(217,119,6,0.7)" if s > 0.3 else
                    "rgba(22,163,74,0.7)"
                    for s in trip_df["stress_score"]
                ],
                hovertemplate="Trip %{x}<br>Stress: <b>%{y:.0f}%</b><extra></extra>",
            ))
            fig_stress.add_trace(go.Scatter(
                x=trip_df["trip_index"],
                y=trip_df["total_score"],
                mode="lines",
                name="Behaviour Score",
                yaxis="y2",
                line=dict(color=C_BLACK, width=2),
                hovertemplate="Trip %{x}<br>Score: <b>%{y:.1f}</b><extra></extra>",
            ))
            fig_stress.update_layout(
                **{**PLOTLY, "height": 260,
                   "margin": dict(l=10, r=10, t=10, b=30),
                   "yaxis":  dict(title="Stress %", gridcolor="#f3f4f6", range=[0, 105]),
                   "yaxis2": dict(title="Score", overlaying="y", side="right",
                                  range=[0, 105], gridcolor="rgba(0,0,0,0)",
                                  showgrid=False),
                   "legend": dict(orientation="h", y=1.1),
                   "barmode": "overlay"},
            )
            st.plotly_chart(fig_stress, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            # Fare vs score scatter
            st.markdown('<div class="section-label">EARNINGS CORRELATION</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Fare vs Behaviour Score</div>', unsafe_allow_html=True)

            fig_scatter = go.Figure(go.Scatter(
                x=trip_df["total_score"],
                y=trip_df["fare"],
                mode="markers",
                marker=dict(
                    color=trip_df["stress_score"],
                    colorscale=[[0, C_GREEN], [0.5, C_AMBER], [1, C_RED]],
                    size=10,
                    line=dict(width=1.5, color=C_BLACK),
                    showscale=True,
                    colorbar=dict(
                        title="Stress",
                        thickness=10,
                        tickfont=dict(size=9),
                        len=0.6,
                    ),
                ),
                text=trip_df["trip_id"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Score: %{x:.1f}<br>"
                    "Fare: ₹%{y:.0f}<extra></extra>"
                ),
            ))

            # Trend line if enough points
            if len(trip_df) >= 4:
                try:
                    z2  = np.polyfit(trip_df["total_score"], trip_df["fare"], 1)
                    p2  = np.poly1d(z2)
                    xs  = np.linspace(trip_df["total_score"].min(), trip_df["total_score"].max(), 50)
                    fig_scatter.add_trace(go.Scatter(
                        x=xs, y=p2(xs), mode="lines",
                        line=dict(color=C_GRAY, dash="dot", width=1.5),
                        hoverinfo="skip", name="Trend",
                    ))
                except Exception:
                    pass

            fig_scatter.update_layout(
                **{**PLOTLY, "height": 260,
                   "margin": dict(l=10, r=10, t=10, b=30),
                   "xaxis": dict(title="Behaviour Score"),
                   "yaxis": dict(title="Fare (₹)"),
                   "showlegend": False},
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Chart 6: Surge analysis ──
        surge = profile.get("surge_analysis", {})
        if surge and surge.get("surge_dist"):
            st.markdown('<div class="section-label">SURGE MULTIPLIER PATTERNS</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="chart-title">Surge Distribution — {surge["surge_pct"]}% of trips at surge pricing · Avg {surge["avg_surge"]}×</div>', unsafe_allow_html=True)

            surge_keys = [str(k) for k in sorted(surge["surge_dist"].keys())]
            surge_vals = [surge["surge_dist"][float(k)] for k in sorted(surge["surge_dist"].keys())]

            fig_surge = go.Figure(go.Bar(
                x=surge_keys,
                y=surge_vals,
                marker_color=[C_BLACK if float(k) > 1.0 else C_GRAY for k in surge_keys],
                text=surge_vals,
                textposition="outside",
                hovertemplate="<b>%{x}× surge</b><br>%{y} trips<extra></extra>",
            ))
            fig_surge.update_layout(
                **{**PLOTLY, "height": 200,
                   "showlegend": False,
                   "margin": dict(l=10, r=10, t=10, b=30),
                   "xaxis": dict(title="Surge Multiplier"),
                   "yaxis": dict(title="Trips", dtick=1)},
            )
            st.plotly_chart(fig_surge, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3 — SHIFT PATTERNS
# ══════════════════════════════════════════════════════════════
with tab_shifts:
    st.markdown('<div class="section-label">DRIVING SHIFT PATTERNS</div>', unsafe_allow_html=True)

    trip_df    = profile["trip_scores_df"]
    heatmap    = profile["shift_heatmap"]
    locations  = profile["location_patterns"]

    if trip_df.empty or not heatmap:
        st.info("Shift pattern data unavailable — no timestamped trips recorded.")
    else:
        # ── Shift summary cards ──
        from utils.behaviour_analytics import SHIFT_BUCKETS, _hour_to_shift

        shift_totals = {}
        if "hour" in trip_df.columns:
            for label, hours in SHIFT_BUCKETS.items():
                mask = trip_df["hour"].isin(list(hours))
                shift_totals[label] = {
                    "trips": int(mask.sum()),
                    "avg_score": round(trip_df.loc[mask, "total_score"].mean(), 1) if mask.sum() > 0 else 0,
                    "avg_fare":  round(trip_df.loc[mask, "fare"].mean(), 1) if mask.sum() > 0 else 0,
                }

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        shift_labels = list(SHIFT_BUCKETS.keys())
        shift_cols   = [sc1, sc2, sc3, sc4, sc5]
        shift_icons  = ["🌅", "☀️", "🌤️", "🌆", "🌙"]

        pref = profile.get("shift_preference", "")
        for col, lbl, icon in zip(shift_cols, shift_labels, shift_icons):
            stats = shift_totals.get(lbl, {"trips": 0, "avg_score": 0, "avg_fare": 0})
            is_pref = pref.lower() in lbl.lower()
            border  = "border: 2px solid #0a0a0a;" if is_pref else ""
            col.markdown(f"""
            <div class="kpi-strip" style="{border}">
              <div style="font-size:1.3rem; margin-bottom:4px;">{icon}</div>
              <div class="kpi-strip-label">{lbl.split("(")[0].strip()}</div>
              <div class="kpi-strip-value">{stats['trips']}</div>
              <div class="kpi-strip-unit">trips</div>
              <div class="kpi-strip-delta delta-{'pos' if stats['avg_score'] >= 70 else 'neu'}">
                Score: {stats['avg_score']:.0f}
              </div>
            </div>
            """, unsafe_allow_html=True)

        if pref:
            st.markdown(f"""
            <div style="margin-top:10px; font-size:0.78rem; color:#6b7280;">
              ◉ Highlighted shift = your registered preference ({pref.replace("_", " ").title()})
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Hourly heatmap ──
        st.markdown('<div class="section-label">HOUR-BY-HOUR ACTIVITY</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Trip Volume & Avg Stress by Hour of Day</div>', unsafe_allow_html=True)

        hours  = list(range(24))
        counts = [heatmap.get(h, {}).get("count", 0) for h in hours]
        stress = [heatmap.get(h, {}).get("avg_stress", 0) for h in hours]
        fares  = [heatmap.get(h, {}).get("avg_fare", 0) for h in hours]

        fig_hm = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hm.add_trace(go.Bar(
            x=hours, y=counts,
            name="Trips",
            marker_color=[
                "rgba(10,10,10,0.85)" if c > 0 else "rgba(10,10,10,0.1)"
                for c in counts
            ],
            hovertemplate="<b>%{x}:00</b><br>Trips: %{y}<extra></extra>",
        ), secondary_y=False)
        fig_hm.add_trace(go.Scatter(
            x=hours, y=stress,
            name="Avg Stress %",
            mode="lines+markers",
            line=dict(color=C_RED, width=2),
            marker=dict(size=6, color=C_RED),
            hovertemplate="<b>%{x}:00</b><br>Stress: %{y:.0f}%<extra></extra>",
        ), secondary_y=True)

        # Shade shift zones
        shift_colors = {
            "Early Morning": "rgba(254,215,170,0.15)",
            "Morning":       "rgba(253,230,138,0.15)",
            "Afternoon":     "rgba(187,247,208,0.10)",
            "Evening":       "rgba(199,210,254,0.15)",
            "Night":         "rgba(30,30,30,0.06)",
        }
        zone_ranges = [(5, 9, "Early Morning"), (9, 12, "Morning"),
                       (12, 17, "Afternoon"), (17, 21, "Evening")]
        for start, end, name in zone_ranges:
            fig_hm.add_vrect(
                x0=start - 0.5, x1=end - 0.5,
                fillcolor=shift_colors.get(name, "rgba(0,0,0,0.03)"),
                line_width=0,
                annotation_text=name,
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color="#9ca3af",
            )

        fig_hm.update_layout(
            **{**PLOTLY, "height": 280,
               "margin": dict(l=10, r=10, t=30, b=30),
               "bargap": 0.15,
               "legend": dict(orientation="h", y=1.08),
               "xaxis":  dict(title="Hour of Day", tickmode="linear", dtick=2),
               "yaxis":  dict(title="Trips", gridcolor="#f3f4f6"),
               "yaxis2": dict(title="Stress %", overlaying="y", side="right",
                              range=[0, 100], showgrid=False)},
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Location patterns ──
        if locations:
            lc1, lc2 = st.columns(2)

            with lc1:
                st.markdown('<div class="section-label">TOP PICKUP ZONES</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Most Frequent Pickup Locations</div>', unsafe_allow_html=True)

                pickups = locations.get("top_pickups", {})
                if pickups:
                    fig_pick = go.Figure(go.Bar(
                        y=list(pickups.keys()),
                        x=list(pickups.values()),
                        orientation="h",
                        marker_color=C_BLACK,
                        text=list(pickups.values()),
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>%{x} pickups<extra></extra>",
                    ))
                    fig_pick.update_layout(
                        **{**PLOTLY, "height": 220,
                           "margin": dict(l=10, r=40, t=10, b=10),
                           "showlegend": False,
                           "yaxis": dict(autorange="reversed"),
                           "xaxis": dict(title="Trip count", dtick=1)},
                    )
                    st.plotly_chart(fig_pick, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No pickup data available.")
                st.markdown("</div>", unsafe_allow_html=True)

            with lc2:
                st.markdown('<div class="section-label">TOP ROUTES</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Most Frequent Routes</div>', unsafe_allow_html=True)

                routes = locations.get("top_routes", {})
                if routes:
                    route_labels = [r[:35] + "…" if len(r) > 35 else r for r in routes.keys()]
                    fig_route = go.Figure(go.Bar(
                        y=route_labels,
                        x=list(routes.values()),
                        orientation="h",
                        marker_color=[C_BLACK, "#374151", "#6b7280",
                                       "#9ca3af", "#d1d5db"][:len(routes)],
                        text=list(routes.values()),
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>%{x} times<extra></extra>",
                    ))
                    fig_route.update_layout(
                        **{**PLOTLY, "height": 220,
                           "margin": dict(l=10, r=40, t=10, b=10),
                           "showlegend": False,
                           "yaxis": dict(autorange="reversed"),
                           "xaxis": dict(title="Count", dtick=1)},
                    )
                    st.plotly_chart(fig_route, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("Insufficient route data.")
                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 4 — BENCHMARKING
# ══════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown('<div class="section-label">FLEET BENCHMARKING</div>', unsafe_allow_html=True)

    percentile = profile["percentile"]
    fleet_avg  = profile["fleet_avg"]
    own_score  = profile["overall_behaviour_score"]

    # ── Percentile hero ──
    pct1, pct2 = st.columns([1, 2])

    with pct1:
        pct_label = (
            "Top performer" if percentile >= 80 else
            "Above average" if percentile >= 60 else
            "Average"       if percentile >= 40 else
            "Below average"
        )
        pct_color = (
            C_GREEN if percentile >= 70 else
            C_AMBER if percentile >= 40 else
            C_RED
        )
        st.markdown(f"""
        <div class="pct-container">
          <div style="font-size:0.68rem; font-weight:700; letter-spacing:0.2em;
                      text-transform:uppercase; color:#9ca3af; margin-bottom:12px;">
            YOUR PERCENTILE RANK
          </div>
          <div style="font-family:'Playfair Display',serif; font-size:4rem;
                      font-weight:800; color:{pct_color}; line-height:1;">
            {percentile}<span style="font-size:1.5rem;">th</span>
          </div>
          <div style="font-size:0.85rem; font-weight:600; color:#374151; margin-top:6px;">
            {pct_label}
          </div>
          <div class="pct-bar-bg" style="margin-top:20px;">
            <div class="pct-bar-fill" style="width:{percentile}%;"></div>
            <div class="pct-marker" style="left:{percentile}%;"></div>
          </div>
          <div style="display:flex; justify-content:space-between;
                      font-size:0.68rem; color:#9ca3af; margin-top:4px;">
            <span>0th</span><span>50th</span><span>100th</span>
          </div>
          <div style="margin-top:20px; padding-top:16px; border-top:1px solid #e5e7eb;">
            <div style="font-size:0.75rem; color:#6b7280; margin-bottom:4px;">Your score</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.5rem; font-weight:500;">
              {own_score:.0f}
              <span style="font-size:0.85rem; color:#9ca3af;">vs {fleet_avg:.0f} fleet avg</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with pct2:
        # Fleet distribution
        st.markdown('<div class="section-label">FLEET SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card" style="height:100%">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Where You Stand vs All Drivers</div>', unsafe_allow_html=True)

        # Build fleet distribution from all summaries
        if not all_summaries.empty:
            fleet_stress = all_summaries.groupby("driver_id")["stress_score"].mean()
            fleet_scores = ((1 - fleet_stress.clip(0, 1)) * 100).values

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=fleet_scores,
                nbinsx=20,
                name="Fleet",
                marker_color="rgba(10,10,10,0.12)",
                marker_line_color="rgba(10,10,10,0.3)",
                marker_line_width=1,
                hovertemplate="Score: %{x:.0f}<br>Drivers: %{y}<extra></extra>",
            ))
            # Your marker
            fig_dist.add_vline(
                x=own_score, line_width=2.5, line_color=C_BLACK,
                annotation_text=f"You ({own_score:.0f})",
                annotation_font_color=C_BLACK,
                annotation_font_size=11,
                annotation_font_family="DM Sans",
            )
            fig_dist.add_vline(
                x=fleet_avg, line_width=1.5, line_color=C_GRAY, line_dash="dash",
                annotation_text=f"Fleet avg ({fleet_avg:.0f})",
                annotation_font_color=C_GRAY,
                annotation_font_size=10,
            )
            fig_dist.update_layout(
                **{**PLOTLY, "height": 290,
                   "margin": dict(l=10, r=10, t=10, b=30),
                   "showlegend": False,
                   "xaxis": dict(title="Behaviour Score"),
                   "yaxis": dict(title="# Drivers")},
            )
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Fleet data unavailable for comparison.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Component comparison ──
    st.markdown('<div class="section-label">COMPONENT-LEVEL COMPARISON</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Your Component Scores vs Fleet Average</div>', unsafe_allow_html=True)

    comp_labels = ["Smoothness", "Speed Discipline", "Cabin Quality", "Consistency"]
    your_vals   = [
        profile["avg_smoothness"],
        profile["avg_speed_score"],
        profile["avg_cabin_score"],
        profile["consistency_score"],
    ]
    fleet_vals  = [fleet_avg] * 4  # simplified fleet avg across all components

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="You",
        x=comp_labels,
        y=your_vals,
        marker_color=C_BLACK,
        text=[f"{v:.0f}" for v in your_vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Your score: %{y:.0f}<extra></extra>",
    ))
    fig_comp.add_trace(go.Bar(
        name="Fleet Avg",
        x=comp_labels,
        y=fleet_vals,
        marker_color="rgba(10,10,10,0.15)",
        text=[f"{v:.0f}" for v in fleet_vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Fleet avg: %{y:.0f}<extra></extra>",
    ))
    fig_comp.update_layout(
        **{**PLOTLY, "height": 260,
           "barmode": "group",
           "margin": dict(l=10, r=10, t=20, b=30),
           "yaxis": dict(range=[0, 115], title="Score /100"),
           "legend": dict(orientation="h", y=1.08)},
    )
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Badges ──
    st.markdown('<div class="section-label">YOUR BADGES</div>', unsafe_allow_html=True)

    earned_badges = profile["badges"]

    if not earned_badges:
        st.markdown("""
        <div style="text-align:center; padding:40px; background:#f9fafb;
                    border:1px solid #e5e7eb; border-radius:12px;">
          <div style="font-size:2rem; margin-bottom:8px;">🎖️</div>
          <div style="font-weight:700; color:#374151; margin-bottom:4px;">No badges yet</div>
          <div style="font-size:0.82rem; color:#9ca3af;">
            Complete more trips with consistent safe driving to earn badges.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show all possible badges, earned highlighted
        from utils.behaviour_analytics import BADGES as ALL_BADGES

        earned_keys = {b["label"] for b in earned_badges}
        badge_cols  = st.columns(4)

        all_badge_list = list(ALL_BADGES.values())
        for i, badge in enumerate(all_badge_list):
            col = badge_cols[i % 4]
            earned = badge["label"] in earned_keys
            locked_cls = "" if earned else "locked"
            detail = next((b["detail"] for b in earned_badges if b["label"] == badge["label"]), badge["desc"])
            col.markdown(f"""
            <div class="badge-card {locked_cls}">
              <div class="badge-icon">{badge['icon']}</div>
              <div class="badge-name">{badge['label']}</div>
              <div class="badge-detail">{detail if earned else badge['desc']}</div>
              {"<div style='margin-top:8px; font-size:0.65rem; color:#16a34a; font-weight:700;'>✓ EARNED</div>" if earned else ""}
            </div>
            """, unsafe_allow_html=True)
            col.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 5 — COACHING
# ══════════════════════════════════════════════════════════════
with tab_coach:
    st.markdown('<div class="section-label">PERSONALISED COACHING REPORT</div>', unsafe_allow_html=True)

    # ── Coach intro card ──
    score  = profile["overall_behaviour_score"]
    grade  = profile["overall_grade"]
    name   = driver_name.split()[0] if driver_name else "Driver"

    coach_msg = (
        f"Strong performance, {name}. Your behaviour score of {score:.0f} puts you in "
        f"the {profile['percentile']}th percentile. Here's how to maintain and improve:"
        if score >= 70 else
        f"Good effort, {name}. Your score of {score:.0f} shows room for improvement. "
        f"Follow these targeted tips to move up:"
        if score >= 50 else
        f"Let's build from here, {name}. Your score of {score:.0f} tells us exactly "
        f"where to focus. These specific actions will make the biggest difference:"
    )

    st.markdown(f"""
    <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:14px;
                padding:24px 28px; margin-bottom:28px; display:flex; gap:20px; align-items:flex-start;">
      <div style="font-size:2.5rem; flex-shrink:0;">🚀</div>
      <div>
        <div style="font-family:'Playfair Display',serif; font-size:1.2rem; font-weight:700;
                    color:#0a0a0a; margin-bottom:6px;">Your Coaching Report — Grade {grade}</div>
        <div style="font-size:0.88rem; color:#4b5563; line-height:1.6;">{coach_msg}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tips ──
    tips = profile["coaching_tips"]

    if not tips:
        st.markdown("""
        <div class="tip-card tip-positive">
          <div class="tip-icon">🏆</div>
          <div>
            <div class="tip-title">Outstanding driving — nothing to flag!</div>
            <div class="tip-body">
              All your metrics are in excellent shape. Keep driving consistently,
              accumulate more trips, and watch your badges grow.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for tip in tips:
            prio_cls = f"tip-{tip['priority']}"
            st.markdown(f"""
            <div class="tip-card {prio_cls}">
              <div class="tip-icon">{tip['icon']}</div>
              <div>
                <div class="tip-title">{tip['title']}</div>
                <div class="tip-body">{tip['body']}</div>
                <div class="tip-metric">{tip['metric']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Score improvement roadmap ──
    st.markdown('<div class="section-label">SCORE IMPROVEMENT ROADMAP</div>', unsafe_allow_html=True)

    current = profile["overall_behaviour_score"]
    next_milestone = 90 if current < 90 else 95 if current < 95 else 100
    gap = round(next_milestone - current, 1)

    r1, r2, r3 = st.columns(3)
    milestones = [
        ("Current Score", f"{current:.0f}", "Where you are now", "→"),
        ("Next Milestone", f"{next_milestone}", f"+{gap} pts needed", "→"),
        ("Top Driver Score", "90+", "Top 10% of fleet", "🏆"),
    ]
    for col, (label, val, sub, icon) in zip([r1, r2, r3], milestones):
        col.markdown(f"""
        <div class="kpi-strip" style="text-align:center;">
          <div style="font-size:1.5rem; margin-bottom:6px;">{icon}</div>
          <div class="kpi-strip-label">{label}</div>
          <div class="kpi-strip-value">{val}</div>
          <div class="kpi-strip-unit">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Weekly focus areas ──
    st.markdown('<div class="section-label">THIS WEEK\'S FOCUS AREAS</div>', unsafe_allow_html=True)

    focus_areas = []
    if profile["avg_smoothness"] < 70:
        focus_areas.append(("🎯", "Smoothness", "Spend 3 full seconds on each brake and acceleration",
                             f"Current: {profile['avg_smoothness']:.0f}/100"))
    if profile["avg_speed_score"] < 70:
        focus_areas.append(("🚦", "Speed", "Stay 5 km/h under the limit on all arterial roads",
                             f"Current: {profile['avg_speed_score']:.0f}/100"))
    if profile["avg_cabin_score"] < 70:
        focus_areas.append(("🔊", "Cabin", "Play soft background music, avoid prolonged phone calls",
                             f"Current: {profile['avg_cabin_score']:.0f}/100"))
    if profile["consistency_score"] < 70:
        focus_areas.append(("📊", "Consistency", "Use the same pre-trip checklist before every shift",
                             f"Current: {profile['consistency_score']:.0f}/100"))

    if not focus_areas:
        st.markdown("""
        <div style="background:#dcfce7; border:1px solid #bbf7d0; border-radius:10px;
                    padding:20px; text-align:center;">
          <div style="font-size:1.5rem;">✅</div>
          <div style="font-weight:700; color:#15803d; margin-top:6px;">All components above target!</div>
          <div style="font-size:0.8rem; color:#166534; margin-top:4px;">
            Focus on maintaining consistency and accumulating more trips.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        fa_cols = st.columns(len(focus_areas))
        for col, (icon, title, action, metric) in zip(fa_cols, focus_areas):
            col.markdown(f"""
            <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px;
                        padding:20px; text-align:center; height:100%;">
              <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
              <div style="font-size:0.82rem; font-weight:700; color:#0a0a0a; margin-bottom:6px;">{title}</div>
              <div style="font-size:0.75rem; color:#4b5563; line-height:1.5; margin-bottom:10px;">{action}</div>
              <div style="font-family:'DM Mono',monospace; font-size:0.7rem; background:#e5e7eb;
                          padding:3px 8px; border-radius:4px; color:#374151;">{metric}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:56px; padding:20px 0; border-top:1px solid #e5e7eb;
            display:flex; justify-content:space-between; align-items:center;
            font-size:0.72rem; color:#9ca3af;">
  <span>🚗 Driver Pulse AI · Driving Behaviour v1.0</span>
  <span style="font-family:'DM Mono',monospace;">
    TRIPS · SUMMARIES · FLAGS · SENSOR DATA
  </span>
  <span>Computed in real-time · No personal data stored</span>
</div>
""", unsafe_allow_html=True)