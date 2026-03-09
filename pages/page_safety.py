"""
2_My_Safety.py
==============
Driver Pulse — My Safety Page
Owned by: Saisha (Safety & Sensor Intelligence Lead)

What this page does:
  1. Loads sensor data (accelerometer + audio + flagged moments)
  2. Runs feature engineering (rolling windows, jerk computation, audio spikes)
  3. Runs the Risk Score model (rule-based + Random Forest blend)
  4. Visualises everything in a polished, glanceable driver-facing UI

Design Aesthetic:
  Dark industrial — like a car dashboard at night.
  Deep charcoal background, amber/red accent for risk, teal for safe.
  Heavy use of Plotly for interactive charts.
  No clutter. Every element earns its place.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_loader import load_all_data, get_driver_data
from utils.feature_engineering import (
    extract_motion_features,
    extract_audio_features,
    get_harsh_events,
    get_audio_spikes,
    detect_conflict_moments,
    build_driver_safety_profile,
    compute_accel_magnitude,
    compute_jerk,
)
from models.risk_model import compute_risk_score


# ──────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
# Page config is handled in app.py



# ──────────────────────────────────────────────────────────────────
#  CUSTOM CSS  — dark dashboard aesthetic
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Import fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=Roboto+Mono:wght@400;500&display=swap');

  /* ── Global background ── */
  .stApp {
    background: #ffffff;
    color: #0f172a;
    font-family: 'Inter', sans-serif;
  }

  /* ── Hide Streamlit default elements ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1400px; }

  /* ── Streamlit default text colour overrides ── */
  .stApp p, .stApp span, .stApp div, .stApp label { color: #0f172a; }
  .stMarkdown { color: #0f172a; }

  /* ── WHITE TEXT — force inside all black/dark backgrounds ── */
  .risk-low-badge *, .risk-med-badge *, .risk-high-badge *,
  .risk-low-badge, .risk-med-badge, .risk-high-badge { color: #ffffff !important; }

  .pill-harsh, .pill-harsh *,
  .pill-conflict, .pill-conflict *,
  .pill-audio, .pill-audio *,
  .pill-speed, .pill-speed * { color: #ffffff !important; }

  /* Active tab text */
  .stTabs [aria-selected="true"],
  .stTabs [aria-selected="true"] * { color: #ffffff !important; }

  /* Selectbox dropdown text */
  .stSelectbox > div > div,
  .stSelectbox > div > div *,
  .stSelectbox > div > div > div { color: #ffffff !important; }
  .stSelectbox svg { fill: #ffffff !important; }
  /* Dropdown option list items (the open menu) stay dark on white */
  [data-baseweb="popover"] *, [data-baseweb="menu"] *,
  [role="option"], [role="option"] * { color: #0f172a !important; }

  /* Button text */
  .stButton > button,
  .stButton > button * { color: #ffffff !important; }

  /* ── Page header band ── */
  .page-header {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
  }
  .page-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #000000, #374151, #000000);
  }
  .page-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #000000;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 0;
  }
  .page-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 4px;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .driver-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 6px 14px;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.8rem;
    color: #334155;
  }

  /* ── Section labels ── */
  .section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 14px;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2e8f0;
  }

  /* ── KPI cards ── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
  }
  .kpi-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    transition: border-color 0.2s;
  }
  .kpi-card:hover { border-color: #94a3b8; }
  .kpi-card.risk-low  { border-left: 3px solid #10b981; }
  .kpi-card.risk-med  { border-left: 3px solid #f59e0b; }
  .kpi-card.risk-high { border-left: 3px solid #ef4444; }
  .kpi-card.neutral   { border-left: 3px solid #3b82f6; }
  .kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    color: #000000;
  }
  .kpi-unit {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 4px;
  }
  .kpi-trend {
    font-size: 0.72rem;
    margin-top: 6px;
  }
  .trend-up   { color: #ef4444; }
  .trend-down { color: #10b981; }
  .trend-flat { color: #94a3b8; }

  /* ── Risk score gauge card ── */
  .risk-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    height: 100%;
  }
  .risk-score-display {
    font-family: 'Rajdhani', sans-serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
  }
  .risk-cat-badge {
    display: inline-block;
    padding: 5px 18px;
    border-radius: 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 10px;
  }
  .risk-low-badge  { background: #000000; color: #ffffff; border: 1px solid #000000; }
  .risk-med-badge  { background: #000000; color: #ffffff; border: 1px solid #000000; }
  .risk-high-badge { background: #000000; color: #ffffff; border: 1px solid #000000; }

  /* ── Timeline event pills ── */
  .event-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    margin: 2px;
  }
  .pill-harsh    { background: #000000; color: #ffffff; border: 1px solid #000000; }
  .pill-audio    { background: #1e1b4b; color: #ffffff; border: 1px solid #1e1b4b; }
  .pill-conflict { background: #000000; color: #ffffff; border: 1px solid #000000; }
  .pill-speed    { background: #1e3a5f; color: #ffffff; border: 1px solid #1e3a5f; }

  /* ── Insight card ── */
  .insight-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
  }
  .insight-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 2px; }
  .insight-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #000000;
    margin-bottom: 3px;
  }
  .insight-body {
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.5;
  }

  /* ── Flag table ── */
  .flag-row {
    display: grid;
    grid-template-columns: 140px 120px 80px 1fr;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid #e2e8f0;
    font-size: 0.8rem;
    align-items: center;
    color: #0f172a;
  }
  .flag-row:last-child { border-bottom: none; }
  .flag-table-header {
    display: grid;
    grid-template-columns: 140px 120px 80px 1fr;
    gap: 12px;
    padding: 8px 16px;
    background: #f1f5f9;
    border-radius: 8px 8px 0 0;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
  }
  .sev-high { color: #dc2626; font-weight: 600; }
  .sev-med  { color: #d97706; font-weight: 600; }
  .sev-low  { color: #059669; font-weight: 600; }

  /* ── Contribution bar ── */
  .contrib-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .contrib-label {
    font-size: 0.75rem;
    color: #334155;
    min-width: 200px;
  }
  .contrib-bar-bg {
    flex: 1;
    background: #e2e8f0;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
  }
  .contrib-bar-fill {
    height: 100%;
    border-radius: 4px;
  }
  .contrib-score {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    min-width: 36px;
    text-align: right;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #e2e8f0;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 7px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 8px 20px;
    border: none;
  }
  .stTabs [aria-selected="true"] {
    background: #000000 !important;
    color: #ffffff !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    padding-top: 24px;
  }

  /* ── Chart containers ── */
  .chart-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
  }
  .chart-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #0f172a;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
  }

  /* ── Selectbox driver picker ── */
  .stSelectbox label {
    font-size: 0.75rem !important;
    color: #475569 !important;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .stSelectbox > div > div {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #000000 !important;
    border-radius: 8px !important;
  }
  .stSelectbox > div > div > div {
    color: #ffffff !important;
  }
  .stSelectbox svg { fill: #ffffff !important; }

  /* ── All Streamlit buttons ── */
  .stButton > button {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #000000 !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
  }
  .stButton > button:hover {
    background: #1f2937 !important;
    border-color: #1f2937 !important;
    color: #ffffff !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #f1f5f9; }
  ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

  /* ── st.info / st.warning boxes ── */
  .stAlert { background: #f1f5f9 !important; color: #0f172a !important;
             border: 1px solid #cbd5e1 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  PLOTLY THEME DEFAULTS
# ──────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#334155", size=11),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(gridcolor="#e2e8f0", showgrid=True, zeroline=False, color="#334155"),
    yaxis=dict(gridcolor="#e2e8f0", showgrid=True, zeroline=False, color="#334155"),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1, font=dict(color="#334155")),
    hoverlabel=dict(bgcolor="#f8fafc", font_color="#0f172a", bordercolor="#cbd5e1"),
)

COLOR_RISK_HIGH   = "#ef4444"
COLOR_RISK_MED    = "#f59e0b"
COLOR_RISK_LOW    = "#10b981"
COLOR_TEAL        = "#14b8a6"
COLOR_PURPLE      = "#8b5cf6"
COLOR_BLUE        = "#3b82f6"
COLOR_AMBER       = "#f59e0b"


# ──────────────────────────────────────────────────────────────────
#  HELPER: risk colour
# ──────────────────────────────────────────────────────────────────
def risk_color(category: str) -> str:
    return {
        "Low": COLOR_RISK_LOW,
        "Medium": COLOR_RISK_MED,
        "High": COLOR_RISK_HIGH,
    }.get(category, COLOR_BLUE)

def severity_color(sev: str) -> str:
    return {"high": COLOR_RISK_HIGH, "medium": COLOR_RISK_MED, "low": COLOR_RISK_LOW}.get(sev.lower(), COLOR_BLUE)


# ──────────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────────
with st.spinner("Loading sensor data…"):
    data = load_all_data()

drivers_df = data["drivers"]
from utils.data_loader import get_drivers_with_sensor_data
all_driver_ids = get_drivers_with_sensor_data(data)


# ──────────────────────────────────────────────────────────────────
#  SESSION / AUTH CHECK
# ──────────────────────────────────────────────────────────────────
# In production this comes from session_state after login.
# For standalone dev/demo we show a driver picker.
if "driver_id" not in st.session_state:
    st.session_state["driver_id"] = "SDRV039"   # default demo driver

demo_mode = not st.session_state.get("logged_in", False)


# ──────────────────────────────────────────────────────────────────
#  PAGE HEADER
# ──────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🛡 My Safety</div>
      <div class="page-subtitle">Real-time risk analysis · Sensor fusion · Driving behaviour</div>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if demo_mode:
        selected_driver = st.selectbox(
            "View driver",
            options=all_driver_ids,
            index=all_driver_ids.index(st.session_state["driver_id"])
                  if st.session_state["driver_id"] in all_driver_ids else 0,
            key="driver_select_safety",
        )
        st.session_state["driver_id"] = selected_driver
    else:
        selected_driver = st.session_state["driver_id"]


driver_id = st.session_state["driver_id"]
driver_data = get_driver_data(driver_id, data)
driver_info = driver_data["driver"]
driver_name = driver_info.get("name", driver_id)


# ──────────────────────────────────────────────────────────────────
#  COMPUTE SAFETY PROFILE & RISK SCORE
# ──────────────────────────────────────────────────────────────────
with st.spinner("Running safety analysis…"):
    profile = build_driver_safety_profile(
        driver_id,
        data["trips"],
        driver_data["acc"],
        driver_data["aud"],
        driver_data["flags"],
    )
    risk_result = compute_risk_score(profile)

risk_score    = risk_result["risk_score"]
risk_cat      = risk_result["risk_category"]
contributions = risk_result["contributions"]
ml_cat        = risk_result["ml_category"]
confidence    = risk_result["confidence"]

harsh_events   = get_harsh_events(driver_data["acc"])
audio_spikes   = get_audio_spikes(driver_data["aud"])
conflict_df    = detect_conflict_moments(driver_data["acc"], driver_data["aud"])
flags_df       = driver_data["flags"]


# ──────────────────────────────────────────────────────────────────
#  DRIVER INFO BAR
# ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:24px;">
  <div class="driver-badge">👤 {driver_name}</div>
  <div class="driver-badge">🆔 {driver_id}</div>
  <div class="driver-badge">🚗 {profile['total_trips']} trips analysed</div>
  <div class="driver-badge">⭐ Rating: {driver_info.get('rating', 'N/A')}</div>
  <div class="driver-badge">📍 {driver_info.get('city', 'N/A')}</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  TOP ROW: RISK GAUGE + 4 KPI CARDS
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">SAFETY OVERVIEW</div>', unsafe_allow_html=True)

col_gauge, col_kpis = st.columns([1, 3])

with col_gauge:
    # Plotly gauge chart
    gauge_color = risk_color(risk_cat)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"font": {"size": 48, "color": gauge_color, "family": "Rajdhani"},
                "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#cbd5e1", "tickfont": {"color": "#94a3b8", "size": 10}},
            "bar": {"color": gauge_color, "thickness": 0.25},
            "bgcolor": "#f8fafc",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],   "color": "#d1fae5"},
                {"range": [35, 65],  "color": "#fef3c7"},
                {"range": [65, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": gauge_color, "width": 3},
                "thickness": 0.8,
                "value": risk_score,
            },
        },
        title={"text": f"<b>RISK SCORE</b>", "font": {"size": 11, "color": "#475569", "family": "Rajdhani"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig_gauge.update_layout(**{**PLOTLY_LAYOUT, "height": 220, "margin": dict(l=20, r=20, t=30, b=10)})
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    cat_class = {"Low": "risk-low-badge", "Medium": "risk-med-badge", "High": "risk-high-badge"}.get(risk_cat, "risk-low-badge")
    st.markdown(f"""
    <div style="text-align:center; margin-top:-10px;">
      <span class="risk-cat-badge {cat_class}">{risk_cat} Risk</span>
      {"<br><span style='font-size:0.7rem; color:#475569; margin-top:6px; display:block'>ML: " + ml_cat + f" · {confidence}% confidence</span>" if confidence else ""}
    </div>
    """, unsafe_allow_html=True)

with col_kpis:
    k1, k2, k3, k4 = st.columns(4)

    harsh_count_val = profile["total_harsh"]
    audio_spike_val = profile["total_audio_spikes"]
    overspeed_val   = profile["total_overspeed"]
    smoothness_val  = profile["avg_smoothness"]
    conflict_val    = len(conflict_df) if not conflict_df.empty else 0
    high_flags_val  = profile["high_flags"]

    # Card 1 — Harsh Events
    hc = "risk-high" if harsh_count_val > 3 else "risk-med" if harsh_count_val > 1 else "risk-low"
    k1.markdown(f"""
    <div class="kpi-card {hc}">
      <div class="kpi-label">⚡ Harsh Events</div>
      <div class="kpi-value">{harsh_count_val}</div>
      <div class="kpi-unit">braking + acceleration</div>
      <div class="kpi-trend {'trend-up' if harsh_count_val>2 else 'trend-flat'}">
        {'▲ Above avg' if harsh_count_val > 2 else '● Within normal'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2 — Audio Spikes
    ac = "risk-high" if audio_spike_val > 5 else "risk-med" if audio_spike_val > 2 else "risk-low"
    k2.markdown(f"""
    <div class="kpi-card {ac}">
      <div class="kpi-label">🔊 Audio Spikes</div>
      <div class="kpi-value">{audio_spike_val}</div>
      <div class="kpi-unit">above threshold events</div>
      <div class="kpi-trend {'trend-up' if audio_spike_val>3 else 'trend-flat'}">
        {'▲ Elevated noise' if audio_spike_val > 3 else '● Cabin normal'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 3 — Conflict Moments
    cc = "risk-high" if conflict_val > 2 else "risk-med" if conflict_val > 0 else "risk-low"
    k3.markdown(f"""
    <div class="kpi-card {cc}">
      <div class="kpi-label">⚠️ Conflict Moments</div>
      <div class="kpi-value">{conflict_val}</div>
      <div class="kpi-unit">motion + audio overlap</div>
      <div class="kpi-trend {'trend-up' if conflict_val>1 else 'trend-down' if conflict_val==0 else 'trend-flat'}">
        {'▲ Needs attention' if conflict_val>1 else '✓ Clean' if conflict_val==0 else '● Monitor'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 4 — Smoothness
    sc = "risk-low" if smoothness_val > 70 else "risk-med" if smoothness_val > 45 else "risk-high"
    k4.markdown(f"""
    <div class="kpi-card {sc}">
      <div class="kpi-label">🎯 Smoothness Index</div>
      <div class="kpi-value">{smoothness_val:.0f}</div>
      <div class="kpi-unit">out of 100</div>
      <div class="kpi-trend {'trend-down' if smoothness_val>70 else 'trend-up' if smoothness_val<45 else 'trend-flat'}">
        {'✓ Smooth driving' if smoothness_val>70 else '▲ Erratic patterns' if smoothness_val<45 else '● Moderate'}
      </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  TABS: detailed analysis sections
# ──────────────────────────────────────────────────────────────────
tab_motion, tab_audio, tab_conflict, tab_flags, tab_score = st.tabs([
    "  ⚡  Motion Analysis  ",
    "  🔊  Audio Analysis  ",
    "  ⚠️  Conflict Detection  ",
    "  🚩  Flagged Events  ",
    "  🧠  Risk Breakdown  ",
])


# ─────────────────────────────────────────
#  TAB 1 — MOTION ANALYSIS
# ─────────────────────────────────────────
with tab_motion:
    st.markdown('<div class="section-label">ACCELEROMETER SIGNALS</div>', unsafe_allow_html=True)

    acc_df = driver_data["acc"].copy()

    if acc_df.empty:
        st.info("No accelerometer data available for this driver's trips yet.")
    else:
        acc_df = acc_df.sort_values("elapsed_seconds")
        acc_df["magnitude"] = np.sqrt(acc_df["accel_x"]**2 + acc_df["accel_y"]**2 + acc_df["accel_z"]**2)

        # ── Chart 1: Acceleration magnitude + speed over time ──
        col_c1, col_c2 = st.columns([3, 2])

        with col_c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Acceleration Magnitude & Speed Over Trip</div>', unsafe_allow_html=True)

            fig_mag = make_subplots(specs=[[{"secondary_y": True}]])

            fig_mag.add_trace(
                go.Scatter(
                    x=acc_df["elapsed_seconds"], y=acc_df["magnitude"],
                    name="Accel Magnitude (g)",
                    line=dict(color=COLOR_AMBER, width=2),
                    fill="tozeroy",
                    fillcolor="rgba(245,158,11,0.08)",
                    hovertemplate="<b>%{y:.2f}g</b> at %{x}s<extra></extra>",
                ),
                secondary_y=False,
            )

            # Threshold line
            fig_mag.add_hline(
                y=8.5, line_dash="dot",
                line_color=COLOR_RISK_HIGH, line_width=1.5,
                annotation_text="Harsh threshold (8.5g)",
                annotation_font_color=COLOR_RISK_HIGH,
                annotation_font_size=10,
            )

            fig_mag.add_trace(
                go.Scatter(
                    x=acc_df["elapsed_seconds"], y=acc_df["speed_kmh"],
                    name="Speed (km/h)",
                    line=dict(color=COLOR_BLUE, width=1.5, dash="dot"),
                    hovertemplate="<b>%{y:.1f} km/h</b> at %{x}s<extra></extra>",
                    opacity=0.7,
                ),
                secondary_y=True,
            )

            # Mark harsh events
            if not harsh_events.empty:
                he = harsh_events[harsh_events["trip_id"].isin(profile["trip_ids"])]
                if not he.empty:
                    fig_mag.add_trace(
                        go.Scatter(
                            x=he["elapsed_seconds"], y=he["magnitude"],
                            mode="markers",
                            marker=dict(color=COLOR_RISK_HIGH, size=10, symbol="x",
                                        line=dict(width=2, color=COLOR_RISK_HIGH)),
                            name="Harsh Event",
                            hovertemplate="<b>HARSH: %{y:.2f}g</b><extra></extra>",
                        ),
                        secondary_y=False,
                    )

            fig_mag.update_layout(
                **{**PLOTLY_LAYOUT, "height": 280,
                   "legend": dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
                   "margin": dict(l=10, r=10, t=20, b=10)},
            )
            fig_mag.update_yaxes(title_text="Acceleration (g)", secondary_y=False,
                                  gridcolor="#1e293b", title_font_color="#475569", title_font_size=10)
            fig_mag.update_yaxes(title_text="Speed (km/h)", secondary_y=True,
                                  gridcolor="#1e293b", title_font_color="#475569", title_font_size=10)
            st.plotly_chart(fig_mag, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">3-Axis Acceleration Breakdown</div>', unsafe_allow_html=True)

            fig_axes = go.Figure()
            axis_colors = [COLOR_RISK_HIGH, COLOR_AMBER, COLOR_TEAL]
            for axis, col in zip(["accel_x", "accel_y", "accel_z"], axis_colors):
                fig_axes.add_trace(go.Scatter(
                    x=acc_df["elapsed_seconds"], y=acc_df[axis],
                    name=axis.replace("accel_", "").upper(),
                    line=dict(color=col, width=1.5),
                    hovertemplate=f"<b>{axis}: %{{y:.2f}}</b><extra></extra>",
                ))
            fig_axes.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                      "legend": dict(orientation="h", y=1.15, bgcolor="rgba(0,0,0,0)"),
                                      "margin": dict(l=10, r=10, t=20, b=10)})
            st.plotly_chart(fig_axes, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Chart 2: Per-trip stats ──
        motion_feats = profile["motion_feats"]
        if not motion_feats.empty:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">PER-TRIP MOTION SUMMARY</div>', unsafe_allow_html=True)

            c3, c4 = st.columns(2)

            with c3:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Harsh Event Count per Trip</div>', unsafe_allow_html=True)
                mf = motion_feats.reset_index()
                bar_colors = [COLOR_RISK_HIGH if v > 2 else COLOR_RISK_MED if v > 0 else COLOR_RISK_LOW
                               for v in mf["harsh_event_count"]]
                fig_harsh = go.Figure(go.Bar(
                    x=mf["trip_id"], y=mf["harsh_event_count"],
                    marker_color=bar_colors,
                    hovertemplate="<b>%{x}</b><br>Harsh events: %{y}<extra></extra>",
                ))
                fig_harsh.update_layout(**{**PLOTLY_LAYOUT, "height": 220,
                                           "margin": dict(l=10, r=10, t=10, b=30)})
                st.plotly_chart(fig_harsh, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            with c4:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Driving Smoothness Index per Trip</div>', unsafe_allow_html=True)
                smooth_colors = [COLOR_RISK_LOW if v > 70 else COLOR_RISK_MED if v > 45 else COLOR_RISK_HIGH
                                  for v in mf["smoothness_index"]]
                fig_smooth = go.Figure(go.Bar(
                    x=mf["trip_id"], y=mf["smoothness_index"],
                    marker_color=smooth_colors,
                    hovertemplate="<b>%{x}</b><br>Smoothness: %{y:.1f}/100<extra></extra>",
                ))
                fig_smooth.add_hline(y=70, line_dash="dot", line_color=COLOR_RISK_LOW,
                                      line_width=1, annotation_text="Good (70)",
                                      annotation_font_color=COLOR_RISK_LOW, annotation_font_size=9)
                fig_smooth.update_layout(**{**PLOTLY_LAYOUT, "height": 220,
                                            "margin": dict(l=10, r=10, t=10, b=30),
                                            "yaxis": dict(range=[0, 110], gridcolor="#1e293b")})
                st.plotly_chart(fig_smooth, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Harsh events list ──
        if not harsh_events.empty:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">DETECTED HARSH EVENTS</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; overflow:hidden;">
              <div class="flag-table-header">
                <span>Timestamp</span><span>Event Type</span><span>Severity</span><span>Details</span>
              </div>
            """, unsafe_allow_html=True)

            he_display = harsh_events.head(15)
            for _, row in he_display.iterrows():
                sev = "high" if row["magnitude"] > 9 else "medium" if row["magnitude"] > 7 else "low"
                sc  = {"harsh_brake": "pill-harsh", "harsh_accel": "pill-harsh",
                       "sudden_jerk": "pill-conflict", "overspeed": "pill-speed"}.get(row["event_type"], "pill-harsh")
                st.markdown(f"""
                <div class="flag-row">
                  <span style="font-family:'Roboto Mono',monospace; color:#94a3b8; font-size:0.75rem">
                    {str(row['timestamp'])[:19] if pd.notnull(row.get('timestamp')) else f"{row['elapsed_seconds']:.0f}s"}
                  </span>
                  <span><span class="event-pill {sc}">{row['event_type'].replace('_',' ').upper()}</span></span>
                  <span class="sev-{sev}">{sev.upper()}</span>
                  <span style="color:#334155; font-size:0.78rem">
                    Magnitude: <b>{row['magnitude']:.2f}g</b> · Speed: <b>{row['speed_kmh']:.1f} km/h</b>
                  </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TAB 2 — AUDIO ANALYSIS
# ─────────────────────────────────────────
with tab_audio:
    st.markdown('<div class="section-label">CABIN AUDIO SIGNALS</div>', unsafe_allow_html=True)

    aud_df = driver_data["aud"].copy()

    if aud_df.empty:
        st.info("No audio data available for this driver's trips.")
    else:
        aud_df = aud_df.sort_values("elapsed_seconds")

        col_a1, col_a2 = st.columns([3, 2])

        with col_a1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Audio Intensity Over Trip (dB)</div>', unsafe_allow_html=True)

            mean_db = aud_df["audio_level_db"].mean()
            std_db  = aud_df["audio_level_db"].std(ddof=0)
            thresh  = mean_db + 1.8 * std_db

            # Colour each point by classification
            class_colors = {
                "quiet": "#1e3a5f", "normal": "#1e40af",
                "conversation": "#0d9488", "loud": "#d97706",
                "very_loud": "#dc2626", "argument": "#7c2d12",
            }
            aud_df["point_color"] = aud_df["audio_classification"].map(class_colors).fillna("#475569")

            fig_audio = go.Figure()
            fig_audio.add_trace(go.Scatter(
                x=aud_df["elapsed_seconds"], y=aud_df["audio_level_db"],
                mode="lines+markers",
                line=dict(color=COLOR_PURPLE, width=2),
                marker=dict(color=aud_df["point_color"], size=8, line=dict(width=1, color="#0d0f14")),
                name="Audio Level",
                hovertemplate="<b>%{y:.1f} dB</b> at %{x}s<br>Class: %{text}<extra></extra>",
                text=aud_df["audio_classification"],
            ))

            # Mean + threshold bands
            fig_audio.add_hrect(y0=thresh, y1=aud_df["audio_level_db"].max() + 5,
                                  fillcolor="rgba(239,68,68,0.06)", line_width=0,
                                  annotation_text="High Stress Zone",
                                  annotation_font_color=COLOR_RISK_HIGH,
                                  annotation_font_size=9)
            fig_audio.add_hline(y=thresh, line_dash="dot", line_color=COLOR_RISK_HIGH,
                                  line_width=1.5, annotation_text=f"Spike thresh ({thresh:.0f} dB)",
                                  annotation_font_color=COLOR_RISK_HIGH, annotation_font_size=9)
            fig_audio.add_hline(y=mean_db, line_dash="dash", line_color="#475569",
                                  line_width=1, annotation_text=f"Mean ({mean_db:.0f} dB)",
                                  annotation_font_color="#475569", annotation_font_size=9)

            fig_audio.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                       "margin": dict(l=10, r=10, t=20, b=10),
                                       "yaxis": dict(title="dB", gridcolor="#1e293b", title_font_size=10)})
            st.plotly_chart(fig_audio, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_a2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Audio Classification Breakdown</div>', unsafe_allow_html=True)

            class_counts = aud_df["audio_classification"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=class_counts.index,
                values=class_counts.values,
                hole=0.55,
                marker=dict(colors=[class_colors.get(c, "#475569") for c in class_counts.index],
                             line=dict(color="#0d0f14", width=2)),
                textinfo="label+percent",
                textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                     "margin": dict(l=10, r=10, t=10, b=10),
                                     "legend": dict(font=dict(size=9))})
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Audio classification legend ──
        st.markdown("""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:16px;">
          <span class="event-pill" style="background:#0c1624; color:#60a5fa; border:1px solid #1e3a5f">Quiet / Normal</span>
          <span class="event-pill" style="background:#0d4444; color:#5eead4; border:1px solid #0f766e">Conversation</span>
          <span class="event-pill" style="background:#3d2000; color:#fbbf24; border:1px solid #92400e">Loud</span>
          <span class="event-pill" style="background:#450a0a; color:#fca5a5; border:1px solid #7f1d1d">Very Loud / Argument ⚠</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-trip audio stress chart ──
        audio_feats = profile["audio_feats"]
        if not audio_feats.empty:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">PER-TRIP AUDIO STRESS INDEX</div>', unsafe_allow_html=True)

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            af = audio_feats.reset_index()
            stress_colors = [COLOR_RISK_HIGH if v > 50 else COLOR_RISK_MED if v > 20 else COLOR_RISK_LOW
                               for v in af["audio_stress_index"]]
            fig_stress = go.Figure(go.Bar(
                x=af["trip_id"], y=af["audio_stress_index"],
                marker_color=stress_colors,
                hovertemplate="<b>%{x}</b><br>Stress Index: %{y:.1f}/100<extra></extra>",
            ))
            fig_stress.add_hline(y=50, line_dash="dot", line_color=COLOR_RISK_HIGH, line_width=1)
            fig_stress.update_layout(**{**PLOTLY_LAYOUT, "height": 200,
                                        "margin": dict(l=10, r=10, t=10, b=30),
                                        "yaxis": dict(range=[0, 110], gridcolor="#1e293b")})
            st.plotly_chart(fig_stress, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Sustained high-audio periods ──
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        loud_periods = aud_df[aud_df["audio_classification"].isin(["loud", "very_loud", "argument"])]
        if not loud_periods.empty:
            st.markdown('<div class="section-label">SUSTAINED HIGH-AUDIO PERIODS</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; overflow:hidden;">
              <div class="flag-table-header">
                <span>Timestamp</span><span>Classification</span><span>Level</span><span>Duration</span>
              </div>
            """, unsafe_allow_html=True)
            for _, row in loud_periods.iterrows():
                cls   = row["audio_classification"]
                sc    = "pill-conflict" if cls == "argument" else "pill-harsh" if cls == "very_loud" else "pill-audio"
                sev_c = "sev-high" if cls in ["argument","very_loud"] else "sev-med"
                st.markdown(f"""
                <div class="flag-row">
                  <span style="font-family:'Roboto Mono',monospace; color:#94a3b8; font-size:0.75rem">
                    {str(row.get('timestamp',''))[:19] or f"{row['elapsed_seconds']:.0f}s"}
                  </span>
                  <span><span class="event-pill {sc}">{cls.upper()}</span></span>
                  <span class="{sev_c}">{row['audio_level_db']:.1f} dB</span>
                  <span style="color:#475569; font-size:0.78rem">
                    Sustained for <b>{row.get('sustained_duration_sec', 0):.0f}s</b>
                  </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TAB 3 — CONFLICT DETECTION
# ─────────────────────────────────────────
with tab_conflict:
    st.markdown('<div class="section-label">SENSOR FUSION — MOTION + AUDIO CONFLICT DETECTION</div>',
                unsafe_allow_html=True)

    # Explainer
    st.markdown("""
    <div class="insight-card">
      <div class="insight-icon">🔬</div>
      <div>
        <div class="insight-title">How Conflict Detection Works</div>
        <div class="insight-body">
          A <strong>conflict moment</strong> is flagged when a harsh motion event (hard braking, sudden jerk)
          overlaps with elevated cabin audio within a ±30-second window.
          The combined score blends the motion severity and audio level — giving a
          single metric that's harder to trigger accidentally than either signal alone.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if conflict_df.empty:
        st.markdown("""
        <div style="text-align:center; padding:60px; background:#f8fafc; border:1px solid #e2e8f0;
                    border-radius:12px; color:#059669;">
          <div style="font-size:3rem; margin-bottom:12px;">✅</div>
          <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem; font-weight:600; color:#000000;">
            No Conflict Moments Detected
          </div>
          <div style="font-size:0.85rem; color:#64748b; margin-top:8px;">
            Motion and audio signals did not overlap significantly in any trip.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Scatter: combined score
        cc1, cc2 = st.columns([2, 1])

        with cc1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Conflict Timeline — Motion vs Audio Score</div>', unsafe_allow_html=True)

            fig_conf = go.Figure()
            sev_cmap = {"high": COLOR_RISK_HIGH, "medium": COLOR_RISK_MED, "low": COLOR_RISK_LOW}
            for sev in ["high", "medium", "low"]:
                subset = conflict_df[conflict_df["severity"] == sev]
                if not subset.empty:
                    fig_conf.add_trace(go.Scatter(
                        x=subset["elapsed_seconds"], y=subset["combined_score"],
                        mode="markers",
                        marker=dict(
                            color=sev_cmap[sev], size=12 if sev == "high" else 9 if sev == "medium" else 7,
                            symbol="diamond" if sev == "high" else "circle",
                            line=dict(width=1.5, color="#0d0f14"),
                        ),
                        name=f"{sev.capitalize()} Severity",
                        hovertemplate=(
                            f"<b>{sev.upper()} CONFLICT</b><br>"
                            "Score: %{y:.2f}<br>At %{x}s<br>"
                            "Audio: %{customdata[0]:.1f} dB<extra></extra>"
                        ),
                        customdata=subset[["audio_level_db"]].values,
                    ))

            fig_conf.add_hline(y=0.7, line_dash="dot", line_color=COLOR_RISK_HIGH,
                                line_width=1, annotation_text="High threshold (0.7)",
                                annotation_font_color=COLOR_RISK_HIGH, annotation_font_size=9)
            fig_conf.add_hline(y=0.4, line_dash="dot", line_color=COLOR_RISK_MED,
                                line_width=1, annotation_text="Medium threshold (0.4)",
                                annotation_font_color=COLOR_RISK_MED, annotation_font_size=9)

            fig_conf.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                      "yaxis": dict(range=[0, 1.1], title="Combined Score",
                                                    gridcolor="#1e293b", title_font_size=10),
                                      "margin": dict(l=10, r=10, t=20, b=10),
                                      "legend": dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)")})
            st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with cc2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Severity Distribution</div>', unsafe_allow_html=True)
            sev_counts = conflict_df["severity"].value_counts()
            fig_sev = go.Figure(go.Bar(
                x=sev_counts.index.str.capitalize(),
                y=sev_counts.values,
                marker_color=[sev_cmap.get(s.lower(), COLOR_BLUE) for s in sev_counts.index],
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            ))
            fig_sev.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                     "margin": dict(l=10, r=10, t=20, b=30)})
            st.plotly_chart(fig_sev, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # Conflict detail table
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">CONFLICT MOMENT DETAIL</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; overflow:hidden;">
          <div style="display:grid; grid-template-columns:80px 120px 110px 110px 110px 1fr; gap:12px;
                       padding:8px 16px; background:#f1f5f9; font-size:0.65rem; font-weight:600;
                       letter-spacing:0.12em; text-transform:uppercase; color:#64748b;">
            <span>Time (s)</span><span>Motion Event</span><span>Motion Score</span>
            <span>Audio Score</span><span>Combined</span><span>Severity</span>
          </div>
        """, unsafe_allow_html=True)

        for _, row in conflict_df.iterrows():
            sev = row["severity"]
            sc  = {"high": "sev-high", "medium": "sev-med", "low": "sev-low"}.get(sev, "")
            st.markdown(f"""
            <div style="display:grid; grid-template-columns:80px 120px 110px 110px 110px 1fr; gap:12px;
                         padding:12px 16px; border-bottom:1px solid #e2e8f0; font-size:0.8rem; align-items:center;">
              <span style="font-family:'Roboto Mono',monospace; color:#64748b">{row['elapsed_seconds']:.0f}s</span>
              <span><span class="event-pill pill-harsh">{row['motion_event'].replace('_',' ').upper()}</span></span>
              <span style="color:#f59e0b">{row['motion_score']:.3f}</span>
              <span style="color:{COLOR_PURPLE}">{row['audio_score']:.3f} ({row['audio_level_db']:.0f}dB)</span>
              <span style="font-weight:600; color:{sev_cmap.get(sev, '#fff')}">{row['combined_score']:.3f}</span>
              <span class="{sc}">{sev.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TAB 4 — FLAGGED EVENTS
# ─────────────────────────────────────────
with tab_flags:
    st.markdown('<div class="section-label">SYSTEM-FLAGGED MOMENTS</div>', unsafe_allow_html=True)

    if flags_df.empty:
        st.info("No flagged moments recorded for this driver.")
    else:
        # Summary KPIs
        fk1, fk2, fk3, fk4 = st.columns(4)
        for col, sev, label, icon in [
            (fk1, "high",   "High Severity",   "🔴"),
            (fk2, "medium", "Medium Severity",  "🟡"),
            (fk3, "low",    "Low Severity",     "🟢"),
        ]:
            count = int((flags_df["severity"] == sev).sum())
            clr   = {"high":"risk-high","medium":"risk-med","low":"risk-low"}[sev]
            col.markdown(f"""
            <div class="kpi-card {clr}">
              <div class="kpi-label">{icon} {label}</div>
              <div class="kpi-value">{count}</div>
              <div class="kpi-unit">flagged events</div>
            </div>
            """, unsafe_allow_html=True)

        fk4.markdown(f"""
        <div class="kpi-card neutral">
          <div class="kpi-label">📋 Total Flags</div>
          <div class="kpi-value">{len(flags_df)}</div>
          <div class="kpi-unit">all severities</div>
        </div>
        """, unsafe_allow_html=True)

        # Score distribution chart
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        fc1, fc2 = st.columns([2, 1])

        with fc1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Combined Score Distribution by Flag Type</div>', unsafe_allow_html=True)
            fig_flags_box = go.Figure()
            for ftype in flags_df["flag_type"].unique():
                subset = flags_df[flags_df["flag_type"] == ftype]
                fig_flags_box.add_trace(go.Box(
                    y=subset["combined_score"],
                    name=ftype.replace("_", " ").title(),
                    boxpoints="all", jitter=0.4, pointpos=0,
                    marker=dict(size=5),
                    line=dict(color=COLOR_AMBER),
                    fillcolor="rgba(245,158,11,0.1)",
                    hovertemplate="<b>%{y:.3f}</b><extra></extra>",
                ))
            fig_flags_box.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                           "margin": dict(l=10, r=10, t=10, b=40),
                                           "yaxis": dict(title="Score", gridcolor="#1e293b", range=[0,1.1])})
            st.plotly_chart(fig_flags_box, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with fc2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Flag Type Frequency</div>', unsafe_allow_html=True)
            ft_counts = flags_df["flag_type"].value_counts()
            fig_ft = go.Figure(go.Bar(
                y=ft_counts.index.str.replace("_"," ").str.title(),
                x=ft_counts.values,
                orientation="h",
                marker_color=COLOR_AMBER,
                hovertemplate="<b>%{y}</b>: %{x}<extra></extra>",
            ))
            fig_ft.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                    "margin": dict(l=10, r=60, t=10, b=10)})
            st.plotly_chart(fig_ft, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # Full events table
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">ALL FLAGGED EVENTS</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; overflow:hidden;">
          <div style="display:grid; grid-template-columns:90px 140px 100px 80px 80px 80px 1fr; gap:10px;
                       padding:8px 16px; background:#f1f5f9; font-size:0.65rem; font-weight:600;
                       letter-spacing:0.12em; text-transform:uppercase; color:#64748b;">
            <span>Trip</span><span>Timestamp</span><span>Type</span>
            <span>Severity</span><span>Motion</span><span>Audio</span><span>Context</span>
          </div>
        """, unsafe_allow_html=True)

        for _, row in flags_df.iterrows():
            sev = row.get("severity", "low").lower()
            sc  = {"sev-high": "pill-harsh","sev-med":"pill-conflict","sev-low":"pill-speed"}.get(
                   f"sev-{sev}", "pill-speed")
            sev_class = f"sev-{sev}"
            flag_type_clean = str(row.get("flag_type","")).replace("_"," ").title()
            st.markdown(f"""
            <div style="display:grid; grid-template-columns:90px 140px 100px 80px 80px 80px 1fr; gap:10px;
                         padding:10px 16px; border-bottom:1px solid #e2e8f0; font-size:0.78rem; align-items:center;">
              <span style="font-family:'Roboto Mono',monospace; color:#64748b; font-size:0.72rem">
                {row.get('trip_id','')}
              </span>
              <span style="color:#94a3b8; font-size:0.72rem">
                {str(row.get('timestamp',''))[:19]}
              </span>
              <span><span class="event-pill {sc}">{flag_type_clean}</span></span>
              <span class="{sev_class}">{sev.upper()}</span>
              # REPLACE WITH:
            <span style="color:{COLOR_AMBER}">{f"{row['motion_score']:.2f}" if pd.notnull(row.get('motion_score')) else '–'}</span>
            <span style="color:{COLOR_PURPLE}">{f"{row['audio_score']:.2f}" if pd.notnull(row.get('audio_score')) else '–'}</span>
              <span style="color:#64748b; font-size:0.74rem">{row.get('context','')}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TAB 5 — RISK SCORE BREAKDOWN
# ─────────────────────────────────────────
with tab_score:
    st.markdown('<div class="section-label">RISK SCORE MODEL BREAKDOWN</div>', unsafe_allow_html=True)

    rb1, rb2 = st.columns([1, 2])

    with rb1:
        # Radial summary
        st.markdown(f"""
        <div class="risk-card">
          <div style="font-size:0.7rem; font-weight:600; letter-spacing:0.15em; color:#475569;
                       text-transform:uppercase; margin-bottom:16px;">COMPOSITE RISK SCORE</div>
          <div class="risk-score-display" style="color:{risk_color(risk_cat)}">{risk_score:.0f}</div>
          <div style="font-size:0.75rem; color:#475569; margin-top:4px;">out of 100</div>
          <div class="risk-cat-badge {'risk-low-badge' if risk_cat=='Low' else 'risk-med-badge' if risk_cat=='Medium' else 'risk-high-badge'}">{risk_cat} Risk</div>
          {"<div style='margin-top:14px; font-size:0.75rem; color:#64748b'>ML Validation: " + ml_cat + (" (" + str(confidence) + "% conf)" if confidence else "") + "</div>" if ml_cat else ""}
          <div style="margin-top:24px; border-top:1px solid #e2e8f0; padding-top:16px; text-align:left;">
            <div style="font-size:0.7rem; color:#64748b; margin-bottom:10px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase">Method</div>
            <div style="font-size:0.78rem; color:#475569; line-height:1.6">
              Rule-based scoring (70%) blended with<br>Random Forest classifier (30%).<br><br>
              Rule score is transparent and explainable.<br>
              RF validates the pattern against 500 synthetic driver profiles.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with rb2:
        st.markdown('<div class="chart-card" style="height:100%">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Score Component Contributions</div>', unsafe_allow_html=True)

        max_contrib = 30  # max any single component can contribute

        for label, value in contributions.items():
            is_bonus = value < 0
            fill_pct = min(abs(value) / max_contrib * 100, 100)
            bar_color = "#10b981" if is_bonus else (
                COLOR_RISK_HIGH if abs(value) > 15 else
                COLOR_RISK_MED  if abs(value) > 7  else COLOR_BLUE
            )
            display_val = f"–{abs(value):.1f}" if is_bonus else f"+{value:.1f}"

            st.markdown(f"""
            <div class="contrib-row">
              <span class="contrib-label">{label}</span>
              <div class="contrib-bar-bg">
                <div class="contrib-bar-fill" style="width:{fill_pct}%; background:{bar_color};"></div>
              </div>
              <span class="contrib-score" style="color:{bar_color}">{display_val}</span>
            </div>
            """, unsafe_allow_html=True)

        # Radar chart of normalised features
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        feature_labels = ["Harsh Events", "Overspeed", "Audio Stress", "Arguments", "High Flags"]
        trips = max(profile["total_trips"], 1)
        raw_values = [
            min(1, profile["total_harsh"] / (trips * 3)),
            min(1, profile["total_overspeed"] / (trips * 2)),
            min(1, profile["avg_audio_stress"] / 80),
            min(1, profile["total_arguments"] / trips),
            min(1, profile["high_flags"] / (trips * 2)),
        ]

        fig_radar = go.Figure(go.Scatterpolar(
            r=raw_values + [raw_values[0]],
            theta=feature_labels + [feature_labels[0]],
            fill="toself",
            fillcolor=f"rgba({','.join(str(int(c,16)) for c in [risk_color(risk_cat)[1:3], risk_color(risk_cat)[3:5], risk_color(risk_cat)[5:7]])},0.15)",
            line=dict(color=risk_color(risk_cat), width=2),
            marker=dict(color=risk_color(risk_cat), size=6),
        ))
        fig_radar.update_layout(
            **{**PLOTLY_LAYOUT,
               "height": 240,
               "margin": dict(l=30, r=30, t=30, b=30),
               "polar": dict(
                   bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 1], gridcolor="#e2e8f0",
                                   tickfont=dict(color="#94a3b8", size=8)),
                   angularaxis=dict(gridcolor="#e2e8f0", tickfont=dict(color="#334155", size=10)),
               )},
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Actionable Insights ──
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">PERSONALISED INSIGHTS & RECOMMENDATIONS</div>', unsafe_allow_html=True)

    insights = []

    if profile["total_harsh"] > 2:
        insights.append(("⚡", "Reduce Harsh Braking",
            f"You had {profile['total_harsh']} harsh braking/acceleration events. "
            "Try to anticipate traffic and apply brakes gradually. This reduces wear and improves passenger comfort."))

    if profile["total_overspeed"] > 1:
        insights.append(("🚗", "Watch Your Speed",
            f"{profile['total_overspeed']} overspeed events detected above 55 km/h. "
            "Staying within safe speed ranges protects you and your passengers."))

    if profile["total_arguments"] > 0:
        insights.append(("🔊", "Cabin Tension Detected",
            f"{profile['total_arguments']} conversation(s) classified as argumentative or very loud. "
            "Consider enabling a calm playlist and staying professionally neutral."))

    if profile["avg_smoothness"] < 60:
        insights.append(("🎯", "Improve Driving Smoothness",
            f"Your smoothness index is {profile['avg_smoothness']:.0f}/100. "
            "Avoid sudden lane changes and accelerate/decelerate gently to improve ride quality and ratings."))

    if risk_cat == "Low" and not insights:
        insights.append(("✅", "Great Safe Driving!",
            "Your risk indicators are all within safe ranges. Keep up the excellent driving habits — "
            "smooth acceleration, controlled speed, and a calm cabin environment."))

    if not insights:
        insights.append(("📊", "Keep Monitoring",
            "Continue your current driving patterns. Safety data will build up over more trips "
            "to give you richer insights."))

    for icon, title, body in insights:
        st.markdown(f"""
        <div class="insight-card">
          <div class="insight-icon">{icon}</div>
          <div>
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px; padding:20px 0; border-top:1px solid #e2e8f0;
            display:flex; justify-content:space-between; align-items:center;
            font-size:0.72rem; color:#94a3b8;">
  <span>🛡 Driver Pulse AI · Safety Module v1.0</span>
  <span style="font-family:'Roboto Mono',monospace">
    Signals: ACCELEROMETER · AUDIO · FLAGGED_MOMENTS
  </span>
  <span>Privacy-safe · No audio recording · Numerical features only</span>
</div>
""", unsafe_allow_html=True)