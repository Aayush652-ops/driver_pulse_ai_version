"""
4_🔥_Burnout_Monitor.py
======================
Driver Pulse — Burnout Risk Monitoring Page
Owned by: Aanvi (Burnout Detection Lead)

What this page does:
  1. Monitors burnout risk using multi-factor analysis
  2. Combines work intensity, stress signals, and earnings pressure
  3. Provides personalized wellness recommendations
  4. Helps drivers maintain sustainable work patterns

Design Aesthetic:
  Wellness-focused design with clear risk indicators.
  Traffic light colors: Green (good), Amber (caution), Red (danger).
  Emphasis on actionable recommendations.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_loader import load_all_data, get_driver_data, get_drivers_with_sensor_data
from utils.burnout_detection import build_burnout_profile

# ──────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
# Page config is handled in app.py


# ──────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap');

  .stApp {
    background: #ffffff;
    color: #0f172a;
    font-family: 'Inter', sans-serif;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1400px; }

  /* Risk Level Badge */
  .risk-badge {
    display: inline-block;
    padding: 1rem 2rem;
    border-radius: 12px;
    font-size: 1.5rem;
    font-weight: 700;
    text-align: center;
    margin: 1rem 0;
  }

  .risk-low {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
  }

  .risk-moderate {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
  }

  .risk-high {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: white;
  }

  /* Recommendation Card */
  .rec-card {
    background: #f9fafb;
    padding: 1.25rem;
    border-radius: 10px;
    margin: 0.75rem 0;
    border-left: 4px solid #cbd5e1;
  }

  .rec-card-critical {
    border-left-color: #dc2626;
    background: #fef2f2;
  }

  .rec-card-high {
    border-left-color: #ef4444;
    background: #fef2f2;
  }

  .rec-card-medium {
    border-left-color: #f59e0b;
    background: #fffbeb;
  }

  .rec-card-low {
    border-left-color: #10b981;
    background: #f0fdf4;
  }

  .rec-title {
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .rec-message {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 0.5rem;
  }

  .rec-action {
    font-size: 0.875rem;
    font-weight: 600;
    color: #1e293b;
    background: #e2e8f0;
    padding: 0.5rem;
    border-radius: 6px;
    display: inline-block;
  }

  /* Factor Card */
  .factor-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border: 2px solid #e2e8f0;
    text-align: center;
  }

  .factor-score {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'Roboto Mono', monospace;
  }

  .factor-label {
    font-size: 1rem;
    font-weight: 600;
    color: #64748b;
    margin-top: 0.5rem;
  }

  /* Stats Row */
  .stat-item {
    display: flex;
    justify-content: space-between;
    padding: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
  }

  .stat-label {
    font-weight: 500;
    color: #64748b;
  }

  .stat-value {
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    color: #0f172a;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
#  LOAD DATA
# ──────────────────────────────────────────────────────────────────
data = load_all_data()

# ──────────────────────────────────────────────────────────────────
#  DRIVER SELECTION
# ──────────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in", False):
    available_drivers = get_drivers_with_sensor_data(data)
    selected_driver = st.selectbox(
        "Choose Driver",
        available_drivers,
        index=available_drivers.index("SDRV039") if "SDRV039" in available_drivers else 0
    )
    st.session_state["driver_id"] = selected_driver
else:
    selected_driver = st.session_state.get("driver_id", "SDRV039")

# ──────────────────────────────────────────────────────────────────
#  GET DRIVER DATA & BUILD PROFILE
# ──────────────────────────────────────────────────────────────────
driver_data = get_driver_data(selected_driver, data)
driver_info = driver_data.get('driver', {})

# Build burnout profile
burnout_profile = build_burnout_profile(driver_data)

# ──────────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────────
st.title("🔥 Burnout Risk Monitor")
st.markdown(f"**Driver:** {driver_info.get('name', selected_driver)} · **Rating:** {driver_info.get('rating', 'N/A')} ⭐")

st.divider()

# ──────────────────────────────────────────────────────────────────
#  BURNOUT RISK SCORE
# ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    # Risk level badge
    risk_class = burnout_profile['risk_level'].lower()
    st.markdown(f"""
    <div class="risk-badge risk-{risk_class}">
        {burnout_profile['risk_level']} RISK
    </div>
    <div style="text-align: center; font-size: 1.25rem; margin-top: 1rem;">
        Primary Factor: <strong>{burnout_profile['primary_factor']}</strong>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Gauge chart for burnout score
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=burnout_profile['burnout_score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Burnout Risk Score", 'font': {'size': 20}},
        number={'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': burnout_profile['risk_color']},
            'steps': [
                {'range': [0, 40], 'color': "#d1fae5"},
                {'range': [40, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig_gauge.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#0f172a", 'family': "Inter"}
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 Risk Breakdown", "📈 Detailed Metrics"])

with tab1:
    st.markdown("### Personalized Wellness Recommendations")

    recommendations = burnout_profile['recommendations']

    if recommendations:
        for rec in recommendations:
            priority_class = rec['priority']
            st.markdown(f"""
            <div class="rec-card rec-card-{priority_class}">
                <div class="rec-title">
                    <span>{rec['icon']}</span>
                    <span>{rec['title']}</span>
                </div>
                <div class="rec-message">{rec['message']}</div>
                <div class="rec-action">💡 {rec['action']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✨ No specific recommendations right now. You're managing your work well!")

with tab2:
    st.markdown("### Risk Factor Breakdown")

    # Risk factors as cards
    col1, col2, col3 = st.columns(3)

    factors = burnout_profile['risk_factors']

    with col1:
        work_score = factors['Work Intensity']
        color = "#10b981" if work_score < 40 else "#f59e0b" if work_score < 70 else "#ef4444"
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-score" style="color: {color};">{work_score:.0f}</div>
            <div class="factor-label">Work Intensity</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        stress_score = factors['Stress Signals']
        color = "#10b981" if stress_score < 40 else "#f59e0b" if stress_score < 70 else "#ef4444"
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-score" style="color: {color};">{stress_score:.0f}</div>
            <div class="factor-label">Stress Signals</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pressure_score = factors['Earnings Pressure']
        color = "#10b981" if pressure_score < 40 else "#f59e0b" if pressure_score < 70 else "#ef4444"
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-score" style="color: {color};">{pressure_score:.0f}</div>
            <div class="factor-label">Earnings Pressure</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart
    categories = list(factors.keys())
    values = list(factors.values())

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode='linear',
                tick0=0,
                dtick=20
            )
        ),
        height=400,
        title="Risk Factor Profile",
        font=dict(family="Inter", color="#0f172a"),
        showlegend=False
    )

    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.markdown("### Detailed Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💼 Work Intensity")
        work = burnout_profile['work_intensity']

        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 10px; border: 2px solid #e2e8f0;">
            <div class="stat-item">
                <span class="stat-label">Hours Driving</span>
                <span class="stat-value">{work['hours_driving']:.1f} hrs</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Trips Completed</span>
                <span class="stat-value">{work['trips_completed']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Trips per Hour</span>
                <span class="stat-value">{work['trips_per_hour']:.1f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Has Taken Break</span>
                <span class="stat-value">{'✅ Yes' if work['has_taken_break'] else '❌ No'}</span>
            </div>
            <div class="stat-item" style="border-bottom: none;">
                <span class="stat-label">Intensity Score</span>
                <span class="stat-value">{work['intensity_score']:.0f}/100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 💰 Earnings Pressure")
        pressure = burnout_profile['earnings_pressure']

        if pressure['has_goal']:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; border: 2px solid #e2e8f0;">
                <div class="stat-item">
                    <span class="stat-label">Goal Gap</span>
                    <span class="stat-value">₹{pressure['goal_gap']:.2f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Goal Gap %</span>
                    <span class="stat-value">{pressure['goal_gap_pct']:.1f}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Behind Pace</span>
                    <span class="stat-value">{'⚠️ Yes' if pressure['behind_pace'] else '✅ No'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Velocity Declining</span>
                    <span class="stat-value">{'⚠️ Yes' if pressure['velocity_declining'] else '✅ No'}</span>
                </div>
                <div class="stat-item" style="border-bottom: none;">
                    <span class="stat-label">Pressure Score</span>
                    <span class="stat-value">{pressure['pressure_score']:.0f}/100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No goal set - no earnings pressure detected")

    with col2:
        st.markdown("#### 😰 Stress Signals")
        stress = burnout_profile['stress_signals']

        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 10px; border: 2px solid #e2e8f0;">
            <div class="stat-item">
                <span class="stat-label">Harsh Events</span>
                <span class="stat-value">{stress['harsh_events']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Audio Spikes</span>
                <span class="stat-value">{stress['audio_spikes']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Conflict Moments</span>
                <span class="stat-value">{stress['conflict_moments']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Stress Events</span>
                <span class="stat-value">{stress['total_stress_events']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Harsh Events/Hour</span>
                <span class="stat-value">{stress['harsh_events_per_hour']:.1f}</span>
            </div>
            <div class="stat-item" style="border-bottom: none;">
                <span class="stat-label">Stress Score</span>
                <span class="stat-value">{stress['stress_score']:.0f}/100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Bar chart of risk factors
        fig_bars = go.Figure()

        factor_names = list(factors.keys())
        factor_values = list(factors.values())
        colors = ['#10b981' if v < 40 else '#f59e0b' if v < 70 else '#ef4444' for v in factor_values]

        fig_bars.add_trace(go.Bar(
            x=factor_values,
            y=factor_names,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{v:.0f}" for v in factor_values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.0f}<extra></extra>'
        ))

        fig_bars.update_layout(
            title="Risk Factor Comparison",
            xaxis_title="Score (0-100)",
            height=250,
            margin=dict(l=150, r=20, t=50, b=50),
            template='plotly_white',
            font=dict(family="Inter", color="#0f172a"),
            showlegend=False
        )

        st.plotly_chart(fig_bars, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────────
#  WELLNESS TIPS
# ──────────────────────────────────────────────────────────────────
st.markdown("### 💚 General Wellness Tips")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: #f0fdf4; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10b981;">
        <h4 style="margin: 0 0 0.5rem 0;">🧘 Take Regular Breaks</h4>
        <p style="margin: 0; color: #475569;">
            Take a 15-minute break every 3 hours. Stretch, hydrate, and rest your eyes.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: #f0fdf4; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10b981;">
        <h4 style="margin: 0 0 0.5rem 0;">🚗 Drive Calmly</h4>
        <p style="margin: 0; color: #475569;">
            Smooth driving reduces stress and improves safety. Focus on steady acceleration.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: #f0fdf4; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10b981;">
        <h4 style="margin: 0 0 0.5rem 0;">🎯 Set Realistic Goals</h4>
        <p style="margin: 0; color: #475569;">
            Balance your earnings goals with your well-being. It's okay to adjust targets.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0; font-size: 0.875rem;">
    Driver Pulse · Burnout Risk Monitor<br>
    Your well-being matters · Drive safe, drive healthy
</div>
""", unsafe_allow_html=True)
