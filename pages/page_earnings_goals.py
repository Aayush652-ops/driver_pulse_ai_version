"""
3_💰_Earnings_Goals.py
====================
Driver Pulse — Earnings & Goals Page
Owned by: Aanvi (Earnings & Goal Tracking Lead)

What this page does:
  1. Tracks real-time earnings velocity (₹/hour)
  2. Monitors goal progress and forecasts achievement
  3. Analyzes best earning hours and patterns
  4. Provides actionable insights to help drivers meet targets

Design Aesthetic:
  Clean professional dashboard with focus on actionable metrics.
  Green for positive/ahead, amber for caution, red for at-risk.
  Interactive charts with Plotly.
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
from utils.earnings_analytics import build_earnings_profile

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

  /* KPI Card Styling */
  .kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  }

  .kpi-card-green {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  }

  .kpi-card-amber {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  }

  .kpi-card-red {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  }

  .kpi-label {
    font-size: 0.875rem;
    font-weight: 500;
    opacity: 0.9;
    margin-bottom: 0.5rem;
  }

  .kpi-value {
    font-size: 2.25rem;
    font-weight: 700;
    font-family: 'Roboto Mono', monospace;
  }

  .kpi-subtitle {
    font-size: 0.875rem;
    opacity: 0.8;
    margin-top: 0.5rem;
  }

  /* Prediction Badge */
  .pred-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.875rem;
  }

  .pred-achieved { background: #10b981; color: white; }
  .pred-ahead { background: #10b981; color: white; }
  .pred-on-track { background: #3b82f6; color: white; }
  .pred-possible { background: #f59e0b; color: white; }
  .pred-at-risk { background: #ef4444; color: white; }
  .pred-unlikely { background: #dc2626; color: white; }

  /* Hour Badge */
  .hour-badge {
    background: #f3f4f6;
    padding: 0.75rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
  }

  .hour-badge-title {
    font-weight: 600;
    font-size: 1rem;
    color: #0f172a;
  }

  .hour-badge-subtitle {
    font-size: 0.875rem;
    color: #64748b;
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

# Check if driver has earnings data
has_earnings_data = not driver_data.get('velocity', pd.DataFrame()).empty or \
                    not driver_data.get('goals', pd.DataFrame()).empty

if not has_earnings_data:
    st.error(f"❌ No earnings data available for driver {selected_driver}")
    st.info("This driver may not have earnings tracking enabled yet.")
    st.stop()

# Build earnings profile
earnings_profile = build_earnings_profile(driver_data)

# Use a single consistent earnings value
current_earnings = earnings_profile.get(
    "current_earnings",
    earnings_profile.get("goal", {}).get("current_earnings", 0)
)

# ──────────────────────────────────────────────────────────────────
#  COMPUTE PREDICTION — must happen before header renders
# ──────────────────────────────────────────────────────────────────
goal_progress = earnings_profile['goal']
prediction = earnings_profile['prediction']
target = goal_progress.get("target_earnings", 0)
projected_final = prediction.get("projected_final_earnings", 0)

# prediction['prediction'] is already set correctly by earnings_analytics.py
# Do NOT override it here — that would de-sync it from confidence & recommendation
prediction_status = prediction['prediction']

# ──────────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💰 Earnings & Goals Dashboard")
    st.markdown(f"**Driver:** {driver_info.get('name', selected_driver)} · **Rating:** {driver_info.get('rating', 'N/A')} ⭐")

with col2:
    if earnings_profile['goal']['has_goal']:
        # Use the computed prediction status so header matches the forecast section
        status_icons = {
            'ACHIEVED': '✅', 'AHEAD': '🚀', 'ON_TRACK': '🎯',
            'POSSIBLE': '⚠️', 'AT_RISK': '🔴', 'UNLIKELY': '❌', 'MISSED': '❌'
        }
        icon = status_icons.get(prediction_status, '🎯')
        label = prediction_status.replace('_', ' ').title()
        st.markdown(f"### {icon} {label}")

st.divider()

# ──────────────────────────────────────────────────────────────────
#  TOP KPI CARDS
# ──────────────────────────────────────────────────────────────────

# Determine card colors based on prediction
if prediction['prediction'] in ['ACHIEVED', 'AHEAD']:
    velocity_color = 'green'
    goal_color = 'green'
elif prediction['prediction'] in ['ON_TRACK', 'POSSIBLE']:
    velocity_color = 'amber'
    goal_color = 'amber'
else:
    velocity_color = 'red'
    goal_color = 'red'

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">TOTAL EARNINGS TODAY</div>
        <div class="kpi-value">₹{earnings_profile['current_earnings']:.2f}</div>
        <div class="kpi-subtitle">{earnings_profile['trips_completed']:.0f} trips completed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    target_velocity = prediction.get(
        "target_velocity",
        prediction.get("required_velocity", 0)
    )

    st.markdown(f"""
    <div class="kpi-card kpi-card-{velocity_color}">
        <div class="kpi-label">CURRENT VELOCITY</div>
        <div class="kpi-value">₹{earnings_profile['current_velocity']:.2f}/hr</div>
        <div class="kpi-subtitle">Target: ₹{target_velocity:.2f}/hr</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if goal_progress['has_goal']:
        st.markdown(f"""
        <div class="kpi-card kpi-card-{goal_color}">
            <div class="kpi-label">GOAL PROGRESS</div>
            <div class="kpi-value">{goal_progress['progress_pct']:.0f}%</div>
            <div class="kpi-subtitle">₹{current_earnings:.0f} / ₹{goal_progress['target_earnings']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">GOAL PROGRESS</div>
            <div class="kpi-value">--</div>
            <div class="kpi-subtitle">No goal set</div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card kpi-card-{goal_color}">
        <div class="kpi-label">PROJECTED FINAL</div>
        <div class="kpi-value">₹{prediction['projected_final_earnings']:.0f}</div>
        <div class="kpi-subtitle">End of shift forecast</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
#  GOAL ACHIEVEMENT FORECAST
# ──────────────────────────────────────────────────────────────────
if goal_progress['has_goal']:
    st.markdown("### 🎯 Goal Achievement Forecast")

    col1, col2 = st.columns([2, 1])

    with col1:
        pred_class = prediction['prediction'].lower().replace('_', '-')
        confidence_pct = prediction['confidence'] * 100

        st.markdown(f"""
        <div style="padding: 1.5rem; background: #f9fafb; border-radius: 12px; border-left: 4px solid #667eea;">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span class="pred-badge pred-{pred_class}">{prediction['prediction']}</span>
                <span style="font-size: 1.25rem; font-weight: 600;">Confidence: {confidence_pct:.0f}%</span>
            </div>
            <p style="font-size: 1.125rem; margin: 0.5rem 0;">
                <strong>📊 {prediction['recommendation']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Gauge chart for goal progress
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=goal_progress['progress_pct'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Goal Progress", 'font': {'size': 18}},
            delta={'reference': 100, 'increasing': {'color': "#10b981"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "#fee2e2"},
                    {'range': [50, 80], 'color': "#fef3c7"},
                    {'range': [80, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))

        fig_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#0f172a", 'family': "Inter"}
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

# ──────────────────────────────────────────────────────────────────
#  MAIN CHARTS
# ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Earnings Timeline", "⏰ Hourly Breakdown", "🚗 Trip Details"])

with tab1:
    st.markdown("### Earnings Over Time")

    velocity_log = earnings_profile['velocity_log']

    if not velocity_log.empty:
        fig = go.Figure()

        # Actual cumulative earnings
        fig.add_trace(go.Scatter(
            x=velocity_log['timestamp'],
            y=velocity_log['cumulative_earnings'],
            mode='lines+markers',
            name='Actual Earnings',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Time:</b> %{x}<br><b>Earnings:</b> ₹%{y:.2f}<extra></extra>'
        ))

        # Target goal line
        if goal_progress['has_goal']:
            fig.add_hline(
                y=goal_progress['target_earnings'],
                line_dash="dash",
                line_color="#10b981",
                annotation_text=f"Goal: ₹{goal_progress['target_earnings']:.0f}",
                annotation_position="right"
            )

            # Projected trajectory
            last_timestamp = velocity_log['timestamp'].iloc[-1]
            last_earnings = velocity_log['cumulative_earnings'].iloc[-1]

            fig.add_trace(go.Scatter(
                x=[last_timestamp, last_timestamp],
                y=[last_earnings, prediction['projected_final_earnings']],
                mode='lines',
                name='Projected',
                line=dict(color='#f59e0b', width=2, dash='dot'),
                hovertemplate='<b>Projected:</b> ₹%{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Cumulative Earnings (₹)",
            hovermode='x unified',
            template='plotly_white',
            font=dict(family="Inter", color="#0f172a"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Velocity chart
        st.markdown("### Earnings Velocity (₹/hour)")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=velocity_log['timestamp'],
            y=velocity_log['current_velocity'],
            mode='lines+markers',
            name='Current Velocity',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='<b>Time:</b> %{x}<br><b>Velocity:</b> ₹%{y:.2f}/hr<extra></extra>'
        ))

        if goal_progress['has_goal']:
            target_velocity = prediction.get("target_velocity", prediction.get("required_velocity", 0))

            fig2.add_hline(
            y=target_velocity,
            line_dash="dash",
            line_color="#10b981",
            annotation_text=f"Target: ₹{target_velocity:.2f}"
)

        fig2.update_layout(
            height=300,
            xaxis_title="Time",
            yaxis_title="Velocity (₹/hour)",
            template='plotly_white',
            font=dict(family="Inter", color="#0f172a"),
            showlegend=False
        )

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No velocity data available yet")

with tab2:
    st.markdown("### Best Earning Hours")

    best_hours = earnings_profile['best_hours']

    if best_hours:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 🏆 Top Performing Hours")
            for i, hour_data in enumerate(best_hours, 1):
                st.markdown(f"""
                <div class="hour-badge">
                    <div class="hour-badge-title">#{i} · {hour_data['hour_range']}</div>
                    <div class="hour-badge-subtitle">
                        💰 ₹{hour_data['total_earnings']:.2f} ·
                        🚗 {hour_data['trip_count']} trips ·
                        📊 ₹{hour_data['earnings_per_trip']:.2f}/trip
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # Bar chart of hourly earnings
            trips = earnings_profile['trips']
            if not trips.empty and 'fare' in trips.columns:
                trips_copy = trips.copy()
                trips_copy['hour'] = pd.to_datetime(trips_copy['date']).dt.hour
                hourly_data = trips_copy.groupby('hour')['fare'].agg(['sum', 'count']).reset_index()
                hourly_data.columns = ['hour', 'total_earnings', 'trip_count']

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=hourly_data['hour'],
                    y=hourly_data['total_earnings'],
                    marker_color='#667eea',
                    hovertemplate='<b>Hour:</b> %{x}:00<br><b>Earnings:</b> ₹%{y:.2f}<extra></extra>'
                ))

                fig.update_layout(
                    title="Earnings by Hour of Day",
                    xaxis_title="Hour",
                    yaxis_title="Total Earnings (₹)",
                    height=400,
                    template='plotly_white',
                    font=dict(family="Inter", color="#0f172a")
                )

                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough trip data to analyze hourly patterns")

with tab3:
    st.markdown("### Trip-by-Trip Earnings Log")

    trips = earnings_profile['trips']

    if not trips.empty:
        # Build display dataframe — use start_time for time column if available
        time_col = 'start_time' if 'start_time' in trips.columns else 'date'
        display_trips = trips[['trip_id', time_col, 'fare', 'distance_km', 'pickup_location', 'dropoff_location']].copy()
        display_trips[time_col] = pd.to_datetime(display_trips[time_col], errors='coerce').dt.strftime('%H:%M:%S')
        display_trips['fare'] = display_trips['fare'].apply(lambda x: f"₹{x:.2f}")
        display_trips['distance_km'] = display_trips['distance_km'].apply(lambda x: f"{x:.2f} km")

        display_trips.columns = ['Trip ID', 'Time', 'Fare', 'Distance', 'Pickup', 'Dropoff']

        st.dataframe(display_trips, use_container_width=True, height=400)

        # Summary stats — use raw numeric trips for calculations
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trips", len(trips))
        col2.metric("Avg Fare", f"₹{trips['fare'].mean():.2f}")
        col3.metric("Total Distance", f"{trips['distance_km'].sum():.1f} km")
        col4.metric("Avg Distance", f"{trips['distance_km'].mean():.2f} km")
    else:
        st.info("No trip data available")

st.divider()

# ──────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0; font-size: 0.875rem;">
    Driver Pulse · Earnings & Goals Dashboard<br>
    Stay on track with real-time insights
</div>
""", unsafe_allow_html=True)
