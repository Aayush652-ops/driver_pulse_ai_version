"""
Implementation of the "My Dashboard" page for DriverPulse AI.

This page aggregates a driver's profile, earnings, safety, burnout,
and AI insights into a single summary view. It uses the SAME analytics
modules as the detailed pages so values stay consistent across the app.
"""

import streamlit as st

from utils.data_loader import load_all_data, get_driver_data
from utils.feature_engineering import build_driver_safety_profile
from models.risk_model import compute_risk_score
from utils.earnings_analytics import build_earnings_profile
from utils.burnout_detection import build_burnout_profile


def run(driver_id: str) -> None:
    """Render the dashboard page for a specific driver."""

    st.markdown(
        """
        <style>
          .dash-hero { border-bottom: 2px solid #0a0a0a; padding-bottom: 26px; margin-bottom: 30px; }
          .dash-title { font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 800; line-height: 1.03; color: #0a0a0a; }
          .dash-sub { font-size: 0.8rem; letter-spacing: 0.18em; text-transform: uppercase; color: #9ca3af; margin-top: 8px; }
          .dash-meta { display:flex; gap:10px; flex-wrap:wrap; margin-top: 18px; }
          .dash-pill { background:#f3f4f6; border-radius: 100px; padding: 6px 14px; font-size: 0.78rem; color:#374151; }
          .dash-section { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: #9ca3af; margin: 18px 0 14px; display:flex; align-items:center; gap:12px; }
          .dash-section:after { content:''; flex:1; height:1px; background:#e5e7eb; }
          .dash-card { background:#ffffff; border:1.5px solid #e5e7eb; border-radius:14px; padding:22px 24px; height:100%; }
          .dash-card-dark { background:#0a0a0a; color:#ffffff; border-color:#0a0a0a; }
          .dash-label { font-size:0.68rem; letter-spacing:0.16em; text-transform:uppercase; color:#9ca3af; font-weight:700; margin-bottom:8px; }
          .dash-card-dark .dash-label { color:rgba(255,255,255,0.55); }
          .dash-value { font-family: 'DM Mono', monospace; font-size:2.2rem; font-weight:500; color:#0a0a0a; line-height:1; }
          .dash-card-dark .dash-value { color:#ffffff; }
          .dash-note { font-size:0.82rem; color:#6b7280; margin-top:8px; }
          .dash-card-dark .dash-note { color:rgba(255,255,255,0.65); }
          .dash-chart { background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:18px 20px 8px; }
          .dash-chart-title { font-size:0.88rem; font-weight:700; color:#0a0a0a; margin-bottom:8px; }
          .ai-card { background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:18px 20px; min-height:142px; }
          .ai-icon { font-size:1.2rem; margin-bottom:10px; }
          .ai-title { font-size:0.82rem; font-weight:700; color:#0a0a0a; margin-bottom:8px; }
          .ai-body { font-size:0.86rem; color:#374151; line-height:1.6; }
          .ai-plan { background:#0a0a0a; border-radius:16px; padding:22px 24px; color:#ffffff; margin: 8px 0 22px; }
          .ai-plan-title { font-size:0.72rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; color:rgba(255,255,255,0.55); margin-bottom:10px; }
          .ai-plan-body { font-size:1rem; line-height:1.7; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Unified data loading
    data = load_all_data()
    driver_data = get_driver_data(driver_id, data)
    driver_profile = driver_data.get("driver", {})

    if not driver_profile:
        st.error(f"No driver found with ID {driver_id}.")
        return

    # Basic info
    name = driver_profile.get("name", "Unknown")
    rating = float(driver_profile.get("rating", 0))
    city = driver_profile.get("city", "Unknown")

    driver_trips = driver_data.get("trips")
    driver_flags = driver_data.get("flags")
    driver_goals = driver_data.get("goals")
    driver_velocity = driver_data.get("velocity")

    # SAFETY — same logic as Safety page
    safety_profile = build_driver_safety_profile(
        driver_id,
        data["trips"],
        driver_data["acc"],
        driver_data["aud"],
        driver_flags,
    )
    risk_result = compute_risk_score(safety_profile)
    safety_score = float(risk_result.get("risk_score", 0))
    risk_category = risk_result.get("risk_category", "Unknown")

    # EARNINGS / GOALS — same logic as Earnings page
    earnings_profile = build_earnings_profile(driver_data)
    total_trips = int(earnings_profile.get("trips_completed", 0))
    total_earnings = float(earnings_profile.get("current_earnings", 0))
    current_velocity = float(earnings_profile.get("current_velocity", 0))

    goal_info = earnings_profile.get("goal", {})
    trip_pct = float(goal_info.get("trip_progress_pct", 0))
    earn_pct = float(goal_info.get("progress_pct", 0))

    # If trip_progress_pct isn't available, fall back gracefully
    if trip_pct == 0 and goal_info.get("has_goal", False):
        target_trips = float(goal_info.get("target_trips", 0))
        if target_trips > 0:
            trip_pct = min(100.0, (total_trips / target_trips) * 100)

    # BURNOUT — same logic as Burnout page
    burnout_profile = build_burnout_profile(driver_data)
    burnout_score = float(
        burnout_profile.get("burnout_score", burnout_profile.get("risk_score", 0))
    )
    burnout_level = burnout_profile.get(
        "risk_level", burnout_profile.get("burnout_level", "Low")
    )

    # AI Insights
    insights = []

    avg_fare = 0.0
    fleet_avg_fare = 0.0
    best_window = None

    if driver_trips is not None and not driver_trips.empty and "fare" in driver_trips.columns:
        avg_fare = float(driver_trips["fare"].mean())

    if data["trips"] is not None and not data["trips"].empty and "fare" in data["trips"].columns:
        fleet_avg_fare = float(data["trips"]["fare"].mean())

    if driver_trips is not None and not driver_trips.empty and "start_time" in driver_trips.columns:
        time_df = driver_trips.copy()
        try:
            time_df["hour"] = time_df["start_time"].astype(str).str.slice(0, 2).astype(int)

            def bucket(h):
                if 6 <= h < 11:
                    return "Morning (6–11 AM)"
                if 11 <= h < 17:
                    return "Afternoon (11 AM–5 PM)"
                if 17 <= h < 22:
                    return "Evening (5–10 PM)"
                return "Night (10 PM–6 AM)"

            time_df["window"] = time_df["hour"].apply(bucket)
            best_by_window = time_df.groupby("window")["fare"].mean().sort_values(ascending=False)

            if not best_by_window.empty:
                best_window = best_by_window.index[0]
                insights.append({
                    "icon": "💰",
                    "title": "Peak earning window",
                    "body": f"Your strongest earning window is {best_window}. Average fare there is ₹{best_by_window.iloc[0]:.0f} per trip.",
                })
        except Exception:
            pass

    harsh_count = 0
    audio_count = 0
    if driver_flags is not None and not driver_flags.empty and "flag_type" in driver_flags.columns:
        harsh_count = int(driver_flags[driver_flags["flag_type"].isin(
            ["moderate_brake", "harsh_braking", "harsh_brake"]
        )].shape[0])

        audio_count = int(driver_flags[driver_flags["flag_type"].isin(
            ["audio_spike", "conflict_moment", "sustained_stress"]
        )].shape[0])

    if harsh_count > 2 or safety_score >= 65:
        insights.append({
            "icon": "🛡️",
            "title": "Safety coaching",
            "body": f"{harsh_count} harsh braking events were detected. Smoother braking will help reduce your risk score from {safety_score:.0f}.",
        })
    else:
        insights.append({
            "icon": "✅",
            "title": "Driving quality",
            "body": f"Your current safety profile is {risk_category.lower()} risk with a score of {safety_score:.0f}. Keep the same smooth driving pattern.",
        })

    if burnout_score >= 60:
        insights.append({
            "icon": "🧘",
            "title": "Fatigue warning",
            "body": f"Burnout risk is {burnout_score:.0f}%. A 15–20 minute break before the next long stretch is recommended.",
        })
    else:
        insights.append({
            "icon": "⚡",
            "title": "Energy check",
            "body": f"Burnout risk is currently {burnout_score:.0f}%, which is manageable. Short breaks between trips will keep it low.",
        })

    if goal_info.get("has_goal", False) and "target_earnings" in goal_info:
        target_earnings = float(goal_info.get("target_earnings", 0))
        remaining = max(target_earnings - total_earnings, 0.0)
        insights.append({
            "icon": "🎯",
            "title": "Goal progress",
            "body": f"You have completed {earn_pct:.0f}% of your earnings goal. Roughly ₹{remaining:.0f} remains to hit target.",
        })
    else:
        delta_pct = ((avg_fare - fleet_avg_fare) / fleet_avg_fare * 100) if fleet_avg_fare else 0.0
        direction = "above" if delta_pct >= 0 else "below"
        insights.append({
            "icon": "📊",
            "title": "Fleet benchmark",
            "body": f"Your average fare is ₹{avg_fare:.0f}, which is {abs(delta_pct):.0f}% {direction} the fleet average.",
        })

    insights = insights[:4]

    if goal_info.get("has_goal", False) and "target_earnings" in goal_info:
        target_earnings = float(goal_info.get("target_earnings", 0))
        remaining_earnings = max(target_earnings - total_earnings, 0.0)
        remaining_trips = max(int(round(remaining_earnings / avg_fare)) if avg_fare else 0, 0)
        plan_body = f"Target the {best_window or 'next strong demand window'}. Approx. ₹{remaining_earnings:.0f} remains to hit your goal — about {remaining_trips} more trips at your current average fare."
    else:
        plan_body = f"Prioritize {best_window or 'your strongest demand window'} and keep the current safety pattern steady. Your average fare is ₹{avg_fare:.0f} across {total_trips} trips."

    # Header
    st.markdown(
        f"""
        <div class="dash-hero">
          <div class="dash-title">Driver Dashboard</div>
          <div class="dash-sub">Daily overview · safety · earnings · goals</div>
          <div class="dash-meta">
            <span class="dash-pill">👤 {name}</span>
            <span class="dash-pill">🆔 {driver_id}</span>
            <span class="dash-pill">📍 {city}</span>
            <span class="dash-pill">⭐ {rating:.1f} rating</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs
    st.markdown('<div class="dash-section">Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f"""
        <div class="dash-card dash-card-dark">
          <div class="dash-label">Safety Score</div>
          <div class="dash-value">{safety_score:.1f}</div>
          <div class="dash-note">{risk_category} risk profile</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c2.markdown(
        f"""
        <div class="dash-card">
          <div class="dash-label">Total Trips</div>
          <div class="dash-value">{total_trips}</div>
          <div class="dash-note">Trips analysed so far</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c3.markdown(
        f"""
        <div class="dash-card">
          <div class="dash-label">Total Earnings</div>
          <div class="dash-value">₹{total_earnings:,.0f}</div>
          <div class="dash-note">Across completed trips</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c4.markdown(
        f"""
        <div class="dash-card">
          <div class="dash-label">Current Velocity</div>
          <div class="dash-value">₹{current_velocity:,.0f}</div>
          <div class="dash-note">Current earnings pace per hour</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c5, c6, c7 = st.columns(3)

    c5.markdown(
        f"""
        <div class="dash-card">
          <div class="dash-label">Driver Rating</div>
          <div class="dash-value">{rating:.2f}</div>
          <div class="dash-note">Platform rating out of 5</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c6.markdown(
        f"""
        <div class="dash-card">
          <div class="dash-label">Burnout Risk</div>
          <div class="dash-value">{burnout_score:.1f}%</div>
          <div class="dash-note">{burnout_level} fatigue profile</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c7.markdown(
        f"""
        <div class="dash-card">
          <div class="dash-label">Goal Completion</div>
          <div class="dash-value">{earn_pct:.0f}%</div>
          <div class="dash-note">Trips: {trip_pct:.0f}% · Earnings: {earn_pct:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # AI insights
    st.markdown('<div class="dash-section">AI Insights</div>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="ai-plan"><div class="ai-plan-title">Today's action plan</div><div class="ai-plan-body">{plan_body}</div></div>""",
        unsafe_allow_html=True,
    )

    i1, i2, i3, i4 = st.columns(4)
    for col, insight in zip([i1, i2, i3, i4], insights):
        col.markdown(
            f"""
            <div class="ai-card">
              <div class="ai-icon">{insight['icon']}</div>
              <div class="ai-title">{insight['title']}</div>
              <div class="ai-body">{insight['body']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Trends
    st.markdown('<div class="dash-section">Trends</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="dash-chart"><div class="dash-chart-title">Earnings Velocity Trend</div>', unsafe_allow_html=True)
        if driver_velocity is not None and not driver_velocity.empty and "timestamp" in driver_velocity.columns and "earnings_velocity" in driver_velocity.columns:
            chart_data = driver_velocity.copy()
            chart_data = chart_data.sort_values("timestamp")
            chart_data = chart_data.set_index("timestamp")
            st.line_chart(chart_data["earnings_velocity"], use_container_width=True)
        else:
            st.info("Earnings velocity data is not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="dash-chart"><div class="dash-chart-title">Safety Event Breakdown</div>', unsafe_allow_html=True)
        if driver_flags is not None and not driver_flags.empty and "flag_type" in driver_flags.columns:
            event_counts = driver_flags["flag_type"].value_counts().rename_axis("event").reset_index(name="count")
            st.bar_chart(event_counts.set_index("event")["count"], use_container_width=True)
        else:
            st.info("No flagged safety events recorded.")
        st.markdown('</div>', unsafe_allow_html=True)
