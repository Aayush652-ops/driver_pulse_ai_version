"""
Implementation of the "My Dashboard" page for DriverPulse AI.

This page aggregates a driver's profile, trip summaries, earnings and
safety data into a succinct overview. Key performance indicators are
styled to match the editorial look of the Driving Behaviour page.
"""

import streamlit as st

from utils.data_loader import load_data, filter_by_driver
from utils.analytics import (
    calculate_safety_score,
    calculate_burnout_probability,
    calculate_goal_progress,
)


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

    # Load datasets
    drivers = load_data("drivers.csv")
    trips = load_data("trip_summaries.csv")
    raw_trips = load_data("trips.csv")
    earnings_log = load_data("earnings_velocity_log.csv")
    flagged = load_data("flagged_moments.csv")
    accel = load_data("accelerometer_data.csv")
    audio = load_data("audio_intensity_data.csv")
    goals = load_data("driver_goals.csv")

    # Filter by driver ID
    driver_profile = filter_by_driver(drivers, driver_id)
    driver_trips = filter_by_driver(trips, driver_id)
    driver_raw_trips = filter_by_driver(raw_trips, driver_id)
    driver_earnings_log = filter_by_driver(earnings_log, driver_id)
    driver_flagged = filter_by_driver(flagged, driver_id)
    driver_accel = filter_by_driver(accel, driver_id)
    driver_audio = filter_by_driver(audio, driver_id)
    driver_goals = filter_by_driver(goals, driver_id)

    if driver_profile.empty:
        st.error(f"No driver found with ID {driver_id}.")
        return

    # Extract basic profile info
    name = driver_profile.iloc[0].get("name", "Unknown")
    rating = float(driver_profile.iloc[0].get("rating", 0))
    city = driver_profile.iloc[0].get("city", "Unknown")

    # Compute KPI values
    total_trips = int(len(driver_trips))
    if not driver_trips.empty:
        if "earnings" in driver_trips.columns:
            total_earnings = float(driver_trips["earnings"].astype(float).sum())
        elif "fare" in driver_trips.columns:
            total_earnings = float(driver_trips["fare"].astype(float).sum())
        else:
            total_earnings = 0.0
    else:
        total_earnings = 0.0

    current_week_earnings = (
        float(driver_earnings_log["earnings_velocity"].astype(float).iloc[-1])
        if not driver_earnings_log.empty
        else 0.0
    )

    safety_score, risk_category = calculate_safety_score(driver_flagged, driver_accel, driver_audio)
    burnout_prob = calculate_burnout_probability(driver_earnings_log, driver_trips)
    trip_pct, earn_pct = calculate_goal_progress(driver_trips, driver_goals)

    # Build AI insights
    insights = []
    avg_fare = float(driver_trips["fare"].mean()) if (not driver_trips.empty and "fare" in driver_trips.columns) else 0.0
    fleet_avg_fare = float(trips["fare"].mean()) if (not trips.empty and "fare" in trips.columns) else 0.0

    best_window = None
    if not driver_raw_trips.empty and "start_time" in driver_raw_trips.columns:
        time_df = driver_raw_trips.copy()
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

    harsh_count = 0
    audio_count = 0
    if not driver_flagged.empty and "flag_type" in driver_flagged.columns:
        harsh_count = int(driver_flagged[driver_flagged["flag_type"].isin(["moderate_brake", "harsh_braking", "harsh_brake"])].shape[0])
        audio_count = int(driver_flagged[driver_flagged["flag_type"].isin(["audio_spike", "conflict_moment", "sustained_stress"])].shape[0])

    if harsh_count > 2 or safety_score < 60:
        insights.append({
            "icon": "🛡️",
            "title": "Safety coaching",
            "body": f"{harsh_count} harsh braking events were detected. Smoother braking will help lift your safety score from {safety_score:.0f}.",
        })
    else:
        insights.append({
            "icon": "✅",
            "title": "Driving quality",
            "body": f"Your current safety profile is {risk_category.lower()} risk with a score of {safety_score:.0f}. Keep the same smooth driving pattern.",
        })

    if burnout_prob >= 60:
        insights.append({
            "icon": "🧘",
            "title": "Fatigue warning",
            "body": f"Burnout risk is {burnout_prob:.0f}%. A 15–20 minute break before the next long stretch is recommended.",
        })
    else:
        insights.append({
            "icon": "⚡",
            "title": "Energy check",
            "body": f"Burnout risk is currently {burnout_prob:.0f}%, which is manageable. Short breaks between trips will keep it low.",
        })

    if earn_pct < 100 and not driver_goals.empty and "target_earnings" in driver_goals.columns:
        target_earnings = float(driver_goals.iloc[0]["target_earnings"])
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

    if not driver_goals.empty and "target_earnings" in driver_goals.columns:
        target_earnings = float(driver_goals.iloc[0]["target_earnings"])
        remaining_earnings = max(target_earnings - total_earnings, 0.0)
        remaining_trips = max(int(round(remaining_earnings / avg_fare)) if avg_fare else 0, 0)
        plan_body = f"Target the {best_window or 'next strong demand window'}. Approx. ₹{remaining_earnings:.0f} remains to hit your goal — about {remaining_trips} more trips at your current average fare."
    else:
        plan_body = f"Prioritize {best_window or 'your strongest demand window'} and keep the current safety pattern steady. Your average fare is ₹{avg_fare:.0f} across {total_trips} trips."

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
          <div class="dash-label">Weekly Earnings</div>
          <div class="dash-value">₹{current_week_earnings:,.0f}</div>
          <div class="dash-note">Latest velocity log snapshot</div>
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
          <div class="dash-value">{burnout_prob:.1f}%</div>
          <div class="dash-note">Estimated from trips + earnings pace</div>
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

    st.markdown('<div class="dash-section">Trends</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="dash-chart"><div class="dash-chart-title">Earnings Velocity Trend</div>', unsafe_allow_html=True)
        if not driver_earnings_log.empty:
            chart_data = driver_earnings_log.copy()
            chart_data["week"] = chart_data["week"].astype(str)
            chart_data = chart_data.set_index("week")
            st.line_chart(chart_data["earnings_velocity"], use_container_width=True)
        else:
            st.info("Earnings velocity data is not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="dash-chart"><div class="dash-chart-title">Safety Event Breakdown</div>', unsafe_allow_html=True)
        event_col = None
        for candidate in ["event_type", "flag_type"]:
            if candidate in driver_flagged.columns:
                event_col = candidate
                break
        if not driver_flagged.empty and event_col:
            event_counts = driver_flagged[event_col].value_counts().rename_axis("event").reset_index(name="count")
            st.bar_chart(event_counts.set_index("event")["count"], use_container_width=True)
        else:
            st.info("No flagged safety events recorded.")
        st.markdown('</div>', unsafe_allow_html=True)
