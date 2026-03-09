"""
burnout_detection.py
-------------------
Burnout risk detection combining work intensity, stress signals, and earnings pressure.
Uses multi-factor scoring to identify drivers at risk of burnout.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_work_intensity(driver_data: dict) -> dict:
    """
    Calculate work intensity metrics from trip and velocity data.

    Args:
        driver_data: Dictionary from data_loader.get_driver_data()

    Returns:
        Dictionary with work intensity metrics
    """
    trips = driver_data.get('trips', pd.DataFrame())
    velocity_log = driver_data.get('velocity', pd.DataFrame())

    if trips.empty:
        return {
            'hours_driving': 0,
            'trips_completed': 0,
            'trips_per_hour': 0,
            'consecutive_hours': 0,
            'minutes_since_last_break': 0,
            'has_taken_break': False,
            'intensity_score': 0
        }

    # Get elapsed hours — velocity log is preferred, but fall back to
    # summing actual trip durations from trips table when log is unavailable
    if not velocity_log.empty and 'elapsed_hours' in velocity_log.columns:
        elapsed_hours = velocity_log['elapsed_hours'].iloc[-1]
    elif 'duration_min' in trips.columns:
        elapsed_hours = trips['duration_min'].sum() / 60
    elif 'start_time' in trips.columns and 'end_time' in trips.columns:
        try:
            trips_t = trips.copy()
            trips_t['start_dt'] = pd.to_datetime(trips_t['date'].astype(str) + ' ' + trips_t['start_time'].astype(str), errors='coerce')
            trips_t['end_dt']   = pd.to_datetime(trips_t['date'].astype(str) + ' ' + trips_t['end_time'].astype(str), errors='coerce')
            elapsed_hours = (trips_t['end_dt'] - trips_t['start_dt']).dt.total_seconds().sum() / 3600
        except Exception:
            elapsed_hours = len(trips) * 0.5  # rough fallback: ~30min per trip
    else:
        elapsed_hours = len(trips) * 0.5  # rough fallback

    trips_completed = len(trips)

    # Calculate trips per hour
    trips_per_hour = trips_completed / elapsed_hours if elapsed_hours > 0 else 0

    # Estimate break time (gap between trips)
    if 'date' in trips.columns and len(trips) > 1:
        trips_sorted = trips.sort_values('date')
        trips_sorted['date'] = pd.to_datetime(trips_sorted['date'])

        # Calculate gaps between trips
        trips_sorted['gap_minutes'] = trips_sorted['date'].diff().dt.total_seconds() / 60

        # Consider gap > 15 minutes as a break
        break_gaps = trips_sorted[trips_sorted['gap_minutes'] > 15]
        has_taken_break = len(break_gaps) > 0

        # Time since last trip (potential ongoing time without break)
        if not trips_sorted.empty:
            last_trip_time = trips_sorted['date'].iloc[-1]
            # For demo, assume current time is shortly after last trip
            minutes_since_last_break = trips_sorted['gap_minutes'].iloc[-1] if len(trips_sorted) > 1 else 0
            if pd.isna(minutes_since_last_break):
                minutes_since_last_break = 0
        else:
            minutes_since_last_break = 0
    else:
        has_taken_break = False
        minutes_since_last_break = 0

    # Calculate intensity score (0-100)
    # NOTE: trips.csv has partial shift data (1-2 trips). Score on RATE not absolute hours,
    # and use goals CSV for full-shift context when available.
    intensity_score = 0

    # Hours driving — use goals CSV current_hours for full shift context if available
    goals = driver_data.get('goals', pd.DataFrame())
    if not goals.empty and 'current_hours' in goals.columns:
        shift_hours = float(goals.iloc[-1].get('current_hours', elapsed_hours))
    else:
        shift_hours = elapsed_hours

    # Hours into shift (max 40 points)
    if shift_hours > 8:
        intensity_score += 40
    elif shift_hours > 6:
        intensity_score += 30
    elif shift_hours > 4:
        intensity_score += 20
    elif shift_hours > 2:
        intensity_score += 10

    # Trips per hour — rate based (max 30 points)
    if trips_per_hour > 4:
        intensity_score += 30
    elif trips_per_hour > 3:
        intensity_score += 20
    elif trips_per_hour > 2:
        intensity_score += 10

    # Break time (max 30 points — penalty for no breaks during long shifts)
    if not has_taken_break and shift_hours > 4:
        intensity_score += 30
    elif not has_taken_break and shift_hours > 2:
        intensity_score += 15

    return {
        'hours_driving': elapsed_hours,
        'trips_completed': trips_completed,
        'trips_per_hour': trips_per_hour,
        'consecutive_hours': elapsed_hours,  # Simplified - assume continuous
        'minutes_since_last_break': minutes_since_last_break,
        'has_taken_break': has_taken_break,
        'intensity_score': min(intensity_score, 100)
    }


def calculate_stress_signals(driver_data: dict) -> dict:
    """
    Aggregate stress signals from sensor data.

    Args:
        driver_data: Dictionary from data_loader.get_driver_data()

    Returns:
        Dictionary with stress signal metrics
    """
    flags = driver_data.get('flags', pd.DataFrame())

    if flags.empty:
        return {
            'harsh_events': 0,
            'audio_spikes': 0,
            'conflict_moments': 0,
            'total_stress_events': 0,
            'harsh_events_per_hour': 0,
            'stress_score': 0
        }
    
    print(flags.columns)

    # Match all actual flag_type values in the data:
    # harsh events  : 'harsh_braking', 'moderate_brake'
    # audio events  : 'audio_spike'
    # conflict/stress: 'conflict_moment', 'sustained_stress'
    harsh_events    = len(flags[flags['flag_type'].isin(['harsh_braking', 'moderate_brake'])])
    audio_spikes    = len(flags[flags['flag_type'] == 'audio_spike'])
    conflict_moments = len(flags[flags['flag_type'].isin(['conflict_moment', 'sustained_stress'])])

    total_stress_events = harsh_events + audio_spikes + conflict_moments

    # Get elapsed hours — fall back to trips duration if velocity log missing
    velocity_log = driver_data.get('velocity', pd.DataFrame())
    trips = driver_data.get('trips', pd.DataFrame())
    if not velocity_log.empty and 'elapsed_hours' in velocity_log.columns:
        elapsed_hours = velocity_log['elapsed_hours'].iloc[-1]
    elif not trips.empty and 'duration_min' in trips.columns:
        elapsed_hours = max(trips['duration_min'].sum() / 60, 0.1)
    else:
        elapsed_hours = 1

    harsh_events_per_hour = harsh_events / elapsed_hours if elapsed_hours > 0 else 0

    # Calculate stress score (0-100)
    # Thresholds calibrated to actual data: median driver has 0 flags,
    # 75th pct = 2 flags, 90th pct = 3 flags, max = 8 flags
    stress_score = 0

    # Harsh events per hour (max 35 points)
    if harsh_events_per_hour > 3:
        stress_score += 35
    elif harsh_events_per_hour > 1.5:
        stress_score += 25
    elif harsh_events_per_hour > 0.5:
        stress_score += 15

    # Audio spikes — absolute count (max 30 points)
    if audio_spikes >= 4:
        stress_score += 30
    elif audio_spikes >= 2:
        stress_score += 20
    elif audio_spikes >= 1:
        stress_score += 10

    # Conflict/sustained stress moments — highest weight (max 35 points)
    if conflict_moments >= 3:
        stress_score += 35
    elif conflict_moments >= 2:
        stress_score += 25
    elif conflict_moments >= 1:
        stress_score += 15

    return {
        'harsh_events': harsh_events,
        'audio_spikes': audio_spikes,
        'conflict_moments': conflict_moments,
        'total_stress_events': total_stress_events,
        'harsh_events_per_hour': harsh_events_per_hour,
        'stress_score': min(stress_score, 100)
    }


def calculate_earnings_pressure(driver_data: dict) -> dict:
    """
    Calculate earnings pressure from goal gap and performance.

    Args:
        driver_data: Dictionary from data_loader.get_driver_data()

    Returns:
        Dictionary with earnings pressure metrics
    """
    goals = driver_data.get('goals', pd.DataFrame())
    velocity_log = driver_data.get('velocity', pd.DataFrame())

    if goals.empty:
        return {
            'has_goal': False,
            'goal_gap': 0,
            'goal_gap_pct': 0,
            'velocity_declining': False,
            'behind_pace': False,
            'pressure_score': 0
        }

    goal = goals.iloc[-1]

    # Calculate goal gap
    target = goal['target_earnings']
    current = goal['current_earnings']
    goal_gap = target - current
    goal_gap_pct = (goal_gap / target * 100) if target > 0 else 0

    # Check if velocity is declining
    velocity_declining = False
    if not velocity_log.empty and len(velocity_log) >= 2:
        velocity_log_sorted = velocity_log.sort_values('timestamp')
        recent_velocity = velocity_log_sorted['current_velocity'].iloc[-1]
        earlier_velocity = velocity_log_sorted['current_velocity'].iloc[0]
        velocity_declining = recent_velocity < earlier_velocity * 0.85

    # Check if behind pace
    behind_pace = goal.get('goal_completion_forecast', '') == 'at_risk'

    # Calculate pressure score (0-100)
    # Calibrated to actual data: median gap_pct = 56%, 30% of drivers are at_risk forecast
    pressure_score = 0

    # Goal gap (max 40 points) — most drivers are 30-80% behind, so tighten thresholds
    if goal_gap_pct > 70:
        pressure_score += 40
    elif goal_gap_pct > 50:
        pressure_score += 30
    elif goal_gap_pct > 30:
        pressure_score += 20
    elif goal_gap_pct > 15:
        pressure_score += 10

    # Behind pace forecast (max 35 points)
    if behind_pace:
        pressure_score += 35

    # Velocity declining (max 25 points)
    if velocity_declining:
        pressure_score += 25

    return {
        'has_goal': True,
        'goal_gap': goal_gap,
        'goal_gap_pct': goal_gap_pct,
        'velocity_declining': velocity_declining,
        'behind_pace': behind_pace,
        'pressure_score': min(pressure_score, 100)
    }


def calculate_burnout_risk(driver_data: dict) -> dict:
    """
    Calculate overall burnout risk score combining all factors.

    Args:
        driver_data: Dictionary from data_loader.get_driver_data()

    Returns:
        Dictionary with burnout risk assessment
    """
    # Get component scores
    work_intensity = calculate_work_intensity(driver_data)
    stress_signals = calculate_stress_signals(driver_data)
    earnings_pressure = calculate_earnings_pressure(driver_data)

    # Weighted combination
    # Work intensity: 35%
    # Stress signals: 40%
    # Earnings pressure: 25%
    burnout_score = (
        work_intensity['intensity_score'] * 0.35 +
        stress_signals['stress_score'] * 0.40 +
        earnings_pressure['pressure_score'] * 0.25
    )

    # Categorize risk level
    # Thresholds are relative to actual data range (max observed ~47).
    # HIGH = top ~5% of drivers, MODERATE = next ~30%, LOW = rest.
    if burnout_score >= 40:
        risk_level = 'HIGH'
        risk_color = 'red'
    elif burnout_score >= 22:
        risk_level = 'MODERATE'
        risk_color = 'orange'
    else:
        risk_level = 'LOW'
        risk_color = 'green'

    # Identify primary risk factor
    risk_factors = {
        'Work Intensity': work_intensity['intensity_score'],
        'Stress Signals': stress_signals['stress_score'],
        'Earnings Pressure': earnings_pressure['pressure_score']
    }
    primary_factor = max(risk_factors, key=risk_factors.get)
    primary_factor_score = risk_factors[primary_factor]

    # Generate recommendations
    recommendations = generate_burnout_recommendations(
        work_intensity, stress_signals, earnings_pressure, burnout_score
    )

    return {
        'burnout_score': burnout_score,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'primary_factor': primary_factor,
        'primary_factor_score': primary_factor_score,

        # Component details
        'work_intensity': work_intensity,
        'stress_signals': stress_signals,
        'earnings_pressure': earnings_pressure,

        # Risk factor breakdown
        'risk_factors': risk_factors,

        # Recommendations
        'recommendations': recommendations
    }


def generate_burnout_recommendations(work_intensity: dict, stress_signals: dict,
                                    earnings_pressure: dict, burnout_score: float) -> list:
    """
    Generate personalized recommendations to reduce burnout risk.

    Args:
        work_intensity: Work intensity metrics
        stress_signals: Stress signal metrics
        earnings_pressure: Earnings pressure metrics
        burnout_score: Overall burnout score

    Returns:
        List of recommendation dictionaries
    """
    recommendations = []

    # Critical recommendations (high priority)
    if burnout_score >= 70:
        recommendations.append({
            'priority': 'critical',
            'icon': '🚨',
            'title': 'Take a Break Immediately',
            'message': 'Your burnout risk is HIGH. Consider ending your shift or taking a 30-minute break.',
            'action': 'End shift or take extended break'
        })

    # Work intensity recommendations
    if work_intensity['hours_driving'] > 8:
        recommendations.append({
            'priority': 'high',
            'icon': '⏰',
            'title': 'Long Shift Alert',
            'message': f"You've been driving for {work_intensity['hours_driving']:.1f} hours. Consider wrapping up soon.",
            'action': 'Plan to end shift within 1 hour'
        })

    if not work_intensity['has_taken_break'] and work_intensity['hours_driving'] > 3:
        recommendations.append({
            'priority': 'high',
            'icon': '☕',
            'title': 'Take a Break',
            'message': "You haven't taken a break yet. Take 15-20 minutes to rest and recharge.",
            'action': 'Take a 15-20 minute break'
        })

    if work_intensity['trips_per_hour'] > 5:
        recommendations.append({
            'priority': 'medium',
            'icon': '🐢',
            'title': 'Slow Down',
            'message': f"You're averaging {work_intensity['trips_per_hour']:.1f} trips/hour. This pace is not sustainable.",
            'action': 'Take longer breaks between trips'
        })

    # Stress signal recommendations
    if stress_signals['conflict_moments'] > 2:
        recommendations.append({
            'priority': 'high',
            'icon': '😰',
            'title': 'High Stress Detected',
            'message': f"{stress_signals['conflict_moments']} conflict moments detected. Take time to decompress.",
            'action': 'Take a break and practice deep breathing'
        })

    if stress_signals['harsh_events_per_hour'] > 4:
        recommendations.append({
            'priority': 'medium',
            'icon': '🚗',
            'title': 'Aggressive Driving Pattern',
            'message': f"{stress_signals['harsh_events_per_hour']:.1f} harsh events/hour. Drive more calmly.",
            'action': 'Focus on smooth acceleration and braking'
        })

    # Earnings pressure recommendations
    if earnings_pressure.get('behind_pace', False):
        recommendations.append({
            'priority': 'medium',
            'icon': '🎯',
            'title': 'Behind on Goal',
            'message': "You're behind pace on your goal. Consider adjusting expectations or strategy.",
            'action': 'Focus on high-value trips or extend shift slightly'
        })

    if earnings_pressure.get('velocity_declining', False):
        recommendations.append({
            'priority': 'medium',
            'icon': '📉',
            'title': 'Earnings Declining',
            'message': "Your earnings rate is declining. This might indicate fatigue.",
            'action': 'Take a break or consider ending shift'
        })

    # Positive reinforcement
    if burnout_score < 30:
        recommendations.append({
            'priority': 'low',
            'icon': '✨',
            'title': 'You\'re Doing Great!',
            'message': "Your burnout risk is low. Keep maintaining healthy work habits.",
            'action': 'Continue current approach'
        })

    # Sort by priority
    priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

    return recommendations


def build_burnout_profile(driver_data: dict) -> dict:
    """
    Build comprehensive burnout profile for a driver.

    Args:
        driver_data: Dictionary from data_loader.get_driver_data()

    Returns:
        Dictionary with complete burnout analytics
    """
    # Calculate burnout risk
    burnout_risk = calculate_burnout_risk(driver_data)

    # Add historical context (simplified for MVP)
    velocity_log = driver_data.get('velocity', pd.DataFrame())

    # Calculate historical burnout trend
    if not velocity_log.empty and len(velocity_log) > 1:
        # Simplified trend: compare current vs average stress indicators
        current_trips = velocity_log['trips_completed'].iloc[-1]
        avg_trips_rate = current_trips / len(velocity_log)

        if avg_trips_rate > 4:
            historical_trend = 'increasing'
        elif avg_trips_rate < 2:
            historical_trend = 'stable'
        else:
            historical_trend = 'moderate'
    else:
        historical_trend = 'stable'

    return {
        **burnout_risk,
        'historical_trend': historical_trend,
        'driver_id': driver_data.get('driver', {}).get('driver_id', 'Unknown')
    }