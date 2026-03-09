"""
earnings_analytics.py
--------------------
Earnings velocity, goal tracking, and earnings forecasting logic.
Designed to transform static earnings data into actionable insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_earnings_velocity(velocity_log: pd.DataFrame, window_trips: int = 3) -> pd.DataFrame:
    """
    Calculate rolling earnings velocity from velocity log.

    Args:
        velocity_log: DataFrame with cumulative_earnings, elapsed_hours, timestamp
        window_trips: Number of trips to use for rolling window calculation

    Returns:
        DataFrame with additional velocity metrics
    """
    if velocity_log.empty:
        return velocity_log

    df = velocity_log.copy().sort_values('timestamp')

    # Calculate instantaneous velocity (₹/hour for each log entry)
    df['instantaneous_velocity'] = df['cumulative_earnings'] / df['elapsed_hours'].replace(0, np.nan)

    # Calculate rolling velocity (average over last N trips)
    if len(df) >= window_trips:
        df['rolling_velocity'] = df['cumulative_earnings'].diff(window_trips) / \
                                 df['elapsed_hours'].diff(window_trips).replace(0, np.nan)
    else:
        df['rolling_velocity'] = df['instantaneous_velocity']

    # Fill NaN values in rolling_velocity with instantaneous_velocity
    df['rolling_velocity'] = df['rolling_velocity'].fillna(df['instantaneous_velocity'])

    return df


def get_goal_progress(driver_goals: pd.DataFrame) -> dict:
    """
    Get current goal progress for a driver.

    Args:
        driver_goals: DataFrame with goal information

    Returns:
        Dictionary with goal progress metrics
    """
    if driver_goals.empty:
        return {
            'has_goal': False,
            'target_earnings': 0,
            'current_earnings': 0,
            'progress_pct': 0,
            'status': 'no_goal',
            'earnings_velocity': 0,
            'target_velocity': 0,
            'forecast': 'unknown'
        }

    # Get the most recent goal (should only be one active per day)
    goal = driver_goals.iloc[-1]

    progress_pct = (goal['current_earnings'] / goal['target_earnings'] * 100) if goal['target_earnings'] > 0 else 0

    return {
        'has_goal': True,
        'goal_id': goal.get('goal_id', ''),
        'target_earnings': goal['target_earnings'],
        'target_hours': goal.get('target_hours', 8),
        'current_earnings': goal['current_earnings'],
        'current_hours': goal.get('current_hours', 0),
        'remaining_earnings': goal['target_earnings'] - goal['current_earnings'],
        'progress_pct': progress_pct,
        'status': goal.get('status', 'unknown'),
        'earnings_velocity': goal.get('earnings_velocity', 0),
        'target_velocity': goal['target_earnings'] / goal.get('target_hours', 8) if goal.get('target_hours', 8) > 0 else 0,
        'forecast': goal.get('goal_completion_forecast', 'unknown'),
        'shift_start': goal.get('shift_start_time', ''),
        'shift_end': goal.get('shift_end_time', ''),
    }


def predict_goal_achievement(goal_progress: dict, velocity_log: pd.DataFrame,
                             current_time: datetime = None) -> dict:
    """
    Predict whether driver will achieve their goal.

    Args:
        goal_progress: Dictionary from get_goal_progress()
        velocity_log: DataFrame with velocity data
        current_time: Current time (defaults to now)

    Returns:
        Dictionary with prediction results
    """
    if not goal_progress['has_goal']:
        return {
            'will_achieve': False,
            'confidence': 0,
            'prediction': 'NO_GOAL',
            'projected_final_earnings': 0,
            'required_velocity': 0,
            'target_velocity': 0,
            'current_velocity': 0,
            'velocity_gap': 0,
            'hours_remaining': 0,
            'earnings_remaining': 0,
            'recommendation': 'Set a daily goal to track your progress'
        }

    # Already achieved — trust 'achieved' status only if earnings actually cover the target
    if goal_progress['status'] == 'achieved' and goal_progress['current_earnings'] >= goal_progress['target_earnings']:
        return {
            'will_achieve': True,
            'confidence': 1.0,
            'prediction': 'ACHIEVED',
            'projected_final_earnings': goal_progress['current_earnings'],
            'required_velocity': 0,
            'target_velocity': goal_progress['target_velocity'],
            'current_velocity': goal_progress['earnings_velocity'],
            'velocity_gap': 0,
            'hours_remaining': 0,
            'earnings_remaining': 0,
            'recommendation': 'Goal achieved! Great work!'
        }

    current_velocity = goal_progress['earnings_velocity']
    target_velocity = goal_progress['target_velocity']
    current_earnings = goal_progress['current_earnings']
    target_earnings = goal_progress['target_earnings']
    current_hours = goal_progress['current_hours']
    target_hours = goal_progress['target_hours']

    # Calculate remaining time
    remaining_hours = target_hours - current_hours
    remaining_earnings = target_earnings - current_earnings

    # Calculate required velocity
    required_velocity = remaining_earnings / remaining_hours if remaining_hours > 0 else 0

    # Project final earnings based on current velocity
    projected_final_earnings = current_earnings + (current_velocity * remaining_hours)

    # Use the CSV's pre-computed forecast as the primary signal — it was calculated
    # with full trip history. Only fall back to velocity math if forecast is missing.
    csv_forecast = goal_progress.get('forecast', '')  # 'ahead', 'on_track', 'at_risk'

    if remaining_hours <= 0:
        will_achieve = current_earnings >= target_earnings
        confidence = 1.0
        prediction = 'ACHIEVED' if will_achieve else 'MISSED'
        recommendation = 'Shift complete' if will_achieve else 'Goal not reached this shift'

    elif csv_forecast == 'ahead':
        will_achieve = True
        confidence = 0.92
        prediction = 'AHEAD'
        pct_ahead = ((current_velocity - required_velocity) / required_velocity * 100) if required_velocity > 0 else 50
        recommendation = f"You're ahead of pace! Keep it up!"

    elif csv_forecast == 'on_track':
        will_achieve = True
        confidence = 0.80
        prediction = 'ON_TRACK'
        recommendation = "You're on track to meet your goal. Stay consistent!"

    elif csv_forecast == 'at_risk':
        # Distinguish AT_RISK vs UNLIKELY based on how far behind
        velocity_ratio = current_velocity / required_velocity if required_velocity > 0 else 0
        if velocity_ratio >= 0.75:
            will_achieve = False
            confidence = 0.40
            prediction = 'AT_RISK'
            gap = required_velocity - current_velocity
            recommendation = f"Pick up pace by ₹{gap:.0f}/hr to reach your goal"
        else:
            will_achieve = False
            confidence = 0.15
            prediction = 'UNLIKELY'
            recommendation = "Consider adjusting your goal or extending your shift"

    else:
        # No CSV forecast — fall back to velocity math
        if current_velocity >= required_velocity:
            velocity_ratio = current_velocity / required_velocity if required_velocity > 0 else 2.0
            will_achieve = True
            confidence = 0.85 if velocity_ratio >= 1.2 else 0.70
            prediction = 'ON_TRACK'
            recommendation = "You're on track to meet your goal. Stay consistent!"
        else:
            will_achieve = False
            confidence = 0.30
            prediction = 'AT_RISK'
            gap = required_velocity - current_velocity
            recommendation = f"Pick up pace by ₹{gap:.0f}/hr to reach your goal"

    return {
        'will_achieve': will_achieve,
        'confidence': confidence,
        'prediction': prediction,
        'projected_final_earnings': projected_final_earnings,
        'required_velocity': required_velocity,
        'target_velocity': target_velocity,
        'current_velocity': current_velocity,
        'velocity_gap': current_velocity - required_velocity,
        'hours_remaining': remaining_hours,
        'earnings_remaining': remaining_earnings,
        'recommendation': recommendation
    }


def analyze_earnings_by_hour(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze earnings patterns by hour of day.

    Args:
        trips: DataFrame with trip data including fare and date columns

    Returns:
        DataFrame with hourly earnings statistics
    """
    if trips.empty or 'fare' not in trips.columns or 'date' not in trips.columns:
        return pd.DataFrame()

    df = trips.copy()

    # Extract hour from timestamp
    df['hour'] = pd.to_datetime(df['date']).dt.hour

    # Group by hour
    hourly = df.groupby('hour').agg({
        'fare': ['sum', 'mean', 'count'],
        'trip_id': 'count'
    }).reset_index()

    hourly.columns = ['hour', 'total_earnings', 'avg_fare', 'fare_count', 'trip_count']

    # Calculate earnings per trip
    hourly['earnings_per_trip'] = hourly['total_earnings'] / hourly['trip_count']

    return hourly


def get_best_earning_hours(trips: pd.DataFrame, top_n: int = 3) -> list:
    """
    Identify the best earning hours for the driver.

    Args:
        trips: DataFrame with trip data
        top_n: Number of top hours to return

    Returns:
        List of tuples (hour, total_earnings, avg_fare)
    """
    hourly = analyze_earnings_by_hour(trips)

    if hourly.empty:
        return []

    # Sort by total earnings
    top_hours = hourly.nlargest(top_n, 'total_earnings')

    result = []
    for _, row in top_hours.iterrows():
        result.append({
            'hour': int(row['hour']),
            'hour_range': f"{int(row['hour']):02d}:00 - {int(row['hour'])+1:02d}:00",
            'total_earnings': row['total_earnings'],
            'avg_fare': row['avg_fare'],
            'trip_count': int(row['trip_count']),
            'earnings_per_trip': row['earnings_per_trip']
        })

    return result


def calculate_earnings_trend(velocity_log: pd.DataFrame, window: int = 5) -> dict:
    """
    Calculate earnings trend (improving/declining/stable).

    Args:
        velocity_log: DataFrame with velocity data
        window: Window size for trend calculation

    Returns:
        Dictionary with trend information
    """
    if velocity_log.empty or len(velocity_log) < 2:
        return {
            'trend': 'stable',
            'trend_direction': 0,
            'velocity_change_pct': 0
        }

    df = velocity_log.copy().sort_values('timestamp')

    # Calculate trend using current_velocity
    if len(df) >= window:
        recent_velocity = df['current_velocity'].tail(window).mean()
        older_velocity = df['current_velocity'].head(window).mean()
    else:
        recent_velocity = df['current_velocity'].iloc[-1]
        older_velocity = df['current_velocity'].iloc[0]

    velocity_change_pct = ((recent_velocity - older_velocity) / older_velocity * 100) if older_velocity > 0 else 0

    # Categorize trend
    if velocity_change_pct > 10:
        trend = 'improving'
        trend_direction = 1
    elif velocity_change_pct < -10:
        trend = 'declining'
        trend_direction = -1
    else:
        trend = 'stable'
        trend_direction = 0

    return {
        'trend': trend,
        'trend_direction': trend_direction,
        'velocity_change_pct': velocity_change_pct,
        'recent_velocity': recent_velocity,
        'older_velocity': older_velocity
    }


def build_earnings_profile(driver_data: dict) -> dict:
    """
    Build comprehensive earnings profile for a driver.

    Args:
        driver_data: Dictionary from data_loader.get_driver_data()

    Returns:
        Dictionary with complete earnings analytics
    """
    velocity_log = driver_data.get('velocity', pd.DataFrame())
    goals = driver_data.get('goals', pd.DataFrame())
    trips = driver_data.get('trips', pd.DataFrame())

    # Calculate velocity metrics
    velocity_enriched = calculate_earnings_velocity(velocity_log)

    # Get goal progress (raw from CSV)
    goal_progress = get_goal_progress(goals)

    # Analyze hourly patterns and trend
    best_hours = get_best_earning_hours(trips)
    trend = calculate_earnings_trend(velocity_enriched)

    # ── Source of truth hierarchy ─────────────────────────────────────
    # driver_goals.csv has accumulated current_earnings across ALL trips (authoritative)
    # trips.csv may be incomplete — use only for trip list display
    # velocity_log has time-based metrics (velocity, elapsed_hours)

    total_trips = len(trips)
    avg_fare = trips['fare'].mean() if not trips.empty and 'fare' in trips.columns else 0

    # Use goals CSV current_earnings as the authoritative earnings figure
    if goal_progress['has_goal'] and goal_progress['current_earnings'] > 0:
        current_earnings = goal_progress['current_earnings']
    else:
        current_earnings = trips['fare'].sum() if not trips.empty and 'fare' in trips.columns else 0

    total_earnings = current_earnings

    # trips_completed: prefer velocity log, fall back to trips table count
    trips_completed = int(velocity_enriched['trips_completed'].iloc[-1]) if not velocity_enriched.empty else total_trips

    # Recalculate goal progress fields using authoritative current_earnings
    if goal_progress['has_goal'] and goal_progress['target_earnings'] > 0:
        goal_progress['current_earnings'] = current_earnings
        goal_progress['remaining_earnings'] = goal_progress['target_earnings'] - current_earnings
        goal_progress['progress_pct'] = current_earnings / goal_progress['target_earnings'] * 100

    # Velocity and elapsed hours from velocity log; fall back to goals CSV
    current_velocity = velocity_enriched['current_velocity'].iloc[-1] if not velocity_enriched.empty else (
        goal_progress.get('earnings_velocity', 0)
    )
    elapsed_hours = velocity_enriched['elapsed_hours'].iloc[-1] if not velocity_enriched.empty else (
        goal_progress.get('current_hours', 0)
    )

    # Predict AFTER goal_progress has been corrected with authoritative earnings
    goal_prediction = predict_goal_achievement(goal_progress, velocity_enriched)

    return {
        # Goal tracking
        'goal': goal_progress,
        'prediction': goal_prediction,

        # Current session
        'current_earnings': current_earnings,
        'current_velocity': current_velocity,
        'elapsed_hours': elapsed_hours,
        'trips_completed': trips_completed,

        # Overall stats
        'total_earnings': total_earnings,
        'total_trips': total_trips,
        'avg_fare': avg_fare,

        # Patterns
        'best_hours': best_hours,
        'trend': trend,

        # Raw data for charts
        'velocity_log': velocity_enriched,
        'trips': trips
    }