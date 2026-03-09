"""
Analytical helper functions for DriverPulse AI.

These functions perform simple computations on driver datasets. They
serve as a baseline implementation for safety scoring, burnout
probability estimation and goal tracking. More sophisticated models
produced by other team members can replace or override these
functions without changing the calling code in the dashboard page.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_safety_score(
    flagged_df: pd.DataFrame,
    accel_df: pd.DataFrame,
    audio_df: pd.DataFrame,
) -> Tuple[float, str]:
    """Compute a rudimentary safety score and category.

    This implementation is intentionally simple and serves as a
    placeholder for Person B's sensor‑based ML model. It combines the
    number of flagged events, variance in accelerometer readings, and
    the number of high audio intensity periods into a single score.

    Parameters
    ----------
    flagged_df : pandas.DataFrame
        Rows from `flagged_moments.csv` for a given driver.
    accel_df : pandas.DataFrame
        Rows from `accelerometer_data.csv` for a given driver.
    audio_df : pandas.DataFrame
        Rows from `audio_intensity_data.csv` for a given driver.

    Returns
    -------
    Tuple[float, str]
        A score between 0 and 100 and a risk category string
        ("Low", "Medium", or "High").
    """
    # Count flagged events
    event_count = len(flagged_df)

    # Compute average variance of acceleration magnitude
    accel_variance = 0.0
    if not accel_df.empty and set({"accel_x", "accel_y", "accel_z"}).issubset(accel_df.columns):
        accel_df = accel_df[["accel_x", "accel_y", "accel_z"]].astype(float)
        accel_magnitude = np.linalg.norm(accel_df.values, axis=1)
        accel_variance = float(np.var(accel_magnitude))

    # Count high audio intensity periods (above 0.7 threshold)
    audio_count = 0
    if not audio_df.empty and "intensity" in audio_df.columns:
        audio_count = int((audio_df["intensity"].astype(float) > 0.7).sum())

    # Combine metrics into a score (weights chosen arbitrarily)
    raw_score = event_count * 2 + accel_variance * 50 + audio_count * 1
    score = max(0.0, min(100.0, raw_score))

    if score < 30:
        category = "Low"
    elif score < 60:
        category = "Medium"
    else:
        category = "High"
    return score, category


def calculate_burnout_probability(
    earnings_log: pd.DataFrame,
    trips: pd.DataFrame,
) -> float:
    """Estimate a simple burnout probability for a driver.

    This placeholder uses the trend of earnings velocity and the total
    number of trips to produce a percentage. A decreasing earnings
    velocity combined with a high number of trips yields a higher
    burnout probability.

    Parameters
    ----------
    earnings_log : pandas.DataFrame
        Rows from `earnings_velocity_log.csv` for the driver.
    trips : pandas.DataFrame
        Rows from `trip_summaries.csv` for the driver.

    Returns
    -------
    float
        Burnout probability as a percentage between 0 and 100.
    """
    if earnings_log.empty:
        return 0.0
    # Compute earnings velocity slope (difference between last and first)
    earnings_values = earnings_log["earnings_velocity"].astype(float).values
    slope = earnings_values[-1] - earnings_values[0]
    # More negative slope => higher burnout risk
    slope_factor = -slope / max(abs(slope), 1)  # -1 to 1
    # Normalize trip count (more trips -> more stress)
    trip_factor = len(trips) / max(len(trips), 50)
    # Combine and scale
    burnout_raw = (slope_factor + trip_factor) * 50 + 20
    burnout = max(0.0, min(100.0, burnout_raw))
    return float(burnout)


def calculate_goal_progress(
    trips: pd.DataFrame,
    goals: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate trip and earnings goal completion percentages.

    Given a driver's trip summaries and weekly goal targets, compute
    what fraction of the trip and earnings goals have been achieved
    so far. If no goals are specified, returns (0, 0).

    Parameters
    ----------
    trips : pandas.DataFrame
        Rows from `trip_summaries.csv` for the driver.
    goals : pandas.DataFrame
        Rows from `driver_goals.csv` for the driver.

    Returns
    -------
    Tuple[float, float]
        (trip_completion_percent, earnings_completion_percent)
    """
    if goals.empty:
        return 0.0, 0.0
    target_trips = float(goals.iloc[0].get("weekly_target_trips", 0))
    target_earnings = float(goals.iloc[0].get("weekly_target_earnings", 0))
    completed_trips = float(len(trips))
    # Compute total earnings from trips. Some datasets store this under
    # 'earnings'; others may use 'fare'. Fall back accordingly.
    if not trips.empty:
        if "earnings" in trips.columns:
            total_earnings = float(trips["earnings"].astype(float).sum())
        elif "fare" in trips.columns:
            total_earnings = float(trips["fare"].astype(float).sum())
        else:
            total_earnings = 0.0
    else:
        total_earnings = 0.0
    trip_pct = (completed_trips / target_trips * 100.0) if target_trips > 0 else 0.0
    earn_pct = (total_earnings / target_earnings * 100.0) if target_earnings > 0 else 0.0
    trip_pct = min(100.0, trip_pct)
    earn_pct = min(100.0, earn_pct)
    return trip_pct, earn_pct