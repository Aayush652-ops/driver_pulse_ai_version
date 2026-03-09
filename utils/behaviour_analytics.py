"""
behaviour_analytics.py
======================
All computation logic for the Driving Behaviour page (Page 5).
Owned by: Saisha (Safety & Sensor Intelligence Lead)

Responsibilities:
  - Compute per-trip behavioural scores (smoothness, aggression, consistency)
  - Build shift-pattern heatmaps
  - Trend analysis across trips
  - Peer benchmarking (percentile vs fleet)
  - Coaching tip generation (rule-based)
  - Streak / badge logic (gamification)

Design principle:
  ALL computation happens here. The page file only calls these functions
  and renders results. Zero pandas logic in the page file.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
#  THRESHOLDS & CONSTANTS
# ─────────────────────────────────────────────────────────────

# Driving behaviour score weights (sum = 100)
WEIGHT_SMOOTHNESS   = 35   # how smooth acceleration/deceleration is
WEIGHT_SPEED        = 25   # speed discipline (avoiding overspeed)
WEIGHT_CABIN        = 20   # cabin audio environment
WEIGHT_CONSISTENCY  = 20   # trip-to-trip consistency

# Aggression scoring thresholds
HARSH_BRAKE_SCORE_PER_EVENT  = 15   # deduct per harsh brake event per trip
HARSH_ACCEL_SCORE_PER_EVENT  = 10
OVERSPEED_SCORE_PER_KM       = 2    # deduct per km over speed limit

# Star-rating thresholds for behaviour score
STAR_THRESHOLDS = [
    (90, 5),
    (75, 4),
    (60, 3),
    (45, 2),
    (0,  1),
]

# Badge definitions
BADGES = {
    "smooth_operator":   {"label": "Smooth Operator",   "icon": "🎯", "desc": "3+ trips with smoothness > 80"},
    "speed_guardian":    {"label": "Speed Guardian",     "icon": "🚦", "desc": "0 overspeed events in last 5 trips"},
    "calm_cabin":        {"label": "Calm Cabin",         "icon": "🧘", "desc": "No audio spikes across all trips"},
    "consistent_pro":    {"label": "Consistent Pro",     "icon": "📊", "desc": "Behaviour score variance < 10 across trips"},
    "early_bird":        {"label": "Early Bird",         "icon": "🌅", "desc": "5+ morning shift trips completed"},
    "night_owl":         {"label": "Night Owl",          "icon": "🌙", "desc": "5+ night shift trips completed"},
    "veteran":           {"label": "Veteran Driver",     "icon": "🏆", "desc": "12+ months on platform"},
    "top_rated":         {"label": "Top Rated",          "icon": "⭐", "desc": "Rating 4.9+"},
}

# Hour buckets for heatmap
SHIFT_BUCKETS = {
    "Early Morning (5–9)":  range(5, 9),
    "Morning (9–12)":       range(9, 12),
    "Afternoon (12–17)":    range(12, 17),
    "Evening (17–21)":      range(17, 21),
    "Night (21–5)":         list(range(21, 24)) + list(range(0, 5)),
}


# ─────────────────────────────────────────────────────────────
#  CORE: PER-TRIP BEHAVIOUR SCORE
# ─────────────────────────────────────────────────────────────

def compute_trip_behaviour_score(
    trip_row: pd.Series,
    motion_events: int,
    audio_events: int,
    stress_score: float,
    flagged_count: int,
    max_severity: str,
) -> Dict:
    """
    Compute a composite behaviour score for a single trip.

    Returns a dict with:
        total_score (0-100), grade (A-F), component scores,
        aggression_index, smoothness_index, cabin_index
    """
    duration_min = max(trip_row.get("duration_min", 1), 1)
    distance_km  = max(trip_row.get("distance_km", 1), 0.1)

    # ── Smoothness component (35 pts) ──
    # Base: start at 100, deduct for motion events per trip
    motion_rate     = motion_events / max(duration_min / 10, 1)  # events per 10 min
    smoothness_raw  = max(0, 100 - motion_rate * 20 - stress_score * 30)
    smoothness_score = min(35, (smoothness_raw / 100) * 35)

    # ── Speed discipline component (25 pts) ──
    # Use stress_score as proxy for speed discipline when no raw data
    severity_penalty = {"none": 0, "low": 5, "medium": 12, "high": 22}.get(
        str(max_severity).lower(), 0
    )
    speed_raw   = max(0, 100 - severity_penalty - stress_score * 15)
    speed_score = min(25, (speed_raw / 100) * 25)

    # ── Cabin environment component (20 pts) ──
    audio_rate    = audio_events / max(duration_min / 10, 1)
    cabin_raw     = max(0, 100 - audio_rate * 15 - stress_score * 20)
    cabin_score   = min(20, (cabin_raw / 100) * 20)

    # ── Consistency placeholder (20 pts) — filled in at fleet level ──
    # For per-trip we use flagged moment density as proxy
    flag_rate         = flagged_count / max(duration_min / 10, 1)
    consistency_raw   = max(0, 100 - flag_rate * 25)
    consistency_score = min(20, (consistency_raw / 100) * 20)

    total = round(smoothness_score + speed_score + cabin_score + consistency_score, 1)
    total = min(100.0, max(0.0, total))

    grade = _score_to_grade(total)
    stars = _score_to_stars(total)

    return {
        "total_score":        total,
        "grade":              grade,
        "stars":              stars,
        "smoothness_score":   round(smoothness_score, 1),
        "speed_score":        round(speed_score, 1),
        "cabin_score":        round(cabin_score, 1),
        "consistency_score":  round(consistency_score, 1),
        "aggression_index":   round(min(100, stress_score * 100), 1),
        "smoothness_index":   round(smoothness_raw, 1),
        "cabin_index":        round(cabin_raw, 1),
    }


def _score_to_grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B+"
    if score >= 60: return "B"
    if score >= 50: return "C"
    if score >= 40: return "D"
    return "F"


def _score_to_stars(score: float) -> int:
    for threshold, stars in STAR_THRESHOLDS:
        if score >= threshold:
            return stars
    return 1


# ─────────────────────────────────────────────────────────────
#  DRIVER BEHAVIOUR PROFILE
# ─────────────────────────────────────────────────────────────

def build_behaviour_profile(
    driver_id: str,
    driver_info: Dict,
    trips_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    all_drivers_df: pd.DataFrame,
    all_summaries_df: pd.DataFrame,
) -> Dict:
    """
    Master function. Builds the full behaviour profile for a driver.
    Handles all edge cases (missing data, empty frames, single trip).
    Returns a comprehensive dict consumed by the page renderer.
    """
    # ── Guard: no trips ──
    if trips_df.empty:
        return _empty_profile(driver_id, driver_info)

    # ── Merge trips with summaries ──
    merged = _merge_trip_summaries(trips_df, summaries_df)

    # ── Compute per-trip behaviour scores ──
    trip_scores = _compute_all_trip_scores(merged, flags_df)

    # ── Aggregate driver-level stats ──
    driver_agg = _aggregate_driver_stats(trip_scores, driver_info, trips_df)

    # ── Consistency score (needs all trips) ──
    driver_agg["consistency_score"] = _compute_consistency(trip_scores)

    # ── Final overall behaviour score ──
    driver_agg["overall_behaviour_score"] = _compute_overall_score(driver_agg)
    driver_agg["overall_grade"]           = _score_to_grade(driver_agg["overall_behaviour_score"])
    driver_agg["overall_stars"]           = _score_to_stars(driver_agg["overall_behaviour_score"])

    # ── Trend (improving / declining / stable) ──
    driver_agg["trend"]       = _compute_trend(trip_scores)
    driver_agg["trend_delta"] = _compute_trend_delta(trip_scores)

    # ── Shift pattern heatmap data ──
    driver_agg["shift_heatmap"] = _build_shift_heatmap(merged)

    # ── Peer benchmarking ──
    driver_agg["percentile"]   = _compute_percentile(driver_agg["overall_behaviour_score"], all_drivers_df, all_summaries_df)
    driver_agg["fleet_avg"]    = _compute_fleet_avg(all_summaries_df)

    # ── Badges ──
    driver_agg["badges"] = _compute_badges(driver_id, driver_info, trip_scores, merged)

    # ── Coaching tips ──
    driver_agg["coaching_tips"] = _generate_coaching_tips(driver_agg, trip_scores)

    # ── Trip-level frame for table/charts ──
    driver_agg["trip_scores_df"] = trip_scores

    # ── Route / location patterns ──
    driver_agg["location_patterns"] = _compute_location_patterns(merged)

    # ── Surge analysis ──
    driver_agg["surge_analysis"] = _compute_surge_analysis(merged)

    return driver_agg


def _empty_profile(driver_id: str, driver_info: Dict) -> Dict:
    """Safe empty profile for drivers with no trip data."""
    return {
        "driver_id":              driver_id,
        "driver_info":            driver_info,
        "has_data":               False,
        "total_trips":            0,
        "overall_behaviour_score": 0,
        "overall_grade":          "–",
        "overall_stars":          0,
        "trend":                  "stable",
        "trend_delta":            0.0,
        "trip_scores_df":         pd.DataFrame(),
        "badges":                 [],
        "coaching_tips":          [],
        "shift_heatmap":          {},
        "percentile":             0,
        "fleet_avg":              0,
        "location_patterns":      {},
        "surge_analysis":         {},
        "consistency_score":      50.0,
        "avg_smoothness":         50.0,
        "avg_speed_score":        50.0,
        "avg_cabin_score":        50.0,
        "avg_stress":             0.0,
        "total_distance_km":      0.0,
        "total_earnings":         0.0,
        "avg_fare":               0.0,
        "best_trip_score":        0.0,
        "worst_trip_score":       0.0,
        "excellent_trips":        0,
        "poor_trips":             0,
    }


def _merge_trip_summaries(trips_df: pd.DataFrame, summaries_df: pd.DataFrame) -> pd.DataFrame:
    """Merge trips with summaries, handling missing summary data gracefully."""
    if summaries_df.empty:
        # Create a minimal summaries frame from trips alone
        merged = trips_df.copy()
        merged["motion_events_count"]   = 0
        merged["audio_events_count"]    = 0
        merged["flagged_moments_count"] = 0
        merged["max_severity"]          = "none"
        merged["stress_score"]          = 0.0
        merged["trip_quality_rating"]   = "unknown"
        merged["earnings_velocity"]     = merged["fare"] / (merged["duration_min"] / 60).replace(0, np.nan)
        return merged

    # Merge on trip_id — left join keeps all trips even if no summary
    merged = trips_df.merge(
        summaries_df[[
            "trip_id", "motion_events_count", "audio_events_count",
            "flagged_moments_count", "max_severity", "stress_score",
            "trip_quality_rating", "earnings_velocity"
        ]],
        on="trip_id",
        how="left",
    )

    # Fill missing summary values safely
    merged["motion_events_count"]   = merged["motion_events_count"].fillna(0).astype(int)
    merged["audio_events_count"]    = merged["audio_events_count"].fillna(0).astype(int)
    merged["flagged_moments_count"] = merged["flagged_moments_count"].fillna(0).astype(int)
    merged["max_severity"]          = merged["max_severity"].fillna("none")
    merged["stress_score"]          = merged["stress_score"].fillna(0.0)
    merged["trip_quality_rating"]   = merged["trip_quality_rating"].fillna("unknown")
    merged["earnings_velocity"]     = merged["earnings_velocity"].fillna(
        merged["fare"] / (merged["duration_min"] / 60).replace(0, np.nan)
    )

    # Parse date and times for heatmap
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    if "start_time" in merged.columns:
        merged["hour"] = pd.to_datetime(
            merged["start_time"], format="%H:%M:%S", errors="coerce"
        ).dt.hour.fillna(0).astype(int)

    return merged


def _compute_all_trip_scores(merged: pd.DataFrame, flags_df: pd.DataFrame) -> pd.DataFrame:
    """Compute behaviour score for every trip row."""
    records = []
    for _, row in merged.iterrows():
        trip_id = row.get("trip_id", "")
        trip_flags = flags_df[flags_df["trip_id"] == trip_id] if not flags_df.empty else pd.DataFrame()

        score_dict = compute_trip_behaviour_score(
            trip_row      = row,
            motion_events = int(row.get("motion_events_count", 0)),
            audio_events  = int(row.get("audio_events_count", 0)),
            stress_score  = float(row.get("stress_score", 0)),
            flagged_count = int(row.get("flagged_moments_count", 0)),
            max_severity  = str(row.get("max_severity", "none")),
        )
        records.append({
            "trip_id":             trip_id,
            "date":                row.get("date", ""),
            "start_time":          row.get("start_time", ""),
            "duration_min":        row.get("duration_min", 0),
            "distance_km":         row.get("distance_km", 0),
            "fare":                row.get("fare", 0),
            "pickup_location":     row.get("pickup_location", ""),
            "dropoff_location":    row.get("dropoff_location", ""),
            "surge_multiplier":    row.get("surge_multiplier", 1.0),
            "trip_quality_rating": row.get("trip_quality_rating", "unknown"),
            "stress_score":        row.get("stress_score", 0),
            "max_severity":        row.get("max_severity", "none"),
            "motion_events":       int(row.get("motion_events_count", 0)),
            "audio_events":        int(row.get("audio_events_count", 0)),
            "flagged_count":       int(row.get("flagged_moments_count", 0)),
            "high_flags":          int((trip_flags["severity"] == "high").sum()) if not trip_flags.empty else 0,
            "hour":                row.get("hour", 0),
            **score_dict,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Sort by date + time for trend analysis
    df = df.sort_values(["date", "start_time"], na_position="last").reset_index(drop=True)
    df["trip_index"] = range(1, len(df) + 1)  # sequential for trend line

    return df


def _aggregate_driver_stats(
    trip_scores: pd.DataFrame,
    driver_info: Dict,
    trips_df: pd.DataFrame,
) -> Dict:
    """Aggregate trip-level scores to driver level."""
    if trip_scores.empty:
        return {}

    n = len(trip_scores)

    return {
        "has_data":          True,
        "driver_info":       driver_info,
        "total_trips":       n,
        "total_distance_km": round(trip_scores["distance_km"].sum(), 1),
        "total_earnings":    round(trip_scores["fare"].sum(), 2),
        "avg_fare":          round(trip_scores["fare"].mean(), 2),
        "avg_duration_min":  round(trip_scores["duration_min"].mean(), 1),
        "avg_distance_km":   round(trip_scores["distance_km"].mean(), 1),

        # Behaviour component averages
        "avg_smoothness":    round(trip_scores["smoothness_score"].mean() / 35 * 100, 1),
        "avg_speed_score":   round(trip_scores["speed_score"].mean() / 25 * 100, 1),
        "avg_cabin_score":   round(trip_scores["cabin_score"].mean() / 20 * 100, 1),
        "avg_stress":        round(trip_scores["stress_score"].mean() * 100, 1),

        # Best / worst
        "best_trip_score":   round(trip_scores["total_score"].max(), 1),
        "worst_trip_score":  round(trip_scores["total_score"].min(), 1),

        # Quality distribution
        "excellent_trips":   int((trip_scores["trip_quality_rating"] == "excellent").sum()),
        "good_trips":        int((trip_scores["trip_quality_rating"] == "good").sum()),
        "fair_trips":        int((trip_scores["trip_quality_rating"] == "fair").sum()),
        "poor_trips":        int((trip_scores["trip_quality_rating"] == "poor").sum()),

        # Severity distribution
        "high_severity_trips": int((trip_scores["max_severity"] == "high").sum()),
        "clean_trips":         int((trip_scores["max_severity"] == "none").sum()),

        # Flag counts
        "total_flagged":     int(trip_scores["flagged_count"].sum()),
        "total_high_flags":  int(trip_scores["high_flags"].sum()),

        # Totals from driver info
        "experience_months": driver_info.get("experience_months", 0),
        "rating":            driver_info.get("rating", 0),
        "city":              driver_info.get("city", ""),
        "shift_preference":  driver_info.get("shift_preference", ""),
    }


def _compute_consistency(trip_scores: pd.DataFrame) -> float:
    """
    Consistency = 100 - normalised std of behaviour scores.
    High consistency = low variance = good driver.
    """
    if trip_scores.empty or len(trip_scores) < 2:
        return 75.0  # neutral default for single-trip drivers

    std = trip_scores["total_score"].std()
    # std of 0 → 100 consistency; std of 20+ → ~0 consistency
    consistency = max(0, 100 - std * 3)
    return round(consistency, 1)


def _compute_overall_score(agg: Dict) -> float:
    """Weighted overall behaviour score from component averages."""
    w_smooth = WEIGHT_SMOOTHNESS / 100
    w_speed  = WEIGHT_SPEED / 100
    w_cabin  = WEIGHT_CABIN / 100
    w_cons   = WEIGHT_CONSISTENCY / 100

    score = (
        agg.get("avg_smoothness", 50)   * w_smooth +
        agg.get("avg_speed_score", 50)  * w_speed  +
        agg.get("avg_cabin_score", 50)  * w_cabin  +
        agg.get("consistency_score", 50)* w_cons
    )
    return round(min(100, max(0, score)), 1)


def _compute_trend(trip_scores: pd.DataFrame) -> str:
    """Detect if scores are improving, declining, or stable over last N trips."""
    if trip_scores.empty or len(trip_scores) < 3:
        return "stable"

    recent = trip_scores["total_score"].tail(min(5, len(trip_scores))).values
    if len(recent) < 2:
        return "stable"

    # Simple linear regression slope
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]

    if slope > 1.5:   return "improving"
    if slope < -1.5:  return "declining"
    return "stable"


def _compute_trend_delta(trip_scores: pd.DataFrame) -> float:
    """Difference between last 3 trips avg and previous 3 trips avg."""
    if trip_scores.empty or len(trip_scores) < 4:
        return 0.0
    scores = trip_scores["total_score"].values
    mid    = len(scores) // 2
    recent_avg = scores[mid:].mean()
    older_avg  = scores[:mid].mean()
    return round(recent_avg - older_avg, 1)


# ─────────────────────────────────────────────────────────────
#  SHIFT HEATMAP
# ─────────────────────────────────────────────────────────────

def _build_shift_heatmap(merged: pd.DataFrame) -> Dict:
    """
    Build hour-of-day trip activity and avg score heatmap.
    Returns dict with hours 0-23 as keys, values = {count, avg_stress, avg_fare}.
    """
    if "hour" not in merged.columns or merged.empty:
        return {}

    result = {}
    for hour in range(24):
        hour_trips = merged[merged["hour"] == hour]
        if hour_trips.empty:
            result[hour] = {"count": 0, "avg_stress": 0.0, "avg_fare": 0.0, "shift": _hour_to_shift(hour)}
        else:
            result[hour] = {
                "count":      len(hour_trips),
                "avg_stress": round(hour_trips["stress_score"].mean() * 100, 1),
                "avg_fare":   round(hour_trips["fare"].mean(), 2),
                "shift":      _hour_to_shift(hour),
            }
    return result


def _hour_to_shift(hour: int) -> str:
    if 5 <= hour < 9:   return "Early Morning"
    if 9 <= hour < 12:  return "Morning"
    if 12 <= hour < 17: return "Afternoon"
    if 17 <= hour < 21: return "Evening"
    return "Night"


# ─────────────────────────────────────────────────────────────
#  PEER BENCHMARKING
# ─────────────────────────────────────────────────────────────

def _compute_percentile(
    driver_score: float,
    all_drivers_df: pd.DataFrame,
    all_summaries_df: pd.DataFrame,
) -> int:
    """
    Compute driver's behaviour score percentile vs fleet.
    Uses stress_score inversion as behaviour proxy for all drivers.
    """
    if all_summaries_df.empty or all_drivers_df.empty:
        return 50

    # Fleet behaviour proxy: lower stress = higher behaviour score
    fleet_stress = all_summaries_df.groupby("driver_id")["stress_score"].mean()
    # Convert stress to behaviour proxy score (invert)
    fleet_scores = (1 - fleet_stress.clip(0, 1)) * 100

    if fleet_scores.empty:
        return 50

    # Percentile of THIS driver vs fleet
    pct = int((fleet_scores < driver_score).mean() * 100)
    return max(1, min(99, pct))


def _compute_fleet_avg(all_summaries_df: pd.DataFrame) -> float:
    """Fleet average behaviour proxy score."""
    if all_summaries_df.empty:
        return 70.0
    avg_stress = all_summaries_df["stress_score"].mean()
    return round((1 - avg_stress) * 100, 1)


# ─────────────────────────────────────────────────────────────
#  BADGES (GAMIFICATION)
# ─────────────────────────────────────────────────────────────

def _compute_badges(
    driver_id: str,
    driver_info: Dict,
    trip_scores: pd.DataFrame,
    merged: pd.DataFrame,
) -> List[Dict]:
    """Award badges based on behaviour criteria."""
    earned = []

    if trip_scores.empty:
        return earned

    # Smooth Operator: 3+ trips with smoothness index > 80
    high_smooth = (trip_scores["smoothness_index"] > 80).sum()
    if high_smooth >= 3:
        earned.append({**BADGES["smooth_operator"], "earned": True,
                       "detail": f"{high_smooth} trips with high smoothness"})

    # Speed Guardian: 0 high-severity trips in last 5
    last5 = trip_scores.tail(5)
    if (last5["max_severity"] == "high").sum() == 0 and len(last5) >= 3:
        earned.append({**BADGES["speed_guardian"], "earned": True,
                       "detail": "No high-severity events in recent trips"})

    # Calm Cabin: no audio events across trips
    if trip_scores["audio_events"].sum() == 0 and len(trip_scores) >= 2:
        earned.append({**BADGES["calm_cabin"], "earned": True,
                       "detail": "Zero audio events across all trips"})

    # Consistent Pro: score variance < 10
    if len(trip_scores) >= 3:
        variance = trip_scores["total_score"].std()
        if variance < 10:
            earned.append({**BADGES["consistent_pro"], "earned": True,
                           "detail": f"Score variance: {variance:.1f} pts"})

    # Early Bird: 3+ morning trips
    if "hour" in trip_scores.columns:
        morning = ((trip_scores["hour"] >= 5) & (trip_scores["hour"] < 9)).sum()
        if morning >= 3:
            earned.append({**BADGES["early_bird"], "earned": True,
                           "detail": f"{morning} early morning trips"})

        # Night Owl: 3+ night trips
        night = ((trip_scores["hour"] >= 21) | (trip_scores["hour"] < 5)).sum()
        if night >= 3:
            earned.append({**BADGES["night_owl"], "earned": True,
                           "detail": f"{night} night trips"})

    # Veteran: 12+ months
    exp = driver_info.get("experience_months", 0)
    if exp >= 12:
        earned.append({**BADGES["veteran"], "earned": True,
                       "detail": f"{exp} months on platform"})

    # Top Rated: 4.9+
    rating = driver_info.get("rating", 0)
    if rating >= 4.9:
        earned.append({**BADGES["top_rated"], "earned": True,
                       "detail": f"Rating: {rating}"})

    return earned


# ─────────────────────────────────────────────────────────────
#  COACHING TIPS
# ─────────────────────────────────────────────────────────────

def _generate_coaching_tips(agg: Dict, trip_scores: pd.DataFrame) -> List[Dict]:
    """
    Generate personalised, actionable coaching tips.
    Priority-sorted: high-impact issues first.
    """
    tips = []

    if trip_scores.empty:
        return tips

    avg_stress    = agg.get("avg_stress", 0)
    avg_smooth    = agg.get("avg_smoothness", 100)
    avg_speed     = agg.get("avg_speed_score", 100)
    avg_cabin     = agg.get("avg_cabin_score", 100)
    consistency   = agg.get("consistency_score", 100)
    high_trips    = agg.get("high_severity_trips", 0)
    total_trips   = max(agg.get("total_trips", 1), 1)
    trend         = agg.get("trend", "stable")

    # ── High-priority tips ──
    if high_trips / total_trips > 0.3:
        tips.append({
            "priority": "high",
            "icon": "⚠️",
            "title": "Reduce High-Severity Events",
            "body": f"{high_trips} of your {total_trips} trips had high-severity flags. "
                    "Focus on anticipating stops — increase following distance and "
                    "brake 2-3 seconds earlier than feels natural.",
            "metric": f"{round(high_trips/total_trips*100)}% of trips",
        })

    if avg_smooth < 55:
        tips.append({
            "priority": "high",
            "icon": "🎯",
            "title": "Smoother Acceleration & Braking",
            "body": "Your smoothness score is below average. Passengers rate smooth "
                    "rides 23% higher. Try the '3-second rule': spend 3 full seconds "
                    "accelerating from 0 to cruising speed, and 3 seconds braking to stop.",
            "metric": f"Smoothness: {avg_smooth:.0f}/100",
        })

    if avg_cabin < 55:
        tips.append({
            "priority": "high",
            "icon": "🔊",
            "title": "Manage Cabin Environment",
            "body": "Elevated audio stress detected in multiple trips. Play soft "
                    "background music to set a calm tone, and avoid engaging in "
                    "stressful conversations while driving.",
            "metric": f"Cabin score: {avg_cabin:.0f}/100",
        })

    # ── Medium-priority tips ──
    if avg_speed < 65:
        tips.append({
            "priority": "medium",
            "icon": "🚦",
            "title": "Speed Discipline",
            "body": "Speed events were detected across multiple trips. Maintaining "
                    "legal speeds reduces your insurance risk and keeps passengers calm. "
                    "Use cruise control on highways when safe.",
            "metric": f"Speed score: {avg_speed:.0f}/100",
        })

    if consistency < 60 and total_trips >= 3:
        tips.append({
            "priority": "medium",
            "icon": "📊",
            "title": "Build Trip Consistency",
            "body": "Your scores vary significantly trip-to-trip. Consistent drivers "
                    "earn more repeat ratings. Try to maintain the same pre-trip "
                    "routine: vehicle check, route preview, mirror adjustment.",
            "metric": f"Consistency: {consistency:.0f}/100",
        })

    if trend == "declining":
        tips.append({
            "priority": "medium",
            "icon": "📉",
            "title": "Recent Performance Dip",
            "body": "Your behaviour scores have dipped in recent trips. This often "
                    "happens with fatigue. Consider taking a 15-minute break between "
                    "back-to-back trips, and stay hydrated.",
            "metric": f"Trend: {agg.get('trend_delta', 0):+.1f} pts",
        })

    # ── Positive reinforcement ──
    if avg_smooth >= 80:
        tips.append({
            "priority": "positive",
            "icon": "✅",
            "title": "Excellent Smoothness — Keep It Up!",
            "body": "Your smooth driving style sets you apart. Passengers notice and "
                    "appreciate this. This directly contributes to higher ratings and "
                    "more repeat ride requests.",
            "metric": f"Smoothness: {avg_smooth:.0f}/100",
        })

    if trend == "improving":
        tips.append({
            "priority": "positive",
            "icon": "📈",
            "title": "Your Scores Are Improving!",
            "body": "Great momentum — your recent trips score higher than your older ones. "
                    "Whatever you changed, keep doing it. You're on track for a better rating.",
            "metric": f"Trend: +{abs(agg.get('trend_delta', 0)):.1f} pts",
        })

    # Limit to 4 most relevant tips
    high   = [t for t in tips if t["priority"] == "high"]
    medium = [t for t in tips if t["priority"] == "medium"]
    pos    = [t for t in tips if t["priority"] == "positive"]

    return (high + medium + pos)[:5]


# ─────────────────────────────────────────────────────────────
#  LOCATION & ROUTE PATTERNS
# ─────────────────────────────────────────────────────────────

def _compute_location_patterns(merged: pd.DataFrame) -> Dict:
    """Analyse pickup/dropoff location patterns."""
    if merged.empty:
        return {}

    pickup_counts  = merged["pickup_location"].value_counts().head(5).to_dict() if "pickup_location" in merged.columns else {}
    dropoff_counts = merged["dropoff_location"].value_counts().head(5).to_dict() if "dropoff_location" in merged.columns else {}

    # Most common routes
    if "pickup_location" in merged.columns and "dropoff_location" in merged.columns:
        merged["route"] = merged["pickup_location"] + " → " + merged["dropoff_location"]
        route_counts = merged["route"].value_counts().head(5).to_dict()
    else:
        route_counts = {}

    return {
        "top_pickups":  pickup_counts,
        "top_dropoffs": dropoff_counts,
        "top_routes":   route_counts,
        "unique_zones": len(set(list(pickup_counts.keys()) + list(dropoff_counts.keys()))),
    }


# ─────────────────────────────────────────────────────────────
#  SURGE ANALYSIS
# ─────────────────────────────────────────────────────────────

def _compute_surge_analysis(merged: pd.DataFrame) -> Dict:
    """Analyse surge multiplier patterns."""
    if merged.empty or "surge_multiplier" not in merged.columns:
        return {}

    surge_dist = merged["surge_multiplier"].value_counts().sort_index().to_dict()
    avg_surge  = round(merged["surge_multiplier"].mean(), 2)
    max_surge  = round(merged["surge_multiplier"].max(), 2)
    surge_trips= int((merged["surge_multiplier"] > 1.0).sum())

    return {
        "avg_surge":    avg_surge,
        "max_surge":    max_surge,
        "surge_trips":  surge_trips,
        "surge_pct":    round(surge_trips / max(len(merged), 1) * 100, 1),
        "surge_dist":   surge_dist,
    }


# ─────────────────────────────────────────────────────────────
#  FLEET BENCHMARKING HELPER (for demo mode dropdown)
# ─────────────────────────────────────────────────────────────

def get_all_driver_summary_scores(
    all_drivers_df: pd.DataFrame,
    all_summaries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a quick per-driver behaviour summary for the fleet leaderboard.
    Used by demo mode to populate the driver picker with meaningful data.
    """
    if all_summaries_df.empty:
        return pd.DataFrame()

    agg = all_summaries_df.groupby("driver_id").agg(
        trips          = ("trip_id", "count"),
        avg_stress     = ("stress_score", "mean"),
        avg_fare       = ("fare", "mean"),
        total_fare     = ("fare", "sum"),
        avg_dist       = ("distance_km", "mean"),
        high_sev_trips = ("max_severity", lambda x: (x == "high").sum()),
        excellent_trips= ("trip_quality_rating", lambda x: (x == "excellent").sum()),
    ).reset_index()

    # Behaviour score proxy
    agg["behaviour_score"] = ((1 - agg["avg_stress"]) * 100).clip(0, 100).round(1)

    # Merge with driver info
    merged = agg.merge(
        all_drivers_df[["driver_id", "name", "city", "rating", "experience_months", "shift_preference"]],
        on="driver_id", how="left"
    )

    return merged.sort_values("behaviour_score", ascending=False).reset_index(drop=True)