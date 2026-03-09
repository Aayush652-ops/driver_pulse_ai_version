"""
feature_engineering.py
-----------------------
Computes all safety-relevant features from raw sensor data.
Owned by: Saisha (Safety & Sensor Intelligence Lead)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


# ─────────────────────────────────────────────
#  THRESHOLDS  (tune here, not scattered around)
# ─────────────────────────────────────────────
ACCEL_MAGNITUDE_HARSH   = 8.5   # g-force combined magnitude for harsh event
JERK_THRESHOLD          = 3.0   # m/s³ — sudden change in acceleration
SPEED_OVERSPEED_KMH     = 55    # km/h — over-speed flag threshold
AUDIO_LOUD_DB           = 68    # dB  — "loud" cabin threshold
AUDIO_VERY_LOUD_DB      = 80    # dB  — "very loud" / argument threshold
AUDIO_SPIKE_STD_FACTOR  = 1.8   # flag if audio > mean + 1.8*std of the trip
WINDOW_SECONDS          = 10    # rolling window size in seconds


# ─────────────────────────────────────────────
#  ACCELEROMETER FEATURES
# ─────────────────────────────────────────────

def compute_accel_magnitude(df: pd.DataFrame) -> pd.Series:
    """Euclidean magnitude of 3-axis acceleration."""
    return np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)


def compute_jerk(df: pd.DataFrame) -> pd.Series:
    """
    Jerk = rate-of-change of acceleration magnitude.
    High jerk → sudden maneuver (harsh brake / sharp turn).
    """
    mag = compute_accel_magnitude(df)
    dt  = df["elapsed_seconds"].diff().replace(0, np.nan)
    return (mag.diff() / dt).abs().fillna(0)


def extract_motion_features(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-trip summary of motion features used by the risk model.

    Returns a DataFrame indexed by trip_id with columns:
        mean_magnitude, max_magnitude, max_jerk, harsh_event_count,
        overspeed_count, accel_variance, smoothness_index
    """
    records = []
    for trip_id, grp in acc_df.groupby("trip_id"):
        grp = grp.sort_values("elapsed_seconds").copy()
        mag  = compute_accel_magnitude(grp)
        jerk = compute_jerk(grp)

        harsh_count    = int((mag > ACCEL_MAGNITUDE_HARSH).sum())
        overspeed_count= int((grp["speed_kmh"] > SPEED_OVERSPEED_KMH).sum())
        accel_var      = float(mag.var())
        max_jerk       = float(jerk.max())

        # Smoothness: inverse of normalised jerk variance (0 = erratic, 1 = smooth)
        jerk_var = float(jerk.var())
        smoothness = max(0.0, 1.0 - min(jerk_var / 10.0, 1.0))

        records.append({
            "trip_id":         trip_id,
            "mean_magnitude":  float(mag.mean()),
            "max_magnitude":   float(mag.max()),
            "max_jerk":        max_jerk,
            "harsh_event_count": harsh_count,
            "overspeed_count": overspeed_count,
            "accel_variance":  accel_var,
            "smoothness_index": round(smoothness * 100, 1),
        })

    return pd.DataFrame(records).set_index("trip_id") if records else pd.DataFrame()


def get_harsh_events(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns individual harsh-event rows with labels for timeline display.
    Label: 'harsh_brake' / 'harsh_accel' / 'overspeed'
    """
    rows = []
    for trip_id, grp in acc_df.groupby("trip_id"):
        grp  = grp.sort_values("elapsed_seconds").copy()
        mag  = compute_accel_magnitude(grp)
        jerk = compute_jerk(grp)

        for idx, row in grp.iterrows():
            m = mag[idx]; j = jerk[idx]
            events = []
            if m > ACCEL_MAGNITUDE_HARSH:
                # Negative accel_y dominant → braking; positive → acceleration
                events.append("harsh_brake" if row["accel_y"] < 0 else "harsh_accel")
            if j > JERK_THRESHOLD:
                events.append("sudden_jerk")
            if row["speed_kmh"] > SPEED_OVERSPEED_KMH:
                events.append("overspeed")
            for ev in events:
                rows.append({
                    "trip_id":        trip_id,
                    "timestamp":      row["timestamp"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "event_type":     ev,
                    "magnitude":      round(m, 3),
                    "jerk":           round(j, 3),
                    "speed_kmh":      row["speed_kmh"],
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  AUDIO FEATURES
# ─────────────────────────────────────────────

def extract_audio_features(aud_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-trip audio summary for the risk model.
    """
    records = []
    for trip_id, grp in aud_df.groupby("trip_id"):
        mean_db  = float(grp["audio_level_db"].mean())
        max_db   = float(grp["audio_level_db"].max())
        std_db   = float(grp["audio_level_db"].std(ddof=0))
        spike_thresh = mean_db + AUDIO_SPIKE_STD_FACTOR * std_db

        spike_count    = int((grp["audio_level_db"] > spike_thresh).sum())
        loud_count     = int((grp["audio_classification"].isin(["loud","very_loud","argument"])).sum())
        argument_count = int((grp["audio_classification"] == "argument").sum())

        # Stress index: weighted mix of spike frequency and argument presence
        stress_idx = min(100, spike_count * 8 + argument_count * 20 + max(0, max_db - AUDIO_LOUD_DB) * 1.5)

        records.append({
            "trip_id":       trip_id,
            "mean_db":       round(mean_db, 1),
            "max_db":        round(max_db, 1),
            "std_db":        round(std_db, 1),
            "spike_count":   spike_count,
            "loud_count":    loud_count,
            "argument_count": argument_count,
            "audio_stress_index": round(stress_idx, 1),
        })

    return pd.DataFrame(records).set_index("trip_id") if records else pd.DataFrame()


def get_audio_spikes(aud_df: pd.DataFrame) -> pd.DataFrame:
    """Individual audio spike rows for timeline display."""
    rows = []
    for trip_id, grp in aud_df.groupby("trip_id"):
        mean_db = grp["audio_level_db"].mean()
        std_db  = grp["audio_level_db"].std(ddof=0)
        thresh  = mean_db + AUDIO_SPIKE_STD_FACTOR * std_db
        spikes  = grp[grp["audio_level_db"] > thresh].copy()
        spikes["trip_id"] = trip_id
        rows.append(spikes)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ─────────────────────────────────────────────
#  SENSOR FUSION  (motion + audio → conflict)
# ─────────────────────────────────────────────

def detect_conflict_moments(
    acc_df: pd.DataFrame,
    aud_df: pd.DataFrame,
    time_tolerance_sec: float = 30.0
) -> pd.DataFrame:
    """
    Overlap harsh motion events with audio spikes within a time window.
    Returns fused conflict rows with a combined_score.
    """
    harsh = get_harsh_events(acc_df)
    spikes = get_audio_spikes(aud_df)

    if harsh.empty or spikes.empty:
        return pd.DataFrame()

    conflicts = []
    for trip_id in set(harsh["trip_id"]) & set(spikes["trip_id"]):
        h = harsh[harsh["trip_id"] == trip_id]
        s = spikes[spikes["trip_id"] == trip_id]
        for _, hrow in h.iterrows():
            nearby = s[
                (s["elapsed_seconds"] >= hrow["elapsed_seconds"] - time_tolerance_sec) &
                (s["elapsed_seconds"] <= hrow["elapsed_seconds"] + time_tolerance_sec)
            ]
            if not nearby.empty:
                audio_score  = min(1.0, nearby["audio_level_db"].max() / 100)
                motion_score = min(1.0, hrow["magnitude"] / 12)
                combined     = round((audio_score + motion_score) / 2, 3)
                conflicts.append({
                    "trip_id":         trip_id,
                    "elapsed_seconds": hrow["elapsed_seconds"],
                    "timestamp":       hrow["timestamp"],
                    "motion_event":    hrow["event_type"],
                    "audio_level_db":  round(nearby["audio_level_db"].max(), 1),
                    "motion_score":    round(motion_score, 3),
                    "audio_score":     round(audio_score, 3),
                    "combined_score":  combined,
                    "severity":        "high" if combined > 0.7 else "medium" if combined > 0.4 else "low",
                })

    return pd.DataFrame(conflicts)


# ─────────────────────────────────────────────
#  DRIVER-LEVEL AGGREGATION
# ─────────────────────────────────────────────

def build_driver_safety_profile(
    driver_id: str,
    trips_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    aud_df: pd.DataFrame,
    flags_df: pd.DataFrame,
) -> Dict:
    """
    Builds a complete safety profile dict for a given driver.
    Used by both the ML risk scorer and the UI.
    """
    # Filter trips for this driver
    driver_trip_ids = trips_df[trips_df["driver_id"] == driver_id]["trip_id"].tolist()

    acc_sub  = acc_df[acc_df["trip_id"].isin(driver_trip_ids)]
    aud_sub  = aud_df[aud_df["trip_id"].isin(driver_trip_ids)]
    flag_sub = flags_df[
        (flags_df["driver_id"] == driver_id) |
        (flags_df["trip_id"].isin(driver_trip_ids))
    ]

    # Compute features
    motion_feats = extract_motion_features(acc_sub)
    audio_feats  = extract_audio_features(aud_sub)

    total_harsh = int(motion_feats["harsh_event_count"].sum()) if not motion_feats.empty else 0
    total_overspeed = int(motion_feats["overspeed_count"].sum()) if not motion_feats.empty else 0
    avg_smoothness = float(motion_feats["smoothness_index"].mean()) if not motion_feats.empty else 50.0

    total_audio_spikes = int(audio_feats["spike_count"].sum()) if not audio_feats.empty else 0
    total_arguments    = int(audio_feats["argument_count"].sum()) if not audio_feats.empty else 0
    avg_audio_stress   = float(audio_feats["audio_stress_index"].mean()) if not audio_feats.empty else 0.0

    high_flags   = int((flag_sub["severity"] == "high").sum())
    med_flags    = int((flag_sub["severity"] == "medium").sum())
    low_flags    = int((flag_sub["severity"] == "low").sum())

    return {
        "driver_id":        driver_id,
        "trip_ids":         driver_trip_ids,
        "total_trips":      len(driver_trip_ids),
        # motion
        "total_harsh":      total_harsh,
        "total_overspeed":  total_overspeed,
        "avg_smoothness":   round(avg_smoothness, 1),
        "motion_feats":     motion_feats,
        # audio
        "total_audio_spikes": total_audio_spikes,
        "total_arguments":  total_arguments,
        "avg_audio_stress": round(avg_audio_stress, 1),
        "audio_feats":      audio_feats,
        # flags
        "high_flags":  high_flags,
        "med_flags":   med_flags,
        "low_flags":   low_flags,
        "flag_df":     flag_sub,
        # raw sensor
        "acc_df":  acc_sub,
        "aud_df":  aud_sub,
    }