"""
data_loader.py
---------------
Centralised, cached data loading utilities for Driver Pulse AI.

This module exposes a high‑level API for loading all CSV datasets into
memory and filtering them down to a single driver. It also provides
simple helper functions (`load_data` and `filter_by_driver`) to remain
compatible with earlier versions of the codebase.

The core loading function `load_all_data` reads required CSVs from the
`data` directory, merges sensor tables with trip IDs to attach
`driver_id`, and performs basic datetime parsing. Results are cached
via Streamlit's `@st.cache_data` decorator so that repeated calls do
not re‑read the files.

Functions
---------
load_all_data() -> dict
    Read all CSV files into a dictionary of DataFrames. Required
    datasets must be present; optional ones will load to empty
    DataFrames if missing.

get_driver_data(driver_id: str, data: dict) -> dict
    Given a driver ID and the result of `load_all_data`, return a
    dictionary containing only the rows relevant to that driver.

get_drivers_with_sensor_data(data: dict) -> list[str]
    Return a list of driver IDs that have accelerometer data. Used
    for populating dropdowns so that blank pages never appear.

load_data(filename: str) -> pandas.DataFrame
    Backwards‑compatible helper that loads a single CSV by name.

filter_by_driver(df: pandas.DataFrame, driver_id: str) -> pandas.DataFrame
    Backwards‑compatible helper that filters a DataFrame to a single
    driver. If no driver column exists it attempts to map via `trip_id`.
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st

# Base directory where CSV files reside. This resolves to
# <project_root>/data irrespective of where the module is imported from.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _path(filename: str) -> str:
    """Return the absolute path to a CSV file in the data directory."""
    return os.path.join(DATA_DIR, filename)


def _safe_read(filename: str) -> pd.DataFrame:
    """Load a CSV if it exists; otherwise return an empty DataFrame.

    Parameters
    ----------
    filename : str
        Name of the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame or empty DataFrame on FileNotFoundError.
    """
    try:
        return pd.read_csv(_path(filename))
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_all_data() -> dict[str, pd.DataFrame]:
    """Load and lightly clean all datasets.

    This function reads a predefined set of CSV files from the
    `data` directory. Sensor tables are augmented with driver IDs by
    joining on the `trips.csv` table. Basic datetime parsing is
    performed for timestamp/date columns. Optional files (earnings,
    goals, summaries) are loaded if present; missing files yield
    empty DataFrames rather than errors. The result is cached for
    the lifetime of the Streamlit session.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A dictionary keyed by semantic names (drivers, trips, acc,
        aud, flags, goals, velocity, summaries).
    """
    # Required datasets – must exist; will raise if missing
    drivers = pd.read_csv(_path("drivers.csv"))
    trips   = pd.read_csv(_path("trips.csv"))
    acc     = pd.read_csv(_path("accelerometer_data.csv"))
    aud     = pd.read_csv(_path("audio_intensity_data.csv"))
    flags   = pd.read_csv(_path("flagged_moments.csv"))

    # Optional datasets – safe if missing
    goals     = _safe_read("driver_goals.csv")
    velocity  = _safe_read("earnings_velocity_log.csv")
    summaries = _safe_read("trip_summaries.csv")

    # Merge driver_id into sensor tables via trip_id
    trip_driver_map = trips[["trip_id", "driver_id"]]
    acc = acc.merge(trip_driver_map, on="trip_id", how="left")
    aud = aud.merge(trip_driver_map, on="trip_id", how="left")

    # Parse datetime fields where possible
    for df, col in [(acc, "timestamp"), (aud, "timestamp"), (flags, "timestamp"), (trips, "date")]:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass

    # Parse optional datetime fields
    for df, col in [(goals, "date"), (velocity, "timestamp")]:
        if not df.empty:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    return {
        "drivers":   drivers,
        "trips":     trips,
        "acc":       acc,
        "aud":       aud,
        "flags":     flags,
        "goals":     goals,
        "velocity":  velocity,
        "summaries": summaries,
    }


def get_driver_data(driver_id: str, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | dict | list]:
    """Filter all datasets down to a specific driver ID.

    Parameters
    ----------
    driver_id : str
        The ID of the driver to retrieve.
    data : dict[str, pandas.DataFrame]
        The dictionary returned from `load_all_data`.

    Returns
    -------
    dict
        A dictionary containing filtered DataFrames for the given
        driver, along with the driver's information and list of trip IDs.
    """
    trips_df = data["trips"]
    trip_ids = trips_df[trips_df["driver_id"] == driver_id]["trip_id"].tolist()

    def safe_filter_driver(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "driver_id" not in df.columns:
            return pd.DataFrame()
        return df[df["driver_id"] == driver_id]

    def safe_filter_trips(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "trip_id" not in df.columns:
            return pd.DataFrame()
        return df[df["trip_id"].isin(trip_ids)]

    return {
        "driver":    data["drivers"][data["drivers"]["driver_id"] == driver_id].iloc[0].to_dict()
                     if (data["drivers"]["driver_id"] == driver_id).any() else {},
        "trips":     trips_df[trips_df["driver_id"] == driver_id],
        "acc":       safe_filter_driver(data["acc"]),
        "aud":       safe_filter_driver(data["aud"]),
        "flags":     data["flags"][(data["flags"]["driver_id"] == driver_id) |
                                      (data["flags"]["trip_id"].isin(trip_ids))]
                     if not data["flags"].empty else pd.DataFrame(),
        "goals":     safe_filter_driver(data["goals"]),
        "velocity":  safe_filter_driver(data["velocity"]),
        "summaries": safe_filter_driver(data["summaries"]),
        "trip_ids":  trip_ids,
    }


def get_drivers_with_sensor_data(data: dict[str, pd.DataFrame]) -> list[str]:
    """Return driver IDs that have accelerometer data.

    This helper is used by UI pages to ensure dropdowns do not show
    drivers without any sensor data (which would render empty charts).
    """
    acc_df = data.get("acc", pd.DataFrame())
    if acc_df.empty or "driver_id" not in acc_df.columns:
        return sorted(data["drivers"]["driver_id"].tolist())
    acc_driver_ids = set(acc_df["driver_id"].dropna().unique())
    return sorted([d for d in data["drivers"]["driver_id"].tolist() if d in acc_driver_ids])


# ---------------------------------------------------------------------------
# Backwards‑compatible helper functions
#
# Some legacy parts of the dashboard still call `load_data` and
# `filter_by_driver` directly. To preserve those APIs we provide
# thin wrappers around `pandas.read_csv` and simple driver filtering. New
# code should prefer using `load_all_data` and `get_driver_data`.
# ---------------------------------------------------------------------------

def load_data(filename: str) -> pd.DataFrame:
    """Load a single CSV file from the data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file to load (e.g. 'drivers.csv').

    Returns
    -------
    pandas.DataFrame
        The contents of the CSV file.
    """
    return pd.read_csv(_path(filename))


def filter_by_driver(df: pd.DataFrame, driver_id: str) -> pd.DataFrame:
    """Return a subset of a DataFrame containing only rows for the given driver.

    If a `driver_id` column exists, it is used directly. Otherwise, if
    a `trip_id` column exists the function will attempt to map trip IDs
    back to drivers by loading `trip_summaries.csv` or `trips.csv`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to filter.
    driver_id : str
        Identifier of the driver to select.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only rows associated with the
        specified driver.
    """
    # If a driver column is present, use it
    if "driver_id" in df.columns:
        return df[df["driver_id"].astype(str) == str(driver_id)].copy()

    # Attempt to locate an alternative driver column (e.g. userId)
    for alt in df.columns:
        normalized = alt.lower().replace("_", "").replace("-", "")
        if normalized in {"driverid", "driver", "userid", "id"}:
            return df[df[alt].astype(str) == str(driver_id)].copy()

    # Fallback: map via trip_id if present
    if "trip_id" in df.columns:
        try:
            # Prefer trip summaries if available; fall back to trips
            try:
                trips_df = pd.read_csv(_path("trip_summaries.csv"))
            except FileNotFoundError:
                trips_df = pd.read_csv(_path("trips.csv"))
            if "trip_id" not in trips_df.columns or "driver_id" not in trips_df.columns:
                raise ValueError("Trip data must contain 'trip_id' and 'driver_id' columns to perform driver filtering.")
            valid_trip_ids = trips_df[trips_df["driver_id"].astype(str) == str(driver_id)]["trip_id"].astype(str)
            mask = df["trip_id"].astype(str).isin(set(valid_trip_ids))
            return df[mask].copy()
        except Exception:
            raise ValueError("DataFrame does not contain a driver identifier column or a trip_id column for mapping.")

    # If no driver or trip columns exist, raise an error
    raise ValueError("DataFrame does not contain a driver identifier column or a trip_id column for mapping.")
