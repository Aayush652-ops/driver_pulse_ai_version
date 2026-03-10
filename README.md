# 🚗 Driver Pulse - Group 4

> **An AI-powered Personal Driver Companion** that analyzes real-time sensor data, performance logs, and earning trends to improve safety, productivity, and well-being for gig economy drivers.

---

## 🔗 Live Deployment & Demo

| Resource | Link |
|----------|------|
| **Live Application** | https://driverpulseaiversion-6xj2xy9ekuxqejuzibdxdb.streamlit.app/ |
| **GitHub Repository** | https://github.com/Aayush652-ops/driver_pulse_ai_version |
| **Demo Video** | https://drive.google.com/file/d/1g9Ntm-34iygFclV1jTOVlKDk7-HkO9hA/view?usp=sharing |

---

## Judge Login Credentials

We have created a dummy driver, to access a particular driver's dashboard. 

Use the following credentials for driver login-
1. **Email** — abc@gmail.com
2. **Password** — 12345

If you are unable to login, then go to sign up and use the following information-
1. **Name** - Devansh Verma 
2. **Driver ID** - SDRV039
Email and Password remain same as above, for login.


## 📐 System Overview

Driver Pulse ingests raw accelerometer, audio intensity, and trip data to produce three real-time driver insights:

1. **Safety Score** — harsh event detection via sensor fusion (motion + audio)
2. **Earnings & Goals Dashboard** — velocity tracking, goal forecasting, hourly pattern analysis
3. **Burnout Monitor** — multi-factor risk scoring combining work intensity, stress signals, and earnings pressure

---

## 🏗️ Repository Structure

```
driver_pulse_ai_version/
├── app.py                  # Streamlit entry point & auth routing
├── requirements.txt        # Python dependencies
├── data/                   # Input CSVs (see Data Sources below)
│   ├── trips.csv
│   ├── accelerometer_data.csv
│   ├── audio_intensity_data.csv
│   ├── flagged_moments.csv
│   ├── driver_goals.csv
│   └── trip_summaries.csv
├── pages/
│   ├── 2_🛡️_My_Safety.py      # Safety & harsh event page
│   ├── 3_💰_Earnings_Goals.py  # Earnings velocity & goal tracking
│   └── 4_🔥_Burnout_Monitor.py # Burnout risk dashboard
├── utils/
│   ├── data_loader.py          # Centralised, cached CSV loading
│   ├── feature_engineering.py  # Sensor feature extraction
│   ├── earnings_analytics.py   # Velocity, forecasting, goal logic
│   └── burnout_detection.py    # Multi-factor burnout scoring
├── auth/                   # Login/session management
├── models/                 # ML model artifacts (if any)
└── database/               # SQLite or DB config
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aayush652-ops/driver_pulse_ai_version.git
cd driver_pulse_ai_version

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
streamlit run app.py
```

### Dependencies (`requirements.txt`)
```
streamlit
pandas
numpy
plotly
scikit-learn
```

---

## 📊 Data Sources & Schema

All data lives in `data/`. The system is designed to work with both live sensor streams and static CSV files for demo purposes.

| File | Key Columns | Used By |
|------|-------------|---------|
| `trips.csv` | trip_id, driver_id, date, start_time, end_time, duration_min, fare, pickup_location, dropoff_location | All pages |
| `accelerometer_data.csv` | trip_id, timestamp, elapsed_seconds, accel_x, accel_y, accel_z, speed_kmh | Safety page, burnout |
| `audio_intensity_data.csv` | trip_id, timestamp, elapsed_seconds, audio_level_db, audio_classification | Safety page, burnout |
| `flagged_moments.csv` | flag_id, trip_id, driver_id, timestamp, flag_type, severity, motion_score, audio_score | Burnout monitor |
| `driver_goals.csv` | goal_id, driver_id, target_earnings, current_earnings, current_hours, status, earnings_velocity, goal_completion_forecast | Earnings page |
| `trip_summaries.csv` | trip_id, driver_id, fare, stress_score, trip_quality_rating | Earnings page |

---

## 🧠 Algorithm Design

### 1. Harsh Event Detection (`feature_engineering.py`)

```
Raw accel (x,y,z) → magnitude = √(x²+y²+z²) → threshold @ 8.5g
Jerk = Δmagnitude/Δtime → threshold @ 3.0 m/s³
Audio level → spike if > mean + 1.8σ per trip
Fusion: motion event within ±30s of audio spike → conflict_moment
```

**Thresholds** (tunable in `feature_engineering.py`):
- `ACCEL_MAGNITUDE_HARSH = 8.5` — combined g-force for harsh event
- `JERK_THRESHOLD = 3.0` — m/s³ for sudden maneuver
- `SPEED_OVERSPEED_KMH = 55` — overspeed flag
- `AUDIO_LOUD_DB = 68` / `AUDIO_VERY_LOUD_DB = 80`

### 2. Earnings Velocity Algorithm (`earnings_analytics.py`)

**Source of truth hierarchy** (handles sparse trip data):
1. `driver_goals.csv` → authoritative `current_earnings` (accumulated across all trips)
2. velocity log → time-based metrics (elapsed_hours, current_velocity)
3. `trips.csv` → trip list display only (may be incomplete)

```
earnings_velocity = current_earnings / current_hours  ($/hr)
required_velocity = remaining_earnings / remaining_hours
forecast = CSV goal_completion_forecast field (pre-computed with full trip history)
projected_final = current_earnings + (current_velocity × remaining_hours)
```

**Cold start handling**: When `current_hours < 0.5`, velocity is unreliable. The system falls back to the pre-computed `goal_completion_forecast` from the goals CSV rather than calculating from sparse data.

### 3. Burnout Risk Scoring (`burnout_detection.py`)

Three-component weighted score:

| Component | Weight | Key Signals |
|-----------|--------|-------------|
| Work Intensity | 35% | Shift hours (from goals CSV), trips/hour rate |
| Stress Signals | 40% | harsh_braking, moderate_brake, audio_spike, conflict_moment, sustained_stress |
| Earnings Pressure | 25% | Goal gap %, at_risk forecast from CSV |

```
burnout_score = intensity × 0.35 + stress × 0.40 + pressure × 0.25
HIGH ≥ 40   |   MODERATE ≥ 22   |   LOW < 22
```

**Threshold calibration**: Thresholds are calibrated to the actual data distribution (median driver: 0 flags, 4.2 shift hours). The HIGH/MODERATE boundaries represent the top ~5% and ~35% of drivers respectively.

---

## ⚠️ Trade-offs & Assumptions

| Decision | What We Chose | What We Sacrificed |
|----------|--------------|-------------------|
| `trips.csv` is sparse | Use `driver_goals.csv` as earnings source of truth | Can't show per-trip breakdown for all trips |
| No velocity log | Derive `elapsed_hours` from `trips.duration_min` or goals `current_hours` | Can't show intra-shift velocity trend |
| Burnout thresholds | Calibrated to demo dataset distribution | Would need recalibration on production data |
| Audio conflict detection | ±30s window for motion+audio fusion | May over/under-count at trip boundaries |
| Goal forecast | Trust CSV `goal_completion_forecast` field | Forecast is a snapshot, not real-time recalculation |


## 👥 Team

| Member | Role |
|--------|------|
| Aayush Gupta | Data Infrastructure & Earnings Analytics Lead |
| Saisha Verma | Safety & Sensor Intelligence Lead |
| Aanvi Jha | Earnings & Goal Tracking Lead |

---

*Driver Pulse — She++ Hackathon 2026*

