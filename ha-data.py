# pip install pandas influxdb-client river holidays scikit-learn


import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import parser
from influxdb_client import InfluxDBClient
from river import anomaly
from river import preprocessing
import pickle
import holidays
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import zscore
from river import linear_model

# Debug settings - define these before helper functions
DEBUG_MODE = True  # Enable detailed logging
USE_TIMEZONE = False  # Set to True to use timezone-aware datetimes

# --- CONFIG ---
INFLUX_URL = "http://192.168.1.4:8086"
TOKEN = "LY86Tqy1cg5-UYTYPMmHI5opIxC2_NtLiZexyHehiqmL7YLGyHOyEeosm9JXAnoVuNaZT5TYNNcMW1eQK3qW3g=="
ORG = "myeHome"
BUCKET = "home-assistant"

# Create models directory if it doesn't exist
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"📁 Created models directory: {MODELS_DIR}")

# Model and data file paths
SAVE_MODEL_PATH = os.path.join(MODELS_DIR, "river_model.pkl")
META_MODEL_PATH = os.path.join(MODELS_DIR, "meta_model.pkl")
HOURLY_MODEL_PATH = os.path.join(MODELS_DIR, "hourly_model.pkl")
DAILY_MODEL_PATH = os.path.join(MODELS_DIR, "daily_model.pkl")
SAVE_TIMESTAMP_PATH = os.path.join(MODELS_DIR, "last_cutoff.txt")
ENTITY_HISTORY_PATH = os.path.join(MODELS_DIR, "entity_history.pkl")

# Time intervals for anomaly detection
CHUNK_SIZE = "5min"  # For immediate anomalies
HOURLY_SIZE = "1h"  # For hourly patterns
DAILY_SIZE = "1d"  # For daily patterns
MISSING_TOKEN = "NaN"
MAX_HISTORY_ENTRIES = 1000  # Limit history size

# Time span configuration
INITIAL_LOOKBACK_HOURS = 24  # Default lookback for first run (1 day)
MAX_LOOKBACK_HOURS = 168  # Maximum lookback limit (7 days)


# Helper functions for timezone handling
def make_naive(dt):
    """Convert a datetime to naive if it has timezone info"""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def format_datetime_for_display(dt):
    """Format a datetime consistently for logging/display"""
    if dt.tzinfo is not None:
        # Format with timezone info
        return f"{dt.isoformat()}"
    else:
        # For naive datetimes, indicate they are UTC (Z)
        return f"{dt.isoformat()}Z"


def ensure_timezone_consistency(dt1, dt2):
    """Make sure two datetimes have consistent timezone info (both aware or both naive)"""
    # Always make a copy to avoid modifying the original objects
    dt1_copy = dt1
    dt2_copy = dt2

    if dt1.tzinfo is not None and dt2.tzinfo is None:
        # dt1 has timezone but dt2 doesn't
        # Force both to be naive for safer comparison
        dt1_copy = make_naive(dt1)
    elif dt1.tzinfo is None and dt2.tzinfo is not None:
        # dt2 has timezone but dt1 doesn't
        # Force both to be naive for safer comparison
        dt2_copy = make_naive(dt2)

    return dt1_copy, dt2_copy


# Helper function to get current time
def get_now():
    """Get current time in a consistent format"""
    if USE_TIMEZONE:
        from datetime import timezone

        return datetime.now(timezone.utc)
    else:
        return datetime.now()


# --- Influx connection ---
client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
query_api = client.query_api()

# --- Load or init model ---
if os.path.exists(SAVE_MODEL_PATH):
    with open(SAVE_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(
        n_trees=25, height=15, window_size=250
    )

# Load or init meta model
if os.path.exists(META_MODEL_PATH):
    with open(META_MODEL_PATH, "rb") as f:
        meta_model = pickle.load(f)
else:
    meta_model = linear_model.LogisticRegression()

# Load or init hourly model
if os.path.exists(HOURLY_MODEL_PATH):
    with open(HOURLY_MODEL_PATH, "rb") as f:
        hourly_model = pickle.load(f)
else:
    hourly_model = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(
        n_trees=25, height=15, window_size=168  # ~1 week of hourly data
    )

# Load or init daily model
if os.path.exists(DAILY_MODEL_PATH):
    with open(DAILY_MODEL_PATH, "rb") as f:
        daily_model = pickle.load(f)
else:
    daily_model = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(
        n_trees=25, height=15, window_size=60  # ~2 months of daily data
    )

# Load entity change history
if os.path.exists(ENTITY_HISTORY_PATH):
    with open(ENTITY_HISTORY_PATH, "rb") as f:
        entity_history = pickle.load(f)
else:
    entity_history = []

# --- Load last cutoff time or determine earliest data ---
now = get_now()  # Get current time first
print(f"🕒 Current time: {format_datetime_for_display(now)}")

if os.path.exists(SAVE_TIMESTAMP_PATH):
    # Use saved cutoff time if available
    with open(SAVE_TIMESTAMP_PATH, "r") as f:
        last_cutoff_str = f.read().strip()
        last_cutoff = parser.isoparse(last_cutoff_str)
        print(
            f"📅 Loaded previous cutoff time: {format_datetime_for_display(last_cutoff)}"
        )

        # Ensure timezone consistency
        last_cutoff, now = ensure_timezone_consistency(last_cutoff, now)

        # Apply maximum lookback limit
        min_allowed_cutoff = now - timedelta(hours=MAX_LOOKBACK_HOURS)
        if last_cutoff < min_allowed_cutoff:
            print(
                f"⚠️ Last cutoff time exceeds maximum lookback of {MAX_LOOKBACK_HOURS} hours"
            )
            print(f"⚠️ Limiting query range to {MAX_LOOKBACK_HOURS} hours")
            last_cutoff = min_allowed_cutoff
else:
    # First run - find the earliest available data point in InfluxDB
    print("📅 No previous cutoff time found - checking earliest data in InfluxDB")

    try:
        earliest_query = f"""
        from(bucket: "{BUCKET}")
          |> range(start: -30d)
          |> filter(fn: (r) => r["_measurement"] == "state")
          |> keep(columns: ["_time"])
          |> sort(columns: ["_time"], desc: false)
          |> limit(n: 1)
        """

        earliest_df = query_api.query_data_frame(earliest_query)

        if not isinstance(earliest_df, pd.DataFrame) and len(earliest_df) > 0:
            earliest_df = pd.concat(earliest_df)

        if not earliest_df.empty:
            earliest_time = earliest_df["_time"].min()
            print(
                f"📊 Earliest data in InfluxDB: {format_datetime_for_display(earliest_time)}"
            )

            # Handle timezone consistently
            if USE_TIMEZONE:
                # Ensure both are timezone-aware
                if earliest_time.tzinfo is None:
                    from datetime import timezone

                    earliest_time = earliest_time.replace(tzinfo=timezone.utc)
            else:
                # Ensure both are timezone-naive
                if earliest_time.tzinfo is not None:
                    earliest_time = make_naive(earliest_time)

            # ALWAYS use earliest time for first run
            last_cutoff = earliest_time

            # Display with consistent formatting
            print(
                f"📅 Using earliest data point as initial cutoff: {format_datetime_for_display(last_cutoff)}"
            )

            # Add small buffer to ensure we get complete first record
            data_age_hours = (now - last_cutoff).total_seconds() / 3600
            print(f"📊 Data spans {data_age_hours:.2f} hours")

        else:
            print("⚠️ Could not find any data in InfluxDB!")
            print(f"⚠️ Using fallback lookback of {INITIAL_LOOKBACK_HOURS} hours")
            last_cutoff = now - timedelta(hours=INITIAL_LOOKBACK_HOURS)

    except Exception as e:
        print(f"⚠️ Error querying earliest data: {e}")
        print(f"⚠️ Using fallback lookback of {INITIAL_LOOKBACK_HOURS} hours")
        last_cutoff = now - timedelta(hours=INITIAL_LOOKBACK_HOURS)
        import traceback

        traceback.print_exc()

# Recalculate time formats after possible adjustments
if USE_TIMEZONE:
    start_iso = last_cutoff.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_iso = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
else:
    # For naive datetimes, assume they're UTC and add Z
    start_iso = last_cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"⏱️ Querying from {start_iso} to {stop_iso}...")
print(f"⏱️ Time span: {(now - last_cutoff).total_seconds() / 3600:.2f} hours")

# --- Query InfluxDB ---

query = f"""
from(bucket: "{BUCKET}")
  |> range(start: time(v: "{start_iso}"), stop: time(v: "{stop_iso}"))
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["_measurement"] == "state")
  |> pivot(rowKey: ["_time"], columnKey: ["entity_id"], valueColumn: "_value")
"""
df = query_api.query_data_frame(query)
df = pd.concat(df) if isinstance(df, list) else df
if df.empty:
    print("⚠️ No new data found.")
    exit()

# --- Separate numeric and text columns ---
df = df.set_index("_time").sort_index()
numeric_df = df.select_dtypes(include="number")
text_df = df.select_dtypes(include="object")


# --- Resample numeric data for different time intervals ---
print(f"📈 Raw numeric data points: {len(numeric_df)}")
print(f"📝 Raw text data points: {len(text_df)}")

numeric_df_5min = numeric_df.resample(CHUNK_SIZE).mean().interpolate().ffill().bfill()
numeric_df_hourly = (
    numeric_df.resample(HOURLY_SIZE).mean().interpolate().ffill().bfill()
)
numeric_df_daily = numeric_df.resample(DAILY_SIZE).mean().interpolate().ffill().bfill()

# Count non-interpolated data points (actual measurements)
actual_5min = len(numeric_df.resample(CHUNK_SIZE).count().dropna(how="all"))
actual_hourly = len(numeric_df.resample(HOURLY_SIZE).count().dropna(how="all"))
actual_daily = len(numeric_df.resample(DAILY_SIZE).count().dropna(how="all"))

print(f"📊 After resampling:")
print(
    f"   - 5min intervals: {len(numeric_df_5min)} total, {actual_5min} with actual data"
)
print(
    f"   - Hourly intervals: {len(numeric_df_hourly)} total, {actual_hourly} with actual data"
)
print(
    f"   - Daily intervals: {len(numeric_df_daily)} total, {actual_daily} with actual data"
)

# --- Add context features ---
ca_holidays = holidays.CA(prov="AB")


def add_context_features(chunk):
    chunk["hour"] = chunk.index.hour
    chunk["day_of_week"] = chunk.index.dayofweek
    chunk["is_weekend"] = (chunk["day_of_week"] >= 5).astype(int)
    chunk["month"] = chunk.index.month
    chunk["season"] = chunk["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    chunk["is_holiday"] = chunk.index.normalize().isin(ca_holidays).astype(int)
    return chunk


numeric_df_5min = add_context_features(numeric_df_5min)
numeric_df_hourly = add_context_features(numeric_df_hourly)
numeric_df_daily = add_context_features(numeric_df_daily)


# --- Add day period feature to hourly data ---
def add_day_period(df):
    # Morning: 5-11, Afternoon: 12-17, Evening: 18-22, Night: 23-4
    df["day_period"] = df["hour"].map(
        lambda h: (
            0 if 5 <= h <= 11 else 1 if 12 <= h <= 17 else 2 if 18 <= h <= 22 else 3
        )
    )
    return df


numeric_df_hourly = add_day_period(numeric_df_hourly)
numeric_df_5min = add_day_period(numeric_df_5min)

# --- Prepare TF-IDF encoders for text columns ---
text_df_5min = text_df.resample(CHUNK_SIZE).first().fillna("")
text_df_hourly = text_df.resample(HOURLY_SIZE).first().fillna("")
text_df_daily = text_df.resample(DAILY_SIZE).first().fillna("")

text_columns = text_df.columns.tolist()
print(f"📝 Preparing text vectorizers for {len(text_columns)} text columns")
vectorizers = {}

for col in text_columns:
    try:
        print(f"  - Initializing vectorizer for {col}")
        # Make sure we have at least one non-empty string to fit
        sample_texts = text_df[col].astype(str).tolist()
        sample_texts = [
            t for t in sample_texts if t and t.lower() not in ["nan", "none", ""]
        ]

        # If we have no valid samples, use a placeholder
        if not sample_texts:
            sample_texts = ["placeholder_text"]

        vectorizer = TfidfVectorizer(max_features=5)
        vectorizer.fit(sample_texts)
        vectorizers[col] = vectorizer

        print(f"  ✓ Vectorizer ready for {col} with {len(sample_texts)} samples")
    except Exception as e:
        print(f"  ❌ Failed to create vectorizer for {col}: {e}")
        # Create a simple fallback vectorizer
        fallback = TfidfVectorizer(max_features=1)
        fallback.fit(["fallback_text"])  # Fit with a single sample
        vectorizers[col] = fallback


def remand_compare(
    current_vector: dict,
    timestamp: pd.Timestamp,
    df_full: pd.DataFrame,
    weeks=8,
    z_thresh=3.0,
):
    """
    Compare current vector to same (weekday, hour, minute) slots in previous weeks.
    - current_vector: dict of current feature values
    - timestamp: timestamp of the current vector
    - df_full: historical data with datetime index
    - weeks: how many past weeks to compare
    - z_thresh: z-score threshold for anomaly flag
    """

    # Extract context
    target_dow = timestamp.dayofweek
    target_hour = timestamp.hour
    target_minute = timestamp.minute

    # Filter to matching weekly slot
    historical = df_full[
        (df_full.index.dayofweek == target_dow)
        & (df_full.index.hour == target_hour)
        & (df_full.index.minute == target_minute)
        & (df_full.index < timestamp)
    ].tail(
        weeks
    )  # limit to last N matching weeks

    if len(historical) < 3:
        print(f"[{timestamp}] ⚠️ Not enough matching history for remand")
        return None, False

    # Build comparison frame
    try:
        hist_df = pd.DataFrame(historical)
        cur_df = pd.DataFrame([current_vector], index=[timestamp])
        full_df = pd.concat([hist_df, cur_df])

        # Calculate z-scores
        zs = zscore(full_df, nan_policy="omit")
        current_z = zs[-1]

        # Count how many features exceed the threshold
        z_dict = dict(zip(full_df.columns, current_z))
        outliers = {k: z for k, z in z_dict.items() if abs(z) > z_thresh}
        is_anomaly = len(outliers) > 0

        print(
            f"[{timestamp}] 📊 Remand Z-anomaly | outliers: {len(outliers)}, max z: {max(outliers.values(), default=0):.2f}"
        )
        return outliers, is_anomaly

    except Exception as e:
        print(f"[{timestamp}] ❌ Remand error: {e}")
        return None, False


def compare_to_yesterday(
    current_vector: dict,
    timestamp: pd.Timestamp,
    df_full: pd.DataFrame,
    compare_ts=None,
    z_thresh=3.0,
):
    """Compare current vector to same time yesterday or specified time"""
    if compare_ts is None:
        compare_ts = timestamp - timedelta(days=1)

    if compare_ts not in df_full.index:
        print(f"[{timestamp}] ⚠️ No matching historical time for comparison")
        return None, False

    hist_vector = df_full.loc[compare_ts]

    # Compute z-score per feature
    z_scores = {}
    outlier_count = 0
    for k in current_vector:
        if k in hist_vector and pd.notna(hist_vector[k]):
            std = np.std([hist_vector[k], current_vector[k]])
            if std == 0:
                continue  # identical values → not outlier
            z = abs(current_vector[k] - hist_vector[k]) / std
            z_scores[k] = z
            if z > z_thresh:
                outlier_count += 1

    print(
        f"[{timestamp}] 🕒 Historical diff | outliers: {outlier_count}, max z: {max(z_scores.values() or [0]):.2f}"
    )
    outliers = {k: z for k, z in z_scores.items() if z > z_thresh}
    return outliers, len(outliers) > 0


# --- Process anomalies at multiple time intervals ---
def process_anomalies(
    df_numeric, df_text, interval_model, interval_name, interval_size
):
    print(f"\n🔍 Processing {interval_name} anomalies...")

    # Check if we have enough data points to process
    if len(df_numeric) < 1:  # Need at least 1 data point
        print(
            f"⚠️ No data for {interval_name} processing. Found {len(df_numeric)} samples."
        )
        return 0

    # Warning for low data volume but still process
    if len(df_numeric) < 3:
        print(
            f"⚠️ Limited data for {interval_name} processing: only {len(df_numeric)} samples available. Processing anyway."
        )

    total_samples = len(df_numeric)
    print(f"📊 Processing {total_samples} {interval_name} samples")
    print(f"🔍 Sample timestamps: {df_numeric.index[0]} to {df_numeric.index[-1]}")

    anomalies_found = 0
    total_processed = 0
    changes_detected = 0
    start_time = get_now()

    # Enable verbose mode for all dataset sizes while debugging
    verbose_mode = True if DEBUG_MODE else total_samples < 10

    # Process in smaller batches to avoid getting stuck
    batch_size = 20
    total_batches = (total_samples + batch_size - 1) // batch_size

    print(
        f"⚙️ Starting {interval_name} processing in {total_batches} batches of {batch_size} samples..."
    )

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        batch_timestamps = list(df_numeric.index[batch_start:batch_end])

        print(
            f"🔄 Processing batch {batch_num+1}/{total_batches}: samples {batch_start}-{batch_end-1}"
        )

        for i, ts in enumerate(batch_timestamps):
            current_index = batch_start + i

            try:
                # Progress reporting for every sample while debugging
                if (
                    DEBUG_MODE
                    or current_index % 3 == 0
                    or current_index == total_samples - 1
                ):
                    elapsed = max(
                        0.001, (get_now() - start_time).total_seconds()
                    )  # Avoid division by zero
                    percent_done = int((current_index / total_samples) * 100)
                    samples_per_sec = current_index / elapsed
                    remaining = max(1, total_samples - current_index)
                    eta_seconds = remaining / max(0.1, samples_per_sec)

                    print(
                        f"⏳ {interval_name} progress: {percent_done}% ({current_index+1}/{total_samples}) - {samples_per_sec:.1f} samples/sec, ETA: {eta_seconds:.0f}s"
                    )

                if DEBUG_MODE:
                    print(
                        f"  📍 [{current_index+1}/{total_samples}] Processing timestamp {ts}"
                    )

                # Extract features for this timestamp
                x = df_numeric.loc[ts].to_dict()

                if DEBUG_MODE:
                    print(f"  📊 Features extracted: {len(x)} numeric features at {ts}")

                total_processed += 1

                # TF-IDF: handle text features
                if DEBUG_MODE:
                    print(f"  📝 Processing text features...")

                # Process text features if available
                if ts in df_text.index:  # Re-enabled text processing
                    if DEBUG_MODE:
                        print(f"  📝 Processing text features for {ts}")
                    for col in text_columns:
                        if col not in vectorizers:
                            if DEBUG_MODE:
                                print(f"  ⚠️ No vectorizer for column {col}")
                            continue

                        vec = vectorizers[col]
                        text = (
                            str(df_text.at[ts, col])
                            if ts in df_text.index
                            else MISSING_TOKEN
                        )
                        if not text or text.lower() in ["nan", "none", ""]:
                            text = MISSING_TOKEN

                        if DEBUG_MODE:
                            print(f"  📄 Text for {col}: '{text}'")

                        tfidf = vec.transform([text]).toarray()[0]
                        for j, v in enumerate(tfidf):
                            x[f"{col}_tfidf_{j}"] = v
                elif DEBUG_MODE:
                    print(f"  ⚠️ No text features found for {ts}")

                x = {k: v for k, v in x.items() if pd.notna(v)}

                if DEBUG_MODE:
                    print(
                        f"  🧮 Scoring with model: {len(x)} features after preprocessing"
                    )

                # Score with the appropriate interval model
                score = interval_model.score_one(x)
                interval_model.learn_one(x)

                if DEBUG_MODE:
                    print(f"  📈 Model score: {score:.4f}")

                # Check for text changes
                prev_ts = ts - pd.Timedelta(interval_size)
                changed_entities = []
                change_details = {}

                for col in text_columns:
                    if ts not in df_text.index or prev_ts not in df_text.index:
                        continue

                    now = str(df_text.at[ts, col])
                    prev = str(df_text.at[prev_ts, col])

                    if not now or now.lower() in ["nan", "none", ""]:
                        now = MISSING_TOKEN
                    if not prev or prev.lower() in ["nan", "none", ""]:
                        prev = MISSING_TOKEN

                    changed = int(now != prev)
                    x[f"{col}_changed_{interval_name}"] = changed
                    if changed:
                        changed_entities.append(col)
                        change_details[col] = f"{prev} → {now}"

                # Only process further if there are changes
                if changed_entities:
                    changes_detected += 1
                    print(
                        f"[{ts}] 📊 {interval_name.upper()} Score: {score:.4f} | Changed: {changed_entities}"
                    )

                    # --- REMAND ---
                    remand_outliers, remand_flag = remand_compare(x, ts, df_numeric)

                    # --- YESTERDAY ---
                    if interval_name == "daily":
                        y_compare_ts = ts - timedelta(
                            days=7
                        )  # Compare to last week for daily
                    else:
                        y_compare_ts = ts - timedelta(
                            days=1
                        )  # Compare to yesterday for others

                    y_outliers, y_flag = compare_to_yesterday(
                        x, ts, df_numeric, y_compare_ts
                    )

                    # Is this an anomaly by any detection method?
                    is_anomaly = (score > 0.4) or remand_flag or y_flag

                    if is_anomaly:
                        anomalies_found += 1
                        print(f"[{ts}] 🚨 {interval_name.upper()} ANOMALY DETECTED")

                        # Generate explanations for changed entities
                        explanations = {}
                        for entity, change in change_details.items():
                            # Parse out the before and after values
                            parts = change.split("→")
                            prev_value = parts[0].strip()
                            current_value = (
                                parts[1].strip() if len(parts) > 1 else change
                            )

                            # Generate explanation
                            explanation = generate_anomaly_explanation(
                                ts,
                                entity,
                                prev_value,
                                current_value,
                                entity_history[:-1],
                            )
                            explanations[entity] = (
                                f"[{interval_name.upper()}] {explanation}"
                            )
                            print(f"[{ts}] ⚠️ {explanations[entity]}")

                        # Store in history
                        history_entry = {
                            "timestamp": ts,
                            "interval": interval_name,
                            "is_anomaly": is_anomaly,
                            "changes": change_details,
                            "explanations": explanations,
                            "score": score,
                            "remand_flag": remand_flag,
                            "compare_flag": y_flag,
                        }
                        entity_history.append(history_entry)

                        # Keep history within size limit
                        if len(entity_history) > MAX_HISTORY_ENTRIES:
                            entity_history.pop(0)  # Remove oldest entry

            except Exception as e:
                print(f"[{ts}] ❌ {interval_name.upper()} ERROR: {e}")
                import traceback

                traceback.print_exc()  # Print full exception traceback

        # After each batch, log completion and reset intermediate state if needed
        print(f"✅ Completed batch {batch_num+1}/{total_batches} for {interval_name}")

    # Final progress and timing information
    elapsed_time = (get_now() - start_time).total_seconds()
    samples_per_sec = total_samples / max(0.001, elapsed_time)

    print(
        f"✅ {interval_name.capitalize()} processing complete in {elapsed_time:.1f} seconds ({samples_per_sec:.1f} samples/sec)"
    )
    print(
        f"📈 Stats: {anomalies_found} anomalies, {changes_detected} changes in {total_samples} samples"
    )

    return anomalies_found


# --- Merge numeric + encoded text and run model ---
# Process each time interval based on available data
anomaly_counts = {"5min": 0, "hourly": 0, "daily": 0}
any_processing_done = False

# Convert time span to different units
time_span_hours = (now - last_cutoff).total_seconds() / 3600  # hours
time_span_minutes = time_span_hours * 60  # minutes
time_span_days = time_span_hours / 24  # days

print(
    f"\n⏱️ Time span details: {time_span_minutes:.1f} minutes, {time_span_hours:.2f} hours, {time_span_days:.2f} days"
)

# Process 5min intervals only if we have enough time span and data
min_5min_span = 5  # minimum minutes needed for 5min processing
if time_span_minutes >= min_5min_span and actual_5min > 0:
    anomaly_counts["5min"] = process_anomalies(
        numeric_df_5min, text_df_5min, model, "5min", CHUNK_SIZE
    )
    any_processing_done = True
else:
    print(
        f"\n⏱️ Skipping 5min processing - only {time_span_minutes:.1f} minutes of data available (need at least {min_5min_span}) or no actual data points"
    )

# Check if we have enough data for hourly processing
if time_span_hours >= 1 and actual_hourly > 0:
    anomaly_counts["hourly"] = process_anomalies(
        numeric_df_hourly, text_df_hourly, hourly_model, "hourly", HOURLY_SIZE
    )
    any_processing_done = True
else:
    print(
        f"\n⏱️ Skipping hourly processing - only {time_span_hours:.2f} hours of data available (need at least 1) or no actual data points"
    )

# Check if we have enough data for daily processing
if time_span_days >= 1 and actual_daily > 0:
    anomaly_counts["daily"] = process_anomalies(
        numeric_df_daily, text_df_daily, daily_model, "daily", DAILY_SIZE
    )
    any_processing_done = True
else:
    print(
        f"\n⏱️ Skipping daily processing - only {time_span_days:.2f} days of data available (need at least 1) or no actual data points"
    )

# --- Save models and new cutoff ---
if any_processing_done:
    # Only save models if we did some processing
    with open(SAVE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(META_MODEL_PATH, "wb") as f:
        pickle.dump(meta_model, f)

    with open(HOURLY_MODEL_PATH, "wb") as f:
        pickle.dump(hourly_model, f)

    with open(DAILY_MODEL_PATH, "wb") as f:
        pickle.dump(daily_model, f)

    # Save the cutoff timestamp in ISO format with proper timezone info
    with open(SAVE_TIMESTAMP_PATH, "w") as f:
        # Always save in a consistent format
        if now.tzinfo is not None:
            # Already has timezone - save with timezone info
            f.write(now.isoformat())
        else:
            # Naive datetime - add Z to indicate UTC
            f.write(now.isoformat() + "Z")
        print(f"✅ Cutoff timestamp updated to: {format_datetime_for_display(now)}")

    # Save entity change history
    with open(ENTITY_HISTORY_PATH, "wb") as f:
        pickle.dump(entity_history, f)
    print("✅ Models updated and entity history saved.")
else:
    print("⚠️ No processing was done - models and cutoff timestamp NOT updated")
    print("⏱️ Will try again from the same cutoff time next run to accumulate more data")

print(
    f"📊 Anomaly summary: {anomaly_counts['5min']} 5-minute, {anomaly_counts['hourly']} hourly, {anomaly_counts['daily']} daily"
)

# Print recent entity changes for reference
if entity_history:
    print("\n🔍 Recent anomalies:")
    # Get the 5 most recent anomalies
    recent_anomalies = [e for e in entity_history if e.get("is_anomaly", False)]
    recent_anomalies.sort(key=lambda x: x["timestamp"], reverse=True)

    for entry in recent_anomalies[:3]:  # Show last 3 anomalies
        ts = entry["timestamp"]
        interval = entry.get("interval", "unknown")
        print(f"[{ts}] ⚠️ {interval.upper()} ANOMALY")
        for entity, explanation in entry.get("explanations", {}).items():
            print(f"  • {explanation}")
        print("")


def get_recent_anomalies(max_entries=5):
    """Return recent anomalies for Home Assistant to use in automations"""
    if not os.path.exists(ENTITY_HISTORY_PATH):
        return []

    try:
        with open(ENTITY_HISTORY_PATH, "rb") as f:
            history = pickle.load(f)

        # Filter only anomalies
        anomalies = [e for e in history if e.get("is_anomaly", False)]

        # Sort by timestamp (newest first) and limit
        anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
        recent = anomalies[:max_entries]

        # Format for Home Assistant
        results = []
        for entry in recent:
            for entity, explanation in entry.get("explanations", {}).items():
                results.append(
                    {
                        "timestamp": entry["timestamp"].isoformat(),
                        "entity_id": entity,
                        "explanation": explanation,
                    }
                )

        return results
    except Exception as e:
        print(f"Error getting recent anomalies: {e}")
        return []


def generate_anomaly_explanation(
    ts, entity, prev_value, current_value, historical_data
):
    """Generate natural language explanation for an anomaly"""
    # Get day of week and hour for context
    dow = ts.strftime("%A")
    hour = ts.hour

    # Check if we have enough history for this entity
    if historical_data is None or len(historical_data) < 3:
        return f"Unusual activity detected: {entity} changed from {prev_value} to {current_value}"

    # Find typical value for this time slot
    similar_times = []
    for hist_entry in historical_data:
        hist_ts = hist_entry["timestamp"]
        if (
            hist_ts.hour == hour
            and hist_ts.dayofweek == ts.dayofweek
            and entity in hist_entry.get("changes", {})
        ):
            similar_times.append(hist_entry)

    if not similar_times:
        return f"Unusual activity detected: {entity} changed to {current_value} at {hour}:00 on {dow}"

    # Generate explanation based on patterns
    time_str = f"{hour}:00"

    if (
        "door" in entity.lower()
        or "window" in entity.lower()
        or "lock" in entity.lower()
    ):
        # Door/window state explanation
        if "open" in current_value.lower() or "unlocked" in current_value.lower():
            return f"Unusual: {entity} is {current_value} at {time_str} on {dow}, but is normally closed at this time"
        else:
            return f"Unusual: {entity} is {current_value} at {time_str} on {dow}, but is normally open at this time"

    elif "light" in entity.lower() or "lamp" in entity.lower():
        # Light state explanation
        if "on" in current_value.lower():
            return f"Unusual: {entity} is on at {time_str} on {dow}, but is normally off at this time"
        else:
            return f"Unusual: {entity} is off at {time_str} on {dow}, but is normally on at this time"

    elif (
        "temperature" in entity.lower()
        or "humidity" in entity.lower()
        or "sensor" in entity.lower()
    ):
        # Sensor value explanation
        try:
            curr_num = float(current_value.split("→")[1].strip())
            return f"Unusual {entity} reading of {curr_num} at {time_str} on {dow} (outside normal range)"
        except:
            return f"Unusual {entity} reading at {time_str} on {dow}: {current_value}"

    # Generic explanation for other entities
    return (
        f"Unusual activity: {entity} changed to {current_value} at {time_str} on {dow}"
    )
