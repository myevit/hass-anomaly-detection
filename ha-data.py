# pip install pandas influxdb-client river holidays scikit-learn


import os
import pandas as pd
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
import numpy as np


class ModelWrapper:
    def __init__(self, model, last_cutoff):
        self.model = model
        self.last_cutoff = last_cutoff


# --- CONFIG ---
INFLUX_URL = "http://192.168.1.4:8086"
TOKEN = "LY86Tqy1cg5-UYTYPMmHI5opIxC2_NtLiZexyHehiqmL7YLGyHOyEeosm9JXAnoVuNaZT5TYNNcMW1eQK3qW3g=="
ORG = "myeHome"
BUCKET = "home-assistant"
MODELS_DIR = "models"
SAVE_MODEL_PATH = os.path.join(MODELS_DIR, "river_model.pkl")
META_MODEL_PATH = os.path.join(MODELS_DIR, "meta_model.pkl")
SAVE_TIMESTAMP_PATH = "last_cutoff.txt"
CHUNK_SIZE = "5min"
MISSING_TOKEN = "NaN"
DEFAULT_DAYS = 7  # Default number of days to look back

# --- Add at the top level, after the config section ---
# Create a list to store all the changes during processing
all_entity_changes = []

# --- Influx connection ---
client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
query_api = client.query_api()

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Load or init model ---
if os.path.exists(SAVE_MODEL_PATH):
    with open(SAVE_MODEL_PATH, "rb") as f:
        model_wrapper = pickle.load(f)
        model = model_wrapper.model
        last_cutoff = model_wrapper.last_cutoff
else:
    model = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(
        n_trees=25, height=15, window_size=250
    )
    last_cutoff = datetime.now(timezone.utc) - timedelta(days=DEFAULT_DAYS)

# Load or init meta model
if os.path.exists(META_MODEL_PATH):
    with open(META_MODEL_PATH, "rb") as f:
        meta_model_wrapper = pickle.load(f)
        meta_model = meta_model_wrapper.model
        meta_last_cutoff = meta_model_wrapper.last_cutoff
else:
    meta_model = linear_model.LogisticRegression()
    meta_last_cutoff = datetime.now(timezone.utc) - timedelta(days=DEFAULT_DAYS)

# Get latest timestamp from InfluxDB
latest_query = f"""
from(bucket: "{BUCKET}")
  |> range(start: -{DEFAULT_DAYS}d)
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["_measurement"] == "state")
  |> first()
  |> pivot(rowKey: ["_time"], columnKey: ["entity_id"], valueColumn: "_value")
"""

# Suppress InfluxDB warnings
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

warnings.simplefilter("ignore", MissingPivotFunction)

latest_df = query_api.query_data_frame(latest_query)
latest_df = pd.concat(latest_df) if isinstance(latest_df, list) else latest_df

if not latest_df.empty:
    latest_time = pd.to_datetime(latest_df["_time"].iloc[0])
    if latest_time > last_cutoff:
        last_cutoff = latest_time
        print(f"📅 Adjusted start time to latest InfluxDB record: {last_cutoff}")

now = datetime.now(timezone.utc)
start = last_cutoff.replace(microsecond=0).isoformat()
stop_time = now.replace(microsecond=0)
stop = stop_time.isoformat()
print(f"⏱️ Querying from {start} to {stop}...")


# --- Query InfluxDB ---

query = f"""
from(bucket: "{BUCKET}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) =>
      (r["_field"] == "value" or r["_field"] == "state") 
  )
  |> map(fn: (r) => ({{
      r with entity_id: r.domain + "." + r.entity_id
  }}))
  |> drop(columns: ["domain", "_measurement", "_start", "_stop"])
"""
df = query_api.query_data_frame(query)
df = pd.concat(df) if isinstance(df, list) else df
if df.empty:
    print("⚠️ No new data found.")
    exit()

# Print shape of raw dataframe
print(f"ℹ️ Query returned {len(df)} rows")

# --- Process the data with a cleaner approach ---
# First split by field type
numeric_df = df[df["_field"] == "value"]
text_df = df[df["_field"] == "state"]

# Then pivot each separately
numeric_df = numeric_df.pivot(
    index="_time", columns="entity_id", values="_value"
).astype(float)
text_df = text_df.pivot(index="_time", columns="entity_id", values="_value").astype(str)

# Fill missing values and resample
numeric_df = numeric_df.interpolate().ffill().bfill()
text_df = text_df.fillna(MISSING_TOKEN)

# Resample to regular intervals
numeric_df = numeric_df.resample(CHUNK_SIZE).mean()
text_df = text_df.resample(CHUNK_SIZE).first()

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


numeric_df = add_context_features(numeric_df)

# --- Prepare TF-IDF encoders for text columns ---
text_df = text_df.resample(CHUNK_SIZE).first().fillna("")
text_columns = text_df.columns.tolist()
vectorizers = {col: TfidfVectorizer(max_features=5) for col in text_columns}
for col in text_columns:
    try:
        # Ensure all values are strings and not empty
        text_data = text_df[col].astype(str).replace(["nan", "None", ""], MISSING_TOKEN)
        vectorizers[col].fit(text_data)
    except Exception as e:
        print(f"⚠️ Error fitting vectorizer for {col}: {e}")
        vectorizers[col] = None  # Mark failed vectorizers


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
        # print(f"[{timestamp}] ⚠️ Not enough matching history for remand")
        return None, False

    # Build comparison frame
    try:
        # Convert current vector values to float
        current_values = {
            k: float(v)
            for k, v in current_vector.items()
            if isinstance(v, (int, float))
        }

        # Create DataFrame with proper numeric types
        hist_df = pd.DataFrame(historical).astype(float)
        cur_df = pd.DataFrame([current_values], index=[timestamp])
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
    current_vector: dict, timestamp: pd.Timestamp, df_full: pd.DataFrame, z_thresh=3.0
):
    """Compare current vector to same time yesterday"""
    yesterday_time = timestamp - timedelta(days=1)

    if yesterday_time not in df_full.index:
        # print(f"[{timestamp}] ⚠️ No matching time yesterday")
        return None, False

    hist_vector = df_full.loc[yesterday_time]

    # Compute z-score per feature
    z_scores = {}
    outlier_count = 0
    for k in current_vector:
        if k in hist_vector and pd.notna(hist_vector[k]):
            try:
                # Ensure both values are numeric
                current_val = float(current_vector[k])
                hist_val = float(hist_vector[k])

                # Calculate standard deviation
                values = [current_val, hist_val]
                std = np.std(values)
                if std == 0:
                    continue  # identical values → not outlier
                z = abs(current_val - hist_val) / std
                z_scores[k] = z
                if z > z_thresh:
                    outlier_count += 1
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error calculating z-score for {k}: {e}")
                continue

    print(
        f"[{timestamp}] 🕒 Yesterday diff | outliers: {outlier_count}, max z: {max(z_scores.values() or [0]):.2f}"
    )
    outliers = {k: z for k, z in z_scores.items() if z > z_thresh}
    return outliers, len(outliers) > 0


# --- Track entity changes ---
def track_entity_changes(
    current_data, previous_data, current_ts, prev_ts, text_cols=None
):
    """Track changes in entity values between current and previous timestamps.
    Returns a dictionary of changed entities with their old and new values."""
    changes = {}

    # Track numeric changes
    for col in current_data.columns:
        if col in previous_data.columns:
            curr_val = current_data[col].iloc[0]
            prev_val = previous_data[col].iloc[0]

            if pd.notna(curr_val) and pd.notna(prev_val) and curr_val != prev_val:
                # Use column name directly - no validation or transformation
                changes[col] = {
                    "previous": prev_val,
                    "current": curr_val,
                    "diff": (
                        curr_val - prev_val
                        if isinstance(curr_val, (int, float))
                        else None
                    ),
                }

    # Track text changes if provided
    if text_cols is not None:
        for col in text_cols:
            if col not in current_data or col not in previous_data:
                continue

            curr_text = current_data.get(col, pd.Series([MISSING_TOKEN])).iloc[0]
            prev_text = previous_data.get(col, pd.Series([MISSING_TOKEN])).iloc[0]

            curr_text = str(curr_text) if pd.notna(curr_text) else MISSING_TOKEN
            prev_text = str(prev_text) if pd.notna(prev_text) else MISSING_TOKEN

            if curr_text != prev_text and curr_text != MISSING_TOKEN:
                # Use column name directly - no validation or transformation
                changes[col] = {"previous": prev_text, "current": curr_text}

    return changes


# --- Merge numeric + encoded text and run model ---
for ts in numeric_df.index:
    try:
        x = numeric_df.loc[ts].to_dict()

        # --- TF-IDF: handle text features ---
        if ts in text_df.index:
            for col in text_columns:
                if vectorizers[col] is None:
                    continue
                try:
                    text = (
                        str(text_df.at[ts, col])
                        if ts in text_df.index
                        else MISSING_TOKEN
                    )
                    if not text or text.lower() in ["nan", "none", ""]:
                        text = MISSING_TOKEN
                    tfidf = vectorizers[col].transform([text]).toarray()[0]
                    for i, v in enumerate(tfidf):
                        x[f"{col}_tfidf_{i}"] = float(v)  # Ensure float conversion
                except Exception as e:
                    print(f"⚠️ Error processing text feature {col} at {ts}: {e}")
                    continue

        # Ensure all feature values are float
        x = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in x.items()
            if pd.notna(v)
        }

        # --- RIVER ---
        score = model.score_one(x)
        model.learn_one(x)

        # Track changes for anomalies
        entity_changes = {}
        prev_ts = ts - pd.Timedelta(CHUNK_SIZE)

        # Store anomaly details if score is significant
        if abs(score) > 0:
            print(f"[{ts}] 🤖 River Score: {score:.4f}")

            # Get current and previous data frames
            current_frame = numeric_df.loc[[ts]]
            if prev_ts in numeric_df.index:
                prev_frame = numeric_df.loc[[prev_ts]]
                entity_changes = track_entity_changes(
                    current_frame,
                    prev_frame,
                    ts,
                    prev_ts,
                    (
                        text_columns
                        if ts in text_df.index and prev_ts in text_df.index
                        else None
                    ),
                )

                if entity_changes:
                    print(f"[{ts}] 📊 Changed entities:")
                    for entity, change in entity_changes.items():
                        change_str = f"'{change['previous']}' → '{change['current']}'"
                        if "diff" in change and change["diff"] is not None:
                            change_str += f" (diff: {change['diff']})"
                        print(f"  - {entity}: {change_str}")

        # --- REMAND ---
        remand_outliers, remand_flag = remand_compare(x, ts, numeric_df)

        # --- YESTERDAY ---
        y_outliers, y_flag = compare_to_yesterday(x, ts, numeric_df)

        if y_flag:
            print(
                f"[{ts}] ⚠️ Anomaly vs yesterday | features: {list(y_outliers.keys())}"
            )
        if remand_flag:
            print(
                f"[{ts}] 🚨 Remand anomaly | features: {list(remand_outliers.keys())}"
            )

        # --- META MODEL ---
        try:
            # Include entity changes in meta model features
            x_meta = {
                "river_score": float(score),
                "remand_outlier_count": float(
                    len(remand_outliers) if remand_outliers else 0
                ),
                "yesterday_outlier_count": float(len(y_outliers) if y_outliers else 0),
                "entity_change_count": float(len(entity_changes)),
            }
            # Add numeric features
            for k, v in x.items():
                if isinstance(v, (int, float)):
                    x_meta[k] = float(v)

            # Track text changes in last 5 minutes
            changed_entities = []
            entity_change_details = {}

            for col in text_columns:
                try:
                    now = (
                        str(text_df.at[ts, col])
                        if ts in text_df.index
                        else MISSING_TOKEN
                    )
                    prev = (
                        str(text_df.at[prev_ts, col])
                        if prev_ts in text_df.index
                        else MISSING_TOKEN
                    )

                    if not now or now.lower() in ["nan", "none", ""]:
                        now = MISSING_TOKEN
                    if not prev or prev.lower() in ["nan", "none", ""]:
                        prev = MISSING_TOKEN

                    changed = int(now != prev)
                    x_meta[f"{col}_changed_5min"] = float(changed)
                    if changed:
                        # Use the column name directly without transformation
                        changed_entities.append(col)
                        entity_change_details[col] = {"previous": prev, "current": now}
                except Exception as e:
                    print(f"⚠️ Error processing text change for {col}: {e}")

            # Don't include non-numeric data in x_meta
            x_meta["changed_count"] = float(len(changed_entities))

            is_meta_anomaly = meta_model.predict_one(x_meta)
            if is_meta_anomaly:
                # Print the actual entities that changed
                if changed_entities:
                    entities_str = ", ".join(changed_entities)
                    print(f"[{ts}] 🧠 Meta anomaly: | Changed entities: {entities_str}")
                    # Print the details of each changed entity
                    print(f"[{ts}] 📱 Entity change details:")
                    for entity, change in entity_change_details.items():
                        print(
                            f"  - {entity}: '{change['previous']}' → '{change['current']}'"
                        )
                else:
                    print(f"[{ts}] 🧠 Meta anomaly: | No text entities changed")

            # Optional: implicit label = any model flagged
            y = float(remand_flag or y_flag or score > 0.4)
            meta_model.learn_one(x_meta, y)

            # Instead of directly updating model_wrapper, collect the changes
            if entity_changes and (abs(score) > 0 or remand_flag or y_flag):
                # Include text entity changes too
                if entity_change_details:
                    entity_changes.update(entity_change_details)

                all_entity_changes.append(
                    {
                        "timestamp": ts,
                        "score": score,
                        "changes": entity_changes,
                        "ha_entities": changed_entities,  # Use the entities directly
                    }
                )
        except Exception as e:
            print(f"[{ts}] ⚠️ Meta model error: {e}")

    except Exception as e:
        print(f"[{ts}] ⚠️ Skipped due to error: {e}")
        import traceback

        traceback.print_exc()


# --- Update the model wrapper with the collected changes ---
if all_entity_changes:
    if not hasattr(model, "change_history"):
        model.change_history = []
    model.change_history.extend(all_entity_changes)
    # Keep only the last 100 changes to avoid memory issues
    if len(model.change_history) > 100:
        model.change_history = model.change_history[-100:]

# --- Save model and new cutoff ---
model_wrapper = ModelWrapper(model, stop_time)
with open(SAVE_MODEL_PATH, "wb") as f:
    pickle.dump(model_wrapper, f)

meta_model_wrapper = ModelWrapper(meta_model, stop_time)
with open(META_MODEL_PATH, "wb") as f:
    pickle.dump(meta_model_wrapper, f)


print("✅ Model updated and saved.")
