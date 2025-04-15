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


# --- Resample numeric data ---
numeric_df = numeric_df.resample(CHUNK_SIZE).mean().interpolate().ffill().bfill()

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
        vectorizers[col].fit(text_df[col].astype(str))
    except:
        pass


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
    current_vector: dict, timestamp: pd.Timestamp, df_full: pd.DataFrame, z_thresh=3.0
):
    """Compare current vector to same time yesterday"""
    yesterday_time = timestamp - timedelta(days=1)

    if yesterday_time not in df_full.index:
        print(f"[{timestamp}] ⚠️ No matching time yesterday")
        return None, False

    hist_vector = df_full.loc[yesterday_time]

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
        f"[{timestamp}] 🕒 Yesterday diff | outliers: {outlier_count}, max z: {max(z_scores.values() or [0]):.2f}"
    )
    outliers = {k: z for k, z in z_scores.items() if z > z_thresh}
    return outliers, len(outliers) > 0


# --- Merge numeric + encoded text and run model ---
for ts in numeric_df.index:
    x = numeric_df.loc[ts].to_dict()

    # --- TF-IDF: handle text features ---
    if ts in text_df.index:
        for col in text_columns:
            vec = vectorizers[col]
            text = str(text_df.at[ts, col]) if ts in text_df.index else MISSING_TOKEN
            if not text or text.lower() in ["nan", "none", ""]:
                text = MISSING_TOKEN
            tfidf = vec.transform([text]).toarray()[0]
            for i, v in enumerate(tfidf):
                x[f"{col}_tfidf_{i}"] = v

    x = {k: v for k, v in x.items() if pd.notna(v)}

    try:
        # --- RIVER ---
        score = model.score_one(x)
        model.learn_one(x)
        print(f"[{ts}] 🤖 River Score: {score:.4f}")

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
        x_meta = {
            "river_score": score,
            "remand_outlier_count": len(remand_outliers) if remand_outliers else 0,
            "yesterday_outlier_count": len(y_outliers) if y_outliers else 0,
            **x,  # full state vector
        }

        # Track text changes in last 5 minutes
        prev_ts = ts - pd.Timedelta(CHUNK_SIZE)
        changed_entities = []

        for col in text_columns:
            now = str(text_df.at[ts, col]) if ts in text_df.index else MISSING_TOKEN
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
            x_meta[f"{col}_changed_5min"] = changed
            if changed:
                changed_entities.append(col)

        x_meta["changed_entities"] = changed_entities  # for logging

        is_meta_anomaly = meta_model.predict_one(x_meta)
        print(
            f"[{ts}] 🧠 Meta anomaly? → {is_meta_anomaly} | Changed: {changed_entities}"
        )

        # Optional: implicit label = any model flagged
        y = int(remand_flag or y_flag or score > 0.4)
        meta_model.learn_one(x_meta, y)

    except Exception as e:
        print(f"[{ts}] ⚠️ Skipped due to error: {e}")


# --- Save model and new cutoff ---
model_wrapper = ModelWrapper(model, stop_time)
with open(SAVE_MODEL_PATH, "wb") as f:
    pickle.dump(model_wrapper, f)

meta_model_wrapper = ModelWrapper(meta_model, stop_time)
with open(META_MODEL_PATH, "wb") as f:
    pickle.dump(meta_model_wrapper, f)


print("✅ Model updated and saved.")
