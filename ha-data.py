import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient
import vowpalwabbit
import pickle
import holidays
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

# --- CONFIG ---
INFLUX_URL = "http://192.168.1.4:8086"
TOKEN = "LY86Tqy1cg5-UYTYPMmHI5opIxC2_NtLiZexyHehiqmL7YLGyHOyEeosm9JXAnoVuNaZT5TYNNcMW1eQK3qW3g=="
ORG = "myeHome"
BUCKET = "home-assistant"
MODELS_DIR = "models"
ANOMALY_MODEL_PATH = os.path.join(MODELS_DIR, "vw_anomaly_model.pkl")
META_MODEL_PATH = os.path.join(MODELS_DIR, "vw_meta_model.pkl")
CHUNK_SIZE = "5min"
DEFAULT_DAYS = 7  # Default number of days to look back
ANOMALY_THRESHOLD = 0.7  # Threshold for anomaly detection

# --- Ensure model directory exists ---
os.makedirs(MODELS_DIR, exist_ok=True)


# --- Model Wrapper Class ---
class VWModelWrapper:
    def __init__(self, model, last_cutoff, change_history=None):
        self.model = model
        self.last_cutoff = last_cutoff
        self.change_history = change_history or []


# --- InfluxDB connection ---
client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
query_api = client.query_api()

# Suppress InfluxDB warnings
warnings.simplefilter("ignore", MissingPivotFunction)


# --- Helper Functions ---
def format_vw_example(
    timestamp,
    numeric_features,
    text_features=None,
    context_features=None,
    label=None,
    tag=None,
):
    """
    Format data as a VW example string using namespaces
    - timestamp: for tag and context features
    - numeric_features: dict of sensor numeric values
    - text_features: dict of sensor text states
    - context_features: dict of time context features (hour, day, etc.)
    - label: optional label for training
    - tag: optional example tag
    """
    example = ""

    # Add label if provided (for anomaly detection, we use quantile regression)
    if label is not None:
        example += f"{label} "

    # Add importance weight if needed
    # example += "1.0 "

    # Add tag if provided (for tracking examples)
    if tag:
        example += f"'{tag} "

    # Add numeric features namespace
    if numeric_features:
        example += "|num "
        for name, value in numeric_features.items():
            if pd.notna(value):
                safe_name = name.replace(":", "_").replace("|", "_")
                example += f"{safe_name}:{value} "

    # Add text features namespace - let VW handle tokenization
    if text_features:
        example += "|txt "
        for name, value in text_features.items():
            if value and pd.notna(value) and value.lower() not in ["nan", "none", ""]:
                safe_name = name.replace(":", "_").replace("|", "_")
                example += f"{safe_name}_{value} "

    # Add time context namespace
    if context_features:
        example += "|ctx "
        for name, value in context_features.items():
            if pd.notna(value):
                safe_name = name.replace(":", "_").replace("|", "_")
                example += f"{safe_name}:{value} "

    return example


def extract_timestamp_features(timestamp):
    """Extract time-based features from timestamp"""
    ca_holidays = holidays.CA(prov="AB")  # Customize based on your location

    features = {
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "is_weekend": int(timestamp.dayofweek >= 5),
        "month": timestamp.month,
        "season": {
            12: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 2,
            9: 3,
            10: 3,
            11: 3,
        }[timestamp.month],
        "is_holiday": int(timestamp.normalize() in ca_holidays),
    }
    return features


def track_changes(current_data, previous_data, entity_cols):
    """Track changes between two consecutive data points"""
    if previous_data is None:
        return {}

    changes = {}

    # Compare all columns
    for col in entity_cols:
        if col in current_data and col in previous_data:
            current_val = current_data[col]
            previous_val = previous_data[col]

            # Only include if values changed and not missing
            if (
                pd.notna(current_val)
                and pd.notna(previous_val)
                and current_val != previous_val
            ):
                try:
                    # For numeric values, calculate difference
                    if isinstance(current_val, (int, float)) and isinstance(
                        previous_val, (int, float)
                    ):
                        changes[col] = {
                            "previous": previous_val,
                            "current": current_val,
                            "diff": current_val - previous_val,
                        }
                    else:
                        # For text values
                        changes[col] = {
                            "previous": str(previous_val),
                            "current": str(current_val),
                        }
                except Exception as e:
                    print(f"⚠️ Error calculating change for {col}: {e}")

    return changes


# --- Load or create models ---
def create_anomaly_model():
    """Create a VW model for anomaly detection using quantile regression"""
    args = "--quiet --loss_function quantile --quantile_tau 0.95 --ngram txt:3 --skips txt:1 --bit_precision 28 --adaptive --normalized"
    return vowpalwabbit.Workspace(args)


def create_meta_model():
    """Create a VW model for meta decisions using binary classification"""
    args = "--quiet --loss_function logistic --binary --ngram txt:2 --bit_precision 28"
    return vowpalwabbit.Workspace(args)


# Load existing models or create new ones
if os.path.exists(ANOMALY_MODEL_PATH):
    with open(ANOMALY_MODEL_PATH, "rb") as f:
        anomaly_wrapper = pickle.load(f)
        anomaly_model = anomaly_wrapper.model
        last_cutoff = anomaly_wrapper.last_cutoff
        change_history = anomaly_wrapper.change_history
else:
    anomaly_model = create_anomaly_model()
    last_cutoff = datetime.now(timezone.utc) - timedelta(days=DEFAULT_DAYS)
    change_history = []

if os.path.exists(META_MODEL_PATH):
    with open(META_MODEL_PATH, "rb") as f:
        meta_wrapper = pickle.load(f)
        meta_model = meta_wrapper.model
else:
    meta_model = create_meta_model()

# --- Find the latest timestamp in InfluxDB ---
latest_query = f"""
from(bucket: "{BUCKET}")
  |> range(start: -{DEFAULT_DAYS}d)
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["_measurement"] == "state")
  |> first()
  |> pivot(rowKey: ["_time"], columnKey: ["entity_id"], valueColumn: "_value")
"""

latest_df = query_api.query_data_frame(latest_query)
latest_df = pd.concat(latest_df) if isinstance(latest_df, list) else latest_df

if not latest_df.empty:
    latest_time = pd.to_datetime(latest_df["_time"].iloc[0])
    if latest_time > last_cutoff:
        last_cutoff = latest_time
        print(f"📅 Adjusted start time to latest InfluxDB record: {last_cutoff}")

# Set up query time range
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

# Execute query
df = query_api.query_data_frame(query)
df = pd.concat(df) if isinstance(df, list) else df
if df.empty:
    print("⚠️ No new data found.")
    exit()

print(f"ℹ️ Query returned {len(df)} rows")

# --- Process the Data ---
# Group by time and field type
df["_time"] = pd.to_datetime(df["_time"])
df_grouped = df.set_index("_time").groupby([pd.Grouper(freq=CHUNK_SIZE), "_field"])

# Create a dictionary to store all data points, indexed by timestamp
all_data = {}
all_entities = set()

# Process each time chunk
for (timestamp, field_type), group in df_grouped:
    if timestamp not in all_data:
        all_data[timestamp] = {"numeric": {}, "text": {}}

    # Store data by field type
    if field_type == "value":
        # Process numeric values
        for _, row in group.iterrows():
            entity_id = row["entity_id"]
            value = float(row["_value"])
            all_data[timestamp]["numeric"][entity_id] = value
            all_entities.add(entity_id)

    elif field_type == "state":
        # Process text states
        for _, row in group.iterrows():
            entity_id = row["entity_id"]
            value = str(row["_value"])
            all_data[timestamp]["text"][entity_id] = value
            all_entities.add(entity_id)

# Sort timestamps
timestamps = sorted(all_data.keys())

# Storage for anomalies and tracking changes
anomalies = []
detected_changes = []
all_entity_changes = []
previous_data = None

# --- Process each timestamp ---
for i, ts in enumerate(timestamps):
    try:
        current_time_data = all_data[ts]

        # Add context features
        context_features = extract_timestamp_features(ts)

        # Format as VW example - no label needed for prediction
        vw_example = format_vw_example(
            timestamp=ts,
            numeric_features=current_time_data["numeric"],
            text_features=current_time_data["text"],
            context_features=context_features,
            tag=str(ts),
        )

        # Get anomaly score from model (higher = more anomalous)
        anomaly_score = anomaly_model.predict(vw_example)

        # Normalize score to 0-1 range
        normalized_score = min(1.0, max(0.0, anomaly_score))

        # Track entity changes since previous timestamp
        entity_changes = {}
        if i > 0:
            prev_ts = timestamps[i - 1]
            previous_data = all_data[prev_ts]

            # Track numeric changes
            entity_changes = track_changes(
                current_time_data["numeric"], previous_data["numeric"], all_entities
            )

            # Add text changes
            text_changes = track_changes(
                current_time_data["text"], previous_data["text"], all_entities
            )
            entity_changes.update(text_changes)

        # Check if this is an anomaly
        is_anomaly = normalized_score > ANOMALY_THRESHOLD

        # Print information if anomaly detected
        if is_anomaly:
            print(f"[{ts}] 🚨 Anomaly detected! Score: {normalized_score:.4f}")

            # Print entity changes if any
            if entity_changes:
                print(f"[{ts}] 📊 Changed entities:")
                for entity, change in entity_changes.items():
                    change_str = f"'{change['previous']}' → '{change['current']}'"
                    if "diff" in change:
                        change_str += f" (diff: {change['diff']:.4f})"
                    print(f"  - {entity}: {change_str}")

        # Run meta model to verify anomaly
        if entity_changes:
            # Create meta-model features
            meta_features = {
                "score": normalized_score,
                "change_count": len(entity_changes),
                "timestamp": ts,
            }

            # Add context features to meta model
            meta_features.update(context_features)

            # Format meta example
            meta_example = ""
            meta_example += "|m "
            for k, v in meta_features.items():
                if k != "timestamp":
                    meta_example += f"{k}:{v} "

            # Add changed entities info
            meta_example += "|c "
            for entity in entity_changes:
                meta_example += f"{entity} "

            # Predict with meta model
            meta_score = meta_model.predict(meta_example)
            is_meta_anomaly = meta_score > 0.5

            if is_meta_anomaly:
                print(f"[{ts}] 🧠 Meta model confirms anomaly")

                # Store change details for future reference
                all_entity_changes.append(
                    {
                        "timestamp": ts,
                        "score": normalized_score,
                        "meta_score": meta_score,
                        "changes": entity_changes,
                        "is_anomaly": is_anomaly or is_meta_anomaly,
                    }
                )

        # Train the main model on this example - we always learn
        # For anomaly models using quantile regression, label=1 is standard
        train_example = format_vw_example(
            timestamp=ts,
            numeric_features=current_time_data["numeric"],
            text_features=current_time_data["text"],
            context_features=context_features,
            label=1,  # For quantile regression
            tag=str(ts),
        )
        anomaly_model.learn(train_example)

        # Train the meta model if we have entity changes
        # Label with 1 if this is a true anomaly, -1 otherwise
        if entity_changes and i > 0:
            # Use the primary anomaly detection as a label for meta model
            meta_label = 1 if is_anomaly else -1

            meta_train_example = f"{meta_label} {meta_example}"
            meta_model.learn(meta_train_example)

    except Exception as e:
        print(f"[{ts}] ⚠️ Error processing data: {e}")
        import traceback

        traceback.print_exc()

# --- Save models and changes ---
# Keep only the last 100 changes to avoid memory issues
if all_entity_changes:
    change_history.extend(all_entity_changes)
    if len(change_history) > 100:
        change_history = change_history[-100:]

# Save anomaly model
anomaly_wrapper = VWModelWrapper(anomaly_model, stop_time, change_history)
with open(ANOMALY_MODEL_PATH, "wb") as f:
    pickle.dump(anomaly_wrapper, f)

# Save meta model
meta_wrapper = VWModelWrapper(meta_model, stop_time)
with open(META_MODEL_PATH, "wb") as f:
    pickle.dump(meta_wrapper, f)

print("✅ Models updated and saved.")
