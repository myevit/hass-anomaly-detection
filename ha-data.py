# pip install influxdb-client holidays vowpalwabbit pandas numpy

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
import pytz

# --- CONFIG ---
INFLUX_URL = "http://192.168.1.4:8086"
TOKEN = "LY86Tqy1cg5-UYTYPMmHI5opIxC2_NtLiZexyHehiqmL7YLGyHOyEeosm9JXAnoVuNaZT5TYNNcMW1eQK3qW3g=="
ORG = "myeHome"
BUCKET = "home-assistant"
MODELS_DIR = "models"
ANOMALY_MODEL_PATH = os.path.join(MODELS_DIR, "vw_anomaly_model.vw")
META_MODEL_PATH = os.path.join(MODELS_DIR, "vw_meta_model.vw")
ANOMALY_HISTORY_PATH = os.path.join(MODELS_DIR, "anomaly_history.pkl")
CHUNK_SIZE = "5min"
DEFAULT_DAYS = 7  # Default number of days to look back
ANOMALY_THRESHOLD = 0.7  # Threshold for anomaly detection
HEURISTIC_ANOMALY_THRESHOLD = 0.0  # Threshold for testing anomaly detection if real anomaly score is 0. must be higher than ANOMALY_THRESHOLD
FORCE_RETRAIN = True  # Set to True to force model retraining

# --- Ensure model directory exists ---
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize last_cutoff and change_history
last_cutoff = datetime.now(timezone.utc) - timedelta(days=DEFAULT_DAYS)
change_history = []

# If force retrain is enabled, delete existing models
if FORCE_RETRAIN:
    print("Forcing model retraining - deleting existing models...")
    if os.path.exists(ANOMALY_MODEL_PATH):
        os.remove(ANOMALY_MODEL_PATH)
    if os.path.exists(META_MODEL_PATH):
        os.remove(META_MODEL_PATH)
    if os.path.exists(ANOMALY_HISTORY_PATH):
        os.remove(ANOMALY_HISTORY_PATH)
    # Reset last_cutoff to ensure we get enough training data
    print(f"Resetting time range to last {DEFAULT_DAYS} days for retraining")
    # last_cutoff is already initialized above
else:
    # Load existing model info if available
    if os.path.exists(ANOMALY_HISTORY_PATH):
        try:
            with open(ANOMALY_HISTORY_PATH, "rb") as f:
                anomaly_history = pickle.load(f)
                last_cutoff = anomaly_history.last_cutoff
                change_history = anomaly_history.change_history
        except Exception as e:
            print(f"Error loading anomaly history: {e}")


# --- Model Info Class ---
class ModelInfo:
    def __init__(self, last_cutoff, change_history=None):
        self.last_cutoff = last_cutoff
        self.change_history = change_history or []


# Function definitions moved here
def analyze_change_history(change_history, now):
    """Analyze change history to detect patterns and frequencies"""
    if not change_history:
        return {
            "entity_frequencies": {},
            "hourly_patterns": {},
            "daily_patterns": {},
            "recommended_threshold": ANOMALY_THRESHOLD,
            "recurring_sequences": [],
        }

    # 1. Entity frequency analysis
    entity_frequencies = {}
    for entry in change_history:
        if "changes" not in entry:
            continue
        for entity in entry["changes"].keys():
            entity_frequencies[entity] = entity_frequencies.get(entity, 0) + 1

    # Sort by frequency
    sorted_entities = sorted(
        entity_frequencies.items(), key=lambda x: x[1], reverse=True
    )

    # 2. Temporal patterns (hourly and daily)
    hourly_patterns = {hour: 0 for hour in range(24)}
    daily_patterns = {day: 0 for day in range(7)}

    for entry in change_history:
        if "timestamp" not in entry:
            continue
        ts = entry["timestamp"]
        hourly_patterns[ts.hour] += 1
        daily_patterns[ts.dayofweek] += 1

    # 3. Calculate recommended threshold based on historical data
    scores = [entry.get("score", 0) for entry in change_history if "score" in entry]
    if scores:
        # Use 75th percentile of historical scores as recommended threshold
        scores.sort()
        percentile_75 = scores[int(len(scores) * 0.75)]
        recommended_threshold = round(min(max(0.3, percentile_75), 0.9), 2)
    else:
        recommended_threshold = ANOMALY_THRESHOLD

    # 4. Find recurring sequences (simplified pattern detection)
    recurring_sequences = []
    if len(change_history) > 5:
        # Look at last 30 days of changes
        cutoff = now - timedelta(days=30)
        recent_changes = [
            entry
            for entry in change_history
            if "timestamp" in entry and entry["timestamp"] > cutoff
        ]

        # Extract entities that changed together frequently
        entity_groups = {}
        for entry in recent_changes:
            if "changes" not in entry:
                continue

            # Create a frozen set of changed entities for this timestamp
            changed_entities = frozenset(entry["changes"].keys())
            if len(changed_entities) > 1:
                entity_groups[changed_entities] = (
                    entity_groups.get(changed_entities, 0) + 1
                )

        # Find groups that occur more than once
        for group, count in entity_groups.items():
            if count > 1 and len(group) > 1:
                recurring_sequences.append({"entities": list(group), "count": count})

        # Sort by frequency
        recurring_sequences.sort(key=lambda x: x["count"], reverse=True)
        # Keep top 5
        recurring_sequences = recurring_sequences[:5]

    return {
        "entity_frequencies": dict(sorted_entities[:10]),  # Top 10 entities
        "hourly_patterns": hourly_patterns,
        "daily_patterns": daily_patterns,
        "recommended_threshold": recommended_threshold,
        "recurring_sequences": recurring_sequences,
    }


def print_change_history_analysis(analysis):
    """Print the analysis of change history in a readable format"""
    print("\n=== ANOMALY HISTORY ANALYSIS ===")

    # 1. Frequently anomalous entities
    print("\n🔄 TOP ANOMALOUS ENTITIES:")
    if analysis["entity_frequencies"]:
        for entity, count in analysis["entity_frequencies"].items():
            print(f"  • {entity}: {count} anomalies")
    else:
        print("  No entity frequency data available")

    # 2. Time patterns
    print("\n⏰ TEMPORAL PATTERNS:")
    # Find peak hours
    hourly = analysis["hourly_patterns"]
    if sum(hourly.values()) > 0:
        max_hourly = max(hourly.values())
        peak_hours = [f"{h}:00" for h, v in hourly.items() if v > max_hourly * 0.7]
        print(f"  • Peak anomaly hours: {', '.join(peak_hours)}")

        # Find peak days
        daily = analysis["daily_patterns"]
        max_daily = max(daily.values())
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        peak_days = [days[d] for d, v in daily.items() if v > max_daily * 0.7]
        print(f"  • Peak anomaly days: {', '.join(peak_days)}")
    else:
        print("  No temporal pattern data available")

    # 3. Threshold recommendation
    current = ANOMALY_THRESHOLD
    recommended = analysis["recommended_threshold"]
    if current != recommended:
        print(f"\n🎯 THRESHOLD RECOMMENDATION:")
        if recommended > current:
            print(f"  Current threshold ({current}) may be too low")
            print(
                f"  Recommended threshold: {recommended} (would reduce false positives)"
            )
        else:
            print(f"  Current threshold ({current}) may be too high")
            print(
                f"  Recommended threshold: {recommended} (would catch more anomalies)"
            )

    # 4. Recurring sequences
    sequences = analysis["recurring_sequences"]
    if sequences:
        print("\n🔁 RECURRING ANOMALY PATTERNS:")
        for seq in sequences:
            entities = seq["entities"]
            if len(entities) > 3:
                entity_str = f"{', '.join(entities[:2])} and {len(entities)-2} more"
            else:
                entity_str = ", ".join(entities)
            print(f"  • {entity_str} changed together {seq['count']} times")

    print("\n================================\n")


# --- InfluxDB connection ---
client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
query_api = client.query_api()

# Suppress InfluxDB warnings
warnings.simplefilter("ignore", MissingPivotFunction)


# --- Helper Functions ---
def clean_value(value):
    """Clean a value to prevent newlines and other problematic characters"""
    if isinstance(value, str):
        # Replace newlines, tabs, etc with spaces
        value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        # Replace multiple spaces with a single space
        while "  " in value:
            value = value.replace("  ", " ")
        # Remove other problematic characters
        value = value.replace("|", "_").replace(":", "_")
    return value


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
    example_parts = []

    # Add label if provided (for anomaly detection, we use quantile regression)
    if label is not None:
        example_parts.append(f"{label}")

    # Add tag if provided (for tracking examples)
    if tag:
        example_parts.append(f"'{tag}")

    # Store namespace strings
    ns_strings = []

    # Add numeric features namespace
    if numeric_features:
        num_parts = []
        for name, value in numeric_features.items():
            if pd.notna(value):
                safe_name = clean_value(str(name))
                num_parts.append(f"{safe_name}:{value}")
        if num_parts:
            ns_strings.append("|num " + " ".join(num_parts))

    # Add text features namespace - let VW handle tokenization
    if text_features:
        txt_parts = []
        for name, value in text_features.items():
            if value and pd.notna(value) and value.lower() not in ["nan", "none", ""]:
                safe_name = clean_value(str(name))
                safe_value = clean_value(str(value))
                txt_parts.append(f"{safe_name}_{safe_value}")
        if txt_parts:
            ns_strings.append("|txt " + " ".join(txt_parts))

    # Add time context namespace
    if context_features:
        ctx_parts = []
        for name, value in context_features.items():
            if pd.notna(value):
                safe_name = clean_value(str(name))
                ctx_parts.append(f"{safe_name}:{value}")
        if ctx_parts:
            ns_strings.append("|ctx " + " ".join(ctx_parts))

    # Combine namespace strings
    if ns_strings:
        example_parts.extend(ns_strings)

    # Join all parts with spaces
    return " ".join(example_parts)


def extract_timestamp_features(timestamp):
    """Extract time-based features from timestamp"""
    ca_holidays = holidays.CA(prov="AB")  # Customize based on your location

    features = {
        "hour": timestamp.hour,
        "hour_quarter": timestamp.minute
        // 15,  # Adds 0, 1, 2, or 3 for quarter of hour
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
                    print(f"Error calculating change for {col}: {e}")

    return changes


def format_timestamp(ts):
    """Convert timestamp to human-readable local time format"""

    # Handle timezone-naive timestamps
    if ts.tzinfo is None:
        # Assume UTC if no timezone info
        ts = ts.replace(tzinfo=timezone.utc)

    # Convert to local timezone
    local_timezone = datetime.now().astimezone().tzinfo
    local_ts = ts.astimezone(local_timezone)

    # Format: "dd.mm.yyyy hh:mm:ss"
    return local_ts.strftime("%d.%m.%Y %H:%M:%S")


# --- Load or create models ---
def create_anomaly_model(save_path=None):
    """Create a VW model for anomaly detection using quantile regression"""
    args = "--quiet --loss_function quantile --quantile_tau 0.9 --ngram txt:3 --skips txt:1 --bit_precision 28 --adaptive --normalized"
    if save_path:
        args += f" -f {save_path} --save_resume"
    return vowpalwabbit.Workspace(args)


def create_meta_model(save_path=None):
    """Create a VW model for meta decisions using binary classification"""
    args = "--quiet --loss_function logistic --binary --ngram txt:2 --bit_precision 28"
    if save_path:
        args += f" -f {save_path} --save_resume"
    return vowpalwabbit.Workspace(args)


# Create or load anomaly model
if os.path.exists(ANOMALY_MODEL_PATH):
    try:
        anomaly_model = vowpalwabbit.Workspace(f"--quiet -i {ANOMALY_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading anomaly model: {e}")
        anomaly_model = create_anomaly_model()
else:
    anomaly_model = create_anomaly_model()

# Create or load meta model
if os.path.exists(META_MODEL_PATH):
    try:
        meta_model = vowpalwabbit.Workspace(f"--quiet -i {META_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading meta model: {e}")
        meta_model = create_meta_model()
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
        print(
            f"Adjusted start time to latest InfluxDB record: {format_timestamp(last_cutoff)}"
        )

# Set up query time range
now = datetime.now(timezone.utc)
start = last_cutoff.replace(microsecond=0).isoformat()
stop_time = now.replace(microsecond=0)
stop = stop_time.isoformat()
print(f"Querying from {format_timestamp(last_cutoff)} to {format_timestamp(now)}...")

# After setting up query time range
time_difference = now - last_cutoff
if time_difference.total_seconds() < 300:  # 300 seconds = 5 minutes
    print(
        f"Only {round(time_difference.total_seconds())} seconds since last update. Need at least 300 seconds (5 minutes) for a complete chunk."
    )
    print("Skipping model update to avoid processing incomplete chunks.")
    exit()

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
    print("No new data found.")
    exit()

print(f"Query returned {len(df)} rows")

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

        # Debug info - uncomment if needed
        # print(f"VW Example: {vw_example[:100]}...")

        # Get anomaly score from model (higher = more anomalous)
        try:
            anomaly_score = anomaly_model.predict(vw_example)
        except Exception as e:
            print(f"Error predicting anomaly: {e}, example length: {len(vw_example)}")
            # Try with a simpler example if the previous one failed
            simple_example = f"1 |num const:1"
            anomaly_score = anomaly_model.predict(simple_example)

        # Normalize score to 0-1 range
        normalized_score = min(1.0, max(0.0, anomaly_score))

        # If all scores are zero, use a simple heuristic based on entity changes
        if normalized_score == 0 and i > 0 and entity_changes:
            # Calculate a simple anomaly score based on number of changes
            change_count = len(entity_changes)
            # More changes = higher score, with a cap at HEURISTIC_ANOMALY_THRESHOLD
            heuristic_score = min(HEURISTIC_ANOMALY_THRESHOLD, change_count / 10)
            print(
                f"[{format_timestamp(ts)}] Using heuristic score: {heuristic_score:.4f} (based on {change_count} changes)"
            )
            normalized_score = heuristic_score

        # Debug: Print all scores to see what values we're getting
        print(
            f"[{format_timestamp(ts)}] Anomaly score: {normalized_score:.4f} (threshold: {ANOMALY_THRESHOLD})"
        )

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
            print(
                f"[{format_timestamp(ts)}] Anomaly detected! Score: {normalized_score:.4f}"
            )

            # Print entity changes if any
            if entity_changes:
                print(f"[{format_timestamp(ts)}] Changed entities:")
                for entity, change in entity_changes.items():
                    change_str = f"'{change['previous']}' → '{change['current']}'"
                    if "diff" in change:
                        change_str += f" (diff: {change['diff']:.4f})"
                    print(f"  - {entity}: {change_str}")

        # Run meta model to verify anomaly
        if entity_changes:
            # Format meta example as a single string
            meta_features_str = f"|m score:{normalized_score} change_count:{len(entity_changes)} hour:{context_features['hour']} day:{context_features['day_of_week']} weekend:{context_features['is_weekend']} |c"

            # Add changed entities names
            for entity in entity_changes:
                # Clean entity name to avoid parsing issues
                safe_entity = clean_value(str(entity))
                meta_features_str += f" {safe_entity}"

            # Predict with meta model
            try:
                meta_score = meta_model.predict(meta_features_str)
                is_meta_anomaly = meta_score > 0.5

                if is_meta_anomaly:
                    print(f"[{format_timestamp(ts)}] Meta model confirms anomaly")

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
            except Exception as e:
                print(f"Error with meta model: {e}")

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

        try:
            anomaly_model.learn(train_example)
        except Exception as e:
            print(f"Error training anomaly model: {e}")

        # Train the meta model if we have entity changes
        # Label with 1 if this is a true anomaly, -1 otherwise
        if entity_changes and i > 0:
            # Use the primary anomaly detection as a label for meta model
            meta_label = 1 if is_anomaly else -1
            meta_train_example = f"{meta_label} {meta_features_str}"

            try:
                meta_model.learn(meta_train_example)
            except Exception as e:
                print(f"Error training meta model: {e}")

    except Exception as e:
        print(f"[{format_timestamp(ts)}] Error processing data: {e}")
        import traceback

        traceback.print_exc()

# --- Save models and changes ---
# Keep track of all entity changes without limitation
if all_entity_changes:
    change_history.extend(all_entity_changes)
    # Limitation removed: now storing entire change history

# Analyze change history for patterns and insights
if change_history:
    analysis_results = analyze_change_history(change_history, stop_time)
    print_change_history_analysis(analysis_results)

    # Optional: Auto-tune threshold based on history
    AUTO_TUNE_THRESHOLD = False  # Set to True to enable auto-tuning
    if AUTO_TUNE_THRESHOLD:
        recommended = analysis_results["recommended_threshold"]
        if (
            abs(ANOMALY_THRESHOLD - recommended) > 0.1
        ):  # Only change if difference is significant
            print(f"Auto-tuning threshold: {ANOMALY_THRESHOLD} → {recommended}")
            ANOMALY_THRESHOLD = recommended

# Print model statistics to verify training has occurred
print(
    f"Anomaly model total examples processed: {round(anomaly_model.get_weighted_examples())}"
)
print(f"Anomaly model total loss: {round(anomaly_model.get_sum_loss())}")
print(
    f"Meta model total examples processed: {round(meta_model.get_weighted_examples())}"
)
print(f"Meta model total loss: {round(meta_model.get_sum_loss())}")
print(f"Anomaly history: {len(change_history)} recorded changes")

# Save models to files
try:
    # Save the trained anomaly model
    anomaly_model.save(ANOMALY_MODEL_PATH)
    print(f"Anomaly model saved to {ANOMALY_MODEL_PATH}")
except Exception as e:
    print(f"Error saving anomaly model: {e}")

try:
    # Save the trained meta model
    meta_model.save(META_MODEL_PATH)
    print(f"Meta model saved to {META_MODEL_PATH}")
except Exception as e:
    print(f"Error saving meta model: {e}")

# Save model info separately (without the model objects)
anomaly_history = ModelInfo(stop_time, change_history)
try:
    with open(ANOMALY_HISTORY_PATH, "wb") as f:
        pickle.dump(anomaly_history, f)
    print(f"Anomaly history saved to {ANOMALY_HISTORY_PATH}")
except Exception as e:
    print(f"Error saving anomaly history: {e}")

print("Models updated and saved.")
