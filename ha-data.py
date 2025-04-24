# ======================================================
# HOME ASSISTANT ANOMALY DETECTION SYSTEM
# ======================================================
# This script analyzes Home Assistant time-series data to detect anomalies
# in smart home device behavior using machine learning techniques.
#
# The system uses Vowpal Wabbit for anomaly detection using quantile regression
#
# Required packages:
# pip install influxdb-client holidays vowpalwabbit pandas numpy

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient
import vowpalwabbit
import holidays
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

# Import config from config.py
try:
    import config

    # Use imported config values
    INFLUX_URL = config.INFLUX_URL
    TOKEN = config.TOKEN
    ORG = config.ORG
    BUCKET = config.BUCKET
    MODELS_DIR = config.MODELS_DIR
    ANOMALY_MODEL_PATH = config.ANOMALY_MODEL_PATH
    CHUNK_SIZE = config.CHUNK_SIZE
    DEFAULT_DAYS = config.DEFAULT_DAYS
    ANOMALY_THRESHOLD = config.ANOMALY_THRESHOLD
    FORCE_RETRAIN = config.FORCE_RETRAIN
    print("Loaded configuration from config.py")
except ImportError:
    print("No config.py found. Please create a config file with required parameters.")
    exit(1)

print("Script starting...")

# === INITIALIZATION ===
# --- Directory Setup ---
# Ensure model directory exists.
os.makedirs(MODELS_DIR, exist_ok=True)

# --- State Initialization ---
# Initialize last data cutoff point
last_cutoff = datetime.now(timezone.utc) - timedelta(days=DEFAULT_DAYS)

# --- Force Retrain Logic ---
# If force retrain is enabled, delete existing models and reset time range.
if FORCE_RETRAIN:
    print("Forcing model retraining - deleting existing models...")
    if os.path.exists(ANOMALY_MODEL_PATH):
        os.remove(ANOMALY_MODEL_PATH)
    # Reset last_cutoff to ensure we get enough training data.
    print(f"Resetting time range to last {DEFAULT_DAYS} days for retraining")
    # last_cutoff is already initialized above.


# === UTILITY FUNCTIONS ===
def clean_value(value):
    """
    Clean and normalize a value for VW formatting.
    Handles nulls, empty strings, and formatting in one place.
    """
    # Handle null/NA values first
    if (
        pd.isna(value)
        or value == ""
        or (isinstance(value, str) and value.lower() in ["nan", "none"])
    ):
        return "NA"

    # For string values, handle problematic characters
    if isinstance(value, str):
        # Replace newlines, tabs with spaces
        value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        # Replace multiple spaces with a single space
        while "  " in value:
            value = value.replace("  ", " ")
        # Replace problematic characters for VW format
        value = value.replace("|", "_").replace(":", "_").replace(" ", "_")

    return str(value).strip()


def is_valid_value(value):
    """Simplified validity check that works with clean_value"""
    return clean_value(value) != "NA"


def format_vw_example(
    timestamp,
    numeric_features,
    text_features=None,
    context_features=None,
    binary_transitions=None,
    binary_durations=None,
    label=None,
    tag=None,
):
    """
    Format data as a Vowpal Wabbit example string using namespaces.
    Optimized for efficiency with fewer redundant operations.
    """
    # Initialize parts with the label (default to 1 if not provided)
    parts = [str(label) if label is not None else "1"]

    # Add tag (use timestamp if not provided)
    if tag:
        parts.append(f"'{tag}")
    elif timestamp is not None:
        parts.append(f"'{timestamp.isoformat()}")

    # Define namespaces and their features in a dictionary for consistent processing
    namespaces = {
        "num": numeric_features or {},
        "txt": text_features or {},
        "bin": {},  # Will populate from binary_transitions
        "dur": {},  # Will populate from binary_durations
        "ctx": context_features or {},
    }

    # Pre-process binary transitions for consistent format
    if binary_transitions:
        for entity, count in binary_transitions.items():
            if count > 0:  # Only include entities with transitions
                safe_name = clean_value(entity)
                namespaces["bin"][f"trs_{safe_name}"] = count

    # Pre-process binary durations for consistent format
    if binary_durations:
        for entity, states in binary_durations.items():
            for state, duration in states.items():
                safe_entity = clean_value(entity)
                safe_state = clean_value(state)
                namespaces["dur"][f"dur_{safe_entity}_{safe_state}"] = duration

    # Process each namespace consistently
    for ns, features in namespaces.items():
        if not features:
            continue

        ns_parts = []
        for name, value in features.items():
            # Use clean_value for all strings, but preserve numeric values
            if isinstance(value, (int, float)):
                if value != 0:  # Skip zero values as they don't add information
                    ns_parts.append(f"{clean_value(name)}:{value}")
            elif is_valid_value(value):
                ns_parts.append(f"{clean_value(name)}_{clean_value(value)}")

        if ns_parts:
            parts.append(f"|{ns} {' '.join(ns_parts)}")

    return " ".join(parts)


def extract_timestamp_features(timestamp):
    """
    Extract time-based features from timestamp for contextual awareness.

    Creates features related to:
    - Hour of day (0-23)
    - Quarter of hour (0-3)
    - Day of week (0-6)
    - Weekend flag (0-1)
    - Month (1-12)
    - Season (0-3)
    - Holiday flag (0-1)
    - Time of day period (0-3: night, morning, afternoon, evening)

    Args:
        timestamp: Datetime object.

    Returns:
        dict: Dictionary of time-based features.
    """
    ca_holidays = holidays.CA(prov="AB")  # Customize based on your location.

    # Determine time of day period
    hour = timestamp.hour
    if 5 <= hour < 12:
        time_period = 1  # Morning (5am to 11:59am)
    elif 12 <= hour < 17:
        time_period = 2  # Afternoon (12pm to 4:59pm)
    elif 17 <= hour < 22:
        time_period = 3  # Evening (5pm to 9:59pm)
    else:
        time_period = 0  # Night (10pm to 4:59am)

    features = {
        "hour": hour,
        "hour_quarter": timestamp.minute // 15,  # 0, 1, 2, or 3
        "day_of_week": timestamp.dayofweek,
        "is_weekend": int(timestamp.dayofweek >= 5),
        "month": timestamp.month,
        "time_period": time_period,  # 0=night, 1=morning, 2=afternoon, 3=evening
        "is_night": int(time_period == 0),
        "is_morning": int(time_period == 1),
        "is_afternoon": int(time_period == 2),
        "is_evening": int(time_period == 3),
        "season": {
            12: 0,  # Winter: Dec-Feb
            1: 0,
            2: 0,
            3: 1,  # Spring: Mar-May
            4: 1,
            5: 1,
            6: 2,  # Summer: Jun-Aug
            7: 2,
            8: 2,
            9: 3,  # Fall: Sep-Nov
            10: 3,
            11: 3,
        }[timestamp.month],
        "is_holiday": int(timestamp.normalize() in ca_holidays),
    }
    return features


def track_changes(current_data, previous_data, entity_cols):
    """
    Track changes between two consecutive data points with optimized processing.
    """
    if not previous_data:
        return {}

    changes = {}

    # Process only entities that exist in both datasets and have changed
    common_cols = set(current_data) & set(previous_data) & entity_cols

    for col in common_cols:
        current_val = current_data[col]
        previous_val = previous_data[col]

        # Skip identical values
        if current_val == previous_val:
            continue

        # Skip invalid values (using our optimized is_valid_value)
        if not (is_valid_value(current_val) and is_valid_value(previous_val)):
            continue

        try:
            # Handle numeric values - add difference calculation
            if isinstance(current_val, (int, float)) and isinstance(
                previous_val, (int, float)
            ):
                changes[col] = {
                    "previous": previous_val,
                    "current": current_val,
                    "diff": current_val - previous_val,
                }
            else:
                # Text values - convert both to string using clean_value for consistency
                changes[col] = {
                    "previous": clean_value(previous_val),
                    "current": clean_value(current_val),
                }
        except Exception as e:
            print(f"Error calculating change for {col}: {e}")

    return changes


def format_timestamp(ts):
    """
    Convert timestamp to human-readable local time format.

    Handles timezone conversion and formatting:
    1. Ensures timestamp has timezone info
    2. Converts to local timezone
    3. Formats as day.month.year hour:minute:second

    Args:
        ts: Datetime object.

    Returns:
        str: Formatted timestamp string.
    """
    # Handle timezone-naive timestamps.
    if ts.tzinfo is None:
        # Assume UTC if no timezone info.
        ts = ts.replace(tzinfo=timezone.utc)

    # Convert to local timezone.
    local_timezone = datetime.now().astimezone().tzinfo
    local_ts = ts.astimezone(local_timezone)

    # Format: "dd.mm.yyyy hh:mm:ss".
    return local_ts.strftime("%d.%m.%Y %H:%M:%S")


# === MODEL FUNCTIONS ===
def create_anomaly_model(save_path=None):
    """
    Create a VW model for anomaly detection using quantile regression.

    Model settings:
    - Quantile regression with tau=0.9 (focuses on upper distribution tail)
    - N-gram processing for text features
    - Adaptive learning rate
    - Feature normalization

    Args:
        save_path: Optional path to save model.

    Returns:
        VW model workspace.
    """
    args = "--loss_function quantile --quantile_tau 0.9 --ngram txt:3 --skips txt:1 --bit_precision 28 --adaptive --normalized"
    if save_path:
        args += f" -f {save_path}"
    return vowpalwabbit.Workspace(args)


# === DATABASE CONNECTION ===
# Connect to InfluxDB and set up query client.
client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
query_api = client.query_api()

# Suppress InfluxDB warnings about pivot functions.
warnings.simplefilter("ignore", MissingPivotFunction)

# === MODEL INITIALIZATION ===
# --- Create or Load Anomaly Model ---
if os.path.exists(ANOMALY_MODEL_PATH):
    try:
        # Load existing model.
        anomaly_model = vowpalwabbit.Workspace(f"-i {ANOMALY_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading anomaly model: {e}")
        # Fall back to creating a new model.
        anomaly_model = create_anomaly_model(ANOMALY_MODEL_PATH)
else:
    # Create new model.
    anomaly_model = create_anomaly_model(ANOMALY_MODEL_PATH)

# === DATA RETRIEVAL ===
# --- Find Latest Data ---
# Query to find the most recent data in InfluxDB.
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

# Adjust start time if needed based on latest data.
if not latest_df.empty:
    latest_time = pd.to_datetime(latest_df["_time"].iloc[0])
    if latest_time > last_cutoff:
        last_cutoff = latest_time
        print(
            f"Adjusted start time to latest InfluxDB record: {format_timestamp(last_cutoff)}"
        )

# --- Query Time Setup ---
# Set up query time range from last cutoff to now.
now = datetime.now(timezone.utc)
start = last_cutoff.replace(microsecond=0).isoformat()
stop_time = now.replace(microsecond=0)
stop = stop_time.isoformat()
print(f"Querying from {format_timestamp(last_cutoff)} to {format_timestamp(now)}...")

# --- Check for Minimum Time Difference ---
# Ensure at least 5 minutes of data to process.
time_difference = now - last_cutoff
if time_difference.total_seconds() < 300:  # 300 seconds = 5 minutes.
    print(
        f"Only {round(time_difference.total_seconds())} seconds since last update. Need at least 300 seconds (5 minutes) for a complete chunk."
    )
    print("Skipping model update to avoid processing incomplete chunks.")
    exit()

# --- InfluxDB Query ---
# Query for all state and value data in the time range.
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

# Execute query.
df = query_api.query_data_frame(query)
df = pd.concat(df) if isinstance(df, list) else df
if df.empty:
    print("No new data found.")
    exit()

print(f"Query returned {len(df)} rows")

# === DATA PROCESSING ===
# --- Time Series Grouping ---
# Convert time column to datetime and group by time and field type.
df["_time"] = pd.to_datetime(df["_time"])

# Sort data by time for proper sequence analysis
df = df.sort_values("_time")

# Create a dictionary to store all data points, indexed by timestamp.
all_data = {}
all_entities = set()

# Track binary state transitions and durations
binary_transitions = {}  # Count of state changes per entity within window
binary_durations = {}  # Duration in seconds per state for binary entities
previous_states = {}  # Track previous state for detecting transitions
previous_times = {}  # Track previous time for duration calculation

# Identify initial state for each entity
for _, row in df.iterrows():
    entity_id = row["entity_id"]
    field_type = row["_field"]

    # Only consider state fields for binary state tracking
    if field_type == "state":
        value = str(row["_value"])
        time = row["_time"]

        # Initialize if we haven't seen this entity before
        if entity_id not in previous_states:
            previous_states[entity_id] = value
            previous_times[entity_id] = time
            # Initialize transition counter
            binary_transitions[entity_id] = 0

# Create groups by time chunk
df_grouped = df.set_index("_time").groupby([pd.Grouper(freq=CHUNK_SIZE), "_field"])

# --- Data Organization ---
# Process each time chunk and organize by field type.
for (timestamp, field_type), group in df_grouped:
    if timestamp not in all_data:
        all_data[timestamp] = {
            "numeric": {},
            "text": {},
            "binary_transitions": {},
            "binary_durations": {},
        }

    # Store data by field type.
    if field_type == "value":
        # Process numeric values - use DataFrame operations instead of loops where possible
        entity_groups = group.groupby("entity_id")

        for entity_id, entity_data in entity_groups:
            all_entities.add(entity_id)
            values = entity_data["_value"].astype(float).values
            timestamps = entity_data.index.to_list()

            if len(values) > 1:
                # Calculate metrics in one pass
                min_value = min(values)
                max_value = max(values)
                value_range = max_value - min_value

                # Calculate time difference in seconds
                time_diff = (max(timestamps) - min(timestamps)).total_seconds()
                rate_of_change = value_range / time_diff if time_diff > 0 else 0.0

                # Store all metrics in a single update operation
                all_data[timestamp]["numeric"].update(
                    {
                        f"{entity_id}_min": min_value,
                        f"{entity_id}_max": max_value,
                        f"{entity_id}_range": value_range,
                        f"{entity_id}_roc": rate_of_change,
                    }
                )
            elif len(values) == 1:
                # Just one value - store with zero change metrics in a single update
                all_data[timestamp]["numeric"].update(
                    {
                        f"{entity_id}_min": values[0],
                        f"{entity_id}_max": values[0],
                        f"{entity_id}_range": 0.0,
                        f"{entity_id}_roc": 0.0,
                    }
                )

    elif field_type == "state":
        # Process text states.
        for _, row in group.iterrows():
            entity_id = row["entity_id"]
            value = str(row["_value"])
            current_time = row.name  # Use index which is the timestamp

            # Store the text state
            all_data[timestamp]["text"][entity_id] = value
            all_entities.add(entity_id)

            # If we have previous state information, we can calculate transitions and durations
            if entity_id in previous_states:
                prev_value = previous_states[entity_id]
                prev_time = previous_times[entity_id]

                # Check if this is a state transition
                if value != prev_value:
                    # Increment transition counter
                    if entity_id in binary_transitions:
                        binary_transitions[entity_id] += 1
                    else:
                        binary_transitions[entity_id] = 1

                # Calculate duration in previous state (in seconds)
                duration = (current_time - prev_time).total_seconds()

                # Update duration counters
                if entity_id not in binary_durations:
                    binary_durations[entity_id] = {}

                if prev_value not in binary_durations[entity_id]:
                    binary_durations[entity_id][prev_value] = duration
                else:
                    binary_durations[entity_id][prev_value] += duration

                # Update previous state and time
                previous_states[entity_id] = value
                previous_times[entity_id] = current_time

        # Store transitions and durations for this time window
        # Copy the counts to avoid sharing the reference
        all_data[timestamp]["binary_transitions"] = binary_transitions.copy()
        all_data[timestamp]["binary_durations"] = binary_durations.copy()

        # Reset transition counters for the next window
        binary_transitions = {entity_id: 0 for entity_id in binary_transitions}

# Sort timestamps for chronological processing.
timestamps = sorted(all_data.keys())

# === ANOMALY DETECTION ===
# --- Main Processing Loop ---
# Process each timestamp for anomaly detection.
for i, ts in enumerate(timestamps):
    try:
        # Get current data point.
        current_time_data = all_data[ts]

        # Add time context features for this timestamp.
        context_features = extract_timestamp_features(ts)

        # Format as VW example - no label needed for prediction.
        vw_example = format_vw_example(
            timestamp=ts,
            numeric_features=current_time_data["numeric"],
            text_features=current_time_data["text"],
            context_features=context_features,
            binary_transitions=current_time_data["binary_transitions"],
            binary_durations=current_time_data["binary_durations"],
        )

        # --- Primary Anomaly Detection ---
        # Get anomaly score from model (higher = more anomalous).
        try:
            anomaly_score = anomaly_model.predict(vw_example)
        except Exception as e:
            print(f"Error predicting anomaly: {e}, example length: {len(vw_example)}")
            # Try with a simpler example if the previous one failed.
            simple_example = f"1 |num const:1"
            anomaly_score = anomaly_model.predict(simple_example)

        # Normalize score to 0-1 range.
        normalized_score = min(1.0, max(0.0, anomaly_score))

        # --- Anomaly Evaluation ---
        # Check if this exceeds the anomaly threshold.
        is_anomaly = normalized_score > ANOMALY_THRESHOLD

        # Output anomaly information if detected.
        if is_anomaly:
            print(
                f"[{format_timestamp(ts)}] Anomaly detected! Score: {normalized_score:.4f}"
            )

            # --- Entity Change Tracking ---
            # Only track what changed if this is actually an anomaly
            entity_changes = {}
            if i > 0:
                prev_ts = timestamps[i - 1]
                previous_data = all_data[prev_ts]

                # Track numeric changes.
                entity_changes = track_changes(
                    current_time_data["numeric"], previous_data["numeric"], all_entities
                )

                # Add text changes.
                text_changes = track_changes(
                    current_time_data["text"], previous_data["text"], all_entities
                )
                entity_changes.update(text_changes)

                # Add binary state transition information to the changes
                for entity, count in current_time_data["binary_transitions"].items():
                    if count > 0 and entity in entity_changes:
                        entity_changes[entity]["transitions"] = count

                # Add duration information to the changes
                for entity, states in current_time_data["binary_durations"].items():
                    if entity in entity_changes:
                        entity_changes[entity]["durations"] = states

            # Print entity changes if any.
            if entity_changes:
                print(f"[{format_timestamp(ts)}] Changed entities:")
                for entity, change in entity_changes.items():
                    change_str = f"'{change['previous']}' → '{change['current']}'"
                    if "diff" in change:
                        change_str += f" (diff: {change['diff']:.4f})"
                    if "transitions" in change:
                        change_str += f" [transitions: {change['transitions']}]"
                    if "durations" in change:
                        durations_str = ", ".join(
                            [f"{s}: {d:.1f}s" for s, d in change["durations"].items()]
                        )
                        change_str += f" [durations: {durations_str}]"
                    print(f"  - {entity}: {change_str}")

                print(f"[{format_timestamp(ts)}] Anomaly details recorded")
                # No need to collect anomalies since we're already printing them

        # --- Model Training ---
        # Train the main model on this example - we always learn from all data.
        # For anomaly models using quantile regression, label=1 is standard.
        train_example = format_vw_example(
            timestamp=ts,
            numeric_features=current_time_data["numeric"],
            text_features=current_time_data["text"],
            context_features=context_features,
            binary_transitions=current_time_data["binary_transitions"],
            binary_durations=current_time_data["binary_durations"],
            label=1,  # For quantile regression.
        )

        try:
            anomaly_model.learn(train_example)
        except Exception as e:
            print(f"Error training anomaly model: {e}")

    except Exception as e:
        print(f"[{format_timestamp(ts)}] Error processing data: {e}")
        import traceback

        traceback.print_exc()

# === RESULTS PROCESSING ===
# --- Model Statistics ---
# Print model statistics to verify training has occurred.
print(
    f"Anomaly model total examples processed: {round(anomaly_model.get_weighted_examples())}"
)
print(f"Anomaly model total loss: {round(anomaly_model.get_sum_loss())}")

# === MODEL PERSISTENCE ===
# --- Save Anomaly Model ---
try:
    # Save the trained anomaly model.
    anomaly_model.save(ANOMALY_MODEL_PATH)
    print(f"Anomaly model saved to {ANOMALY_MODEL_PATH}")
except Exception as e:
    print(f"Error saving anomaly model: {e}")

print("Model updated and saved.")
