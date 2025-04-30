# ======================================================
# HOME ASSISTANT ANOMALY DETECTION SYSTEM
# ======================================================
# This script analyzes Home Assistant time-series data to detect anomalies
# in smart home device behavior using machine learning techniques.
#
# The system uses Autoformer for anomaly detection in time series data
#
# Required packages:
# pip install influxdb-client holidays pandas numpy torch

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient
import torch
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

    # Autoformer specific configs
    SEQUENCE_LENGTH = config.SEQUENCE_LENGTH
    PREDICTION_LENGTH = config.PREDICTION_LENGTH
    TRAINING_EPOCHS = config.TRAINING_EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    LEARNING_RATE = config.LEARNING_RATE
    SIGMA_THRESHOLD = config.SIGMA_THRESHOLD

    print("Loaded configuration from config.py")
except ImportError:
    print("No config.py found. Please create a config file with required parameters.")
    exit(1)

# === DATABASE CONNECTION ===
# Connect to InfluxDB and set up query client.
client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
query_api = client.query_api()

# Suppress InfluxDB warnings about pivot functions.
warnings.simplefilter("ignore", MissingPivotFunction)

# Initialize anomaly model as a global variable (will be set in main)
anomaly_model = None

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
    Create an Autoformer model for anomaly detection.

    Args:
        save_path: Optional path to save model.

    Returns:
        Autoformer anomaly detector instance.
    """
    # We'll determine the exact feature count when we have the data
    # Default to 4 features (will be updated before training)
    return AnomalyDetector(
        model_path=save_path if os.path.exists(save_path or "") else None,
        seq_len=SEQUENCE_LENGTH,
        pred_len=PREDICTION_LENGTH,
        enc_in=4,  # This will be updated dynamically before training
        dec_in=4,  # This will be updated dynamically before training
        c_out=4,  # This will be updated dynamically before training
    )


def process_data_chunk(
    df,
    previous_states=None,
    previous_times=None,
    binary_transitions=None,
    binary_durations=None,
):
    """
    Process a chunk of data and extract features for anomaly detection.

    Args:
        df: DataFrame with the data chunk
        previous_states: Dictionary of previous states for each entity (can be None for first chunk)
        previous_times: Dictionary of previous timestamps for each entity (can be None for first chunk)
        binary_transitions: Dictionary of transition counts for each entity (can be None for first chunk)
        binary_durations: Dictionary of durations for each entity's states (can be None for first chunk)

    Returns:
        tuple: (all_data, all_entities, previous_states, previous_times, binary_transitions, binary_durations)
    """
    # Initialize state tracking if this is the first chunk
    if previous_states is None:
        previous_states = {}
    if previous_times is None:
        previous_times = {}
    if binary_transitions is None:
        binary_transitions = {}
    if binary_durations is None:
        binary_durations = {}

    # Create dictionaries to store processed data
    all_data = {}
    all_entities = set()

    # Ensure data is sorted by time
    df = df.sort_values("_time")
    df["_time"] = pd.to_datetime(df["_time"])

    # Identify initial state for each entity if not already tracked
    for _, row in df.iterrows():
        entity_id = row["entity_id"]
        field_type = row["_field"]

        # Only consider state fields for binary state tracking
        if field_type == "state" and entity_id not in previous_states:
            value = str(row["_value"])
            time = row["_time"]

            # Initialize tracking
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
            # Process numeric values
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

    return (
        all_data,
        all_entities,
        previous_states,
        previous_times,
        binary_transitions,
        binary_durations,
    )


def detect_anomalies(
    all_data,
    all_entities,
    anomaly_model,
    processed_data=None,
    feature_names=None,
    anomalies_detected=None,
):
    """
    Detect anomalies in the given data.

    Args:
        all_data: Dictionary of processed data
        all_entities: Set of all entities
        anomaly_model: AnomalyDetector model
        processed_data: List of previously processed data (for continuous detection)
        feature_names: List of feature names (for continuous detection)
        anomalies_detected: List of previously detected anomalies (for continuous detection)

    Returns:
        tuple: (processed_data, feature_names, anomalies_detected)
    """
    # Initialize outputs if not provided
    if processed_data is None:
        processed_data = []
    if feature_names is None:
        feature_names = []
    if anomalies_detected is None:
        anomalies_detected = []

    # Sort timestamps for chronological processing
    timestamps = sorted(all_data.keys())

    # First pass: collect all feature names across all timestamps
    all_possible_features = set()
    for ts in timestamps:
        current_time_data = all_data[ts]

        # Collect numeric feature names
        all_possible_features.update(current_time_data["numeric"].keys())

        # Collect binary transition feature names
        for entity in current_time_data["binary_transitions"].keys():
            if current_time_data["binary_transitions"][entity] > 0:
                all_possible_features.add(f"trans_{entity}")

    # Add context features
    if timestamps:
        context_features = extract_timestamp_features(timestamps[0])
        for name in context_features.keys():
            if isinstance(context_features[name], (int, float)):
                all_possible_features.add(f"ctx_{name}")

    # Convert to sorted list for consistent ordering
    all_feature_names = sorted(list(all_possible_features))
    print(f"Total feature count: {len(all_feature_names)}")

    # Second pass: process each timestamp with consistent features
    for i, ts in enumerate(timestamps):
        try:
            # Get current data point
            current_time_data = all_data[ts]

            # Add time context features for this timestamp
            context_features = extract_timestamp_features(ts)

            # Create feature vector with zeros for all possible features
            feature_vector = np.zeros(len(all_feature_names))
            feature_dict = {}

            # Add numeric features
            for name, value in current_time_data["numeric"].items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    if name in all_possible_features:
                        idx = all_feature_names.index(name)
                        feature_vector[idx] = float(value)
                        feature_dict[name] = float(value)

            # Add context features
            for name, value in context_features.items():
                feature_name = f"ctx_{name}"
                if (
                    isinstance(value, (int, float))
                    and not pd.isna(value)
                    and feature_name in all_possible_features
                ):
                    idx = all_feature_names.index(feature_name)
                    feature_vector[idx] = float(value)
                    feature_dict[feature_name] = float(value)

            # Convert binary transitions to numeric features
            for entity, count in current_time_data["binary_transitions"].items():
                feature_name = f"trans_{entity}"
                if count > 0 and feature_name in all_possible_features:
                    idx = all_feature_names.index(feature_name)
                    feature_vector[idx] = float(count)
                    feature_dict[feature_name] = float(count)

            # Store processed data
            processed_data.append(feature_vector)

            # Set feature names if not already set
            if not feature_names:
                feature_names = all_feature_names

            # Calculate global index (including previously processed data)
            global_index = len(processed_data) - 1

            # Evaluate with existing model (if we have enough data)
            if global_index >= SEQUENCE_LENGTH and anomaly_model is not None:
                try:
                    # Get sequences for prediction - ensure all vectors have consistent length
                    sequence_data = np.array(
                        processed_data[-SEQUENCE_LENGTH - PREDICTION_LENGTH :]
                    )

                    # Check for consistent shape
                    if len(sequence_data.shape) != 2:
                        print(
                            f"Warning: Inconsistent sequence shape: {sequence_data.shape}"
                        )
                        continue

                    # First time? Update model dimensions
                    if (
                        hasattr(anomaly_model, "enc_in")
                        and anomaly_model.enc_in != sequence_data.shape[1]
                    ):
                        print(
                            f"Updating model dimensions to match data: {sequence_data.shape[1]} features"
                        )
                        # Create new model with correct dimensions
                        n_features = sequence_data.shape[1]
                        anomaly_model = AnomalyDetector(
                            seq_len=SEQUENCE_LENGTH,
                            pred_len=PREDICTION_LENGTH,
                            enc_in=n_features,
                            dec_in=n_features,
                            c_out=n_features,
                        )

                    # Detect anomalies
                    is_anomaly, anomaly_score, normalized_score = (
                        anomaly_model.detect_anomaly(
                            sequence_data, threshold_sigmas=SIGMA_THRESHOLD
                        )
                    )

                    # Check if this exceeds the anomaly threshold
                    if normalized_score > ANOMALY_THRESHOLD:
                        print(
                            f"[{format_timestamp(ts)}] Anomaly detected! Score: {normalized_score:.4f}"
                        )

                        # Only track what changed if this is actually an anomaly
                        entity_changes = {}
                        if global_index > 0 and i > 0:
                            # If we're in the same chunk, use the previous timestamp
                            if i > 0:
                                prev_ts = timestamps[i - 1]
                                previous_data = all_data[prev_ts]

                                # Track numeric changes
                                entity_changes = track_changes(
                                    current_time_data["numeric"],
                                    previous_data["numeric"],
                                    all_entities,
                                )

                                # Add text changes
                                text_changes = track_changes(
                                    current_time_data["text"],
                                    previous_data["text"],
                                    all_entities,
                                )
                                entity_changes.update(text_changes)

                                # Add binary state transition information to the changes
                                for entity, count in current_time_data[
                                    "binary_transitions"
                                ].items():
                                    if count > 0 and entity in entity_changes:
                                        entity_changes[entity]["transitions"] = count

                                # Add duration information to the changes
                                for entity, states in current_time_data[
                                    "binary_durations"
                                ].items():
                                    if entity in entity_changes:
                                        entity_changes[entity]["durations"] = states

                        # Print entity changes if any
                        if entity_changes:
                            print(f"[{format_timestamp(ts)}] Changed entities:")
                            for entity, change in entity_changes.items():
                                change_str = (
                                    f"'{change['previous']}' → '{change['current']}'"
                                )
                                if "diff" in change:
                                    change_str += f" (diff: {change['diff']:.4f})"
                                if "transitions" in change:
                                    change_str += (
                                        f" [transitions: {change['transitions']}]"
                                    )
                                if "durations" in change:
                                    durations_str = ", ".join(
                                        [
                                            f"{s}: {d:.1f}s"
                                            for s, d in change["durations"].items()
                                        ]
                                    )
                                    change_str += f" [durations: {durations_str}]"
                                print(f"  - {entity}: {change_str}")

                            print(f"[{format_timestamp(ts)}] Anomaly details recorded")

                        # Store anomaly information
                        anomalies_detected.append(
                            {
                                "timestamp": ts,
                                "score": normalized_score,
                                "raw_score": anomaly_score,
                                "changes": entity_changes,
                            }
                        )
                except Exception as e:
                    print(f"[{format_timestamp(ts)}] Error detecting anomalies: {e}")
                    import traceback

                    traceback.print_exc()

        except Exception as e:
            print(f"[{format_timestamp(ts)}] Error processing data: {e}")
            import traceback

            traceback.print_exc()

    return processed_data, feature_names, anomalies_detected


def train_model_with_data(processed_data, anomaly_model, save_path=None):
    """
    Train the model with the processed data.

    Args:
        processed_data: List of processed data vectors
        anomaly_model: AnomalyDetector model
        save_path: Path to save the model (optional)

    Returns:
        AnomalyDetector: Trained model
    """
    if len(processed_data) > SEQUENCE_LENGTH + PREDICTION_LENGTH:
        print(f"\nTraining model on {len(processed_data)} data points...")

        # Convert to numpy array
        train_data = np.array(processed_data)

        # Update model dimensions if needed
        n_features = train_data.shape[1]
        if hasattr(anomaly_model, "enc_in") and anomaly_model.enc_in != n_features:
            print(f"Updating model dimensions for training: {n_features} features")
            anomaly_model = AnomalyDetector(
                seq_len=SEQUENCE_LENGTH,
                pred_len=PREDICTION_LENGTH,
                enc_in=n_features,
                dec_in=n_features,
                c_out=n_features,
            )

        # Train the model
        anomaly_model.train(
            train_data,
            epochs=TRAINING_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
        )

        # Save the trained model if path provided
        if save_path:
            try:
                anomaly_model.save_model(save_path)
                print(f"Anomaly model saved to {save_path}")
            except Exception as e:
                print(f"Error saving anomaly model: {e}")

        return anomaly_model
    else:
        print(
            f"Not enough data for training. Need at least {SEQUENCE_LENGTH + PREDICTION_LENGTH} points, got {len(processed_data)}."
        )
        return anomaly_model


def query_influxdb_with_time_chunks(start_time, end_time, bucket, max_chunk_days=1):
    """
    Query InfluxDB with time chunks to avoid querying large time windows at once.
    For queries spanning more than a day, splits into daily chunks.

    Args:
        start_time: Start time for the query (datetime with timezone)
        end_time: End time for the query (datetime with timezone)
        bucket: InfluxDB bucket name
        max_chunk_days: Maximum number of days per chunk (default: 1)

    Returns:
        DataFrame with combined results
    """
    global anomaly_model
    all_results = []

    # Calculate total duration
    total_duration = end_time - start_time
    total_days = total_duration.total_seconds() / (24 * 60 * 60)

    print(
        f"Query spans {total_days:.2f} days - {'splitting into daily chunks' if total_days > max_chunk_days else 'using single query'}"
    )

    # If query is less than max_chunk_days, do a single query
    if total_days <= max_chunk_days:
        print(
            f"Executing single query from {format_timestamp(start_time)} to {format_timestamp(end_time)}"
        )
        query = f"""
        from(bucket: "{bucket}")
          |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
          |> filter(fn: (r) =>
              (r["_field"] == "value" or r["_field"] == "state") 
          )
          |> map(fn: (r) => ({{
              r with entity_id: r.domain + "." + r.entity_id
          }}))
          |> drop(columns: ["domain", "_measurement", "_start", "_stop"])
        """

        df = query_api.query_data_frame(query)
        if isinstance(df, list):
            all_results.extend(df)
        else:
            all_results.append(df)

        # Process this chunk immediately
        if not df.empty:
            process_and_train_chunk(
                pd.concat(all_results) if len(all_results) > 1 else all_results[0]
            )

        return (
            pd.DataFrame()
            if not all_results
            else pd.concat(all_results) if len(all_results) > 1 else all_results[0]
        )
    else:
        # Always split into daily chunks
        print(f"Splitting query into daily chunks")
        chunk_start = start_time
        chunk_count = 0

        # Initialize state tracking variables that will persist across chunks
        all_processed_data = []
        all_feature_names = []
        all_anomalies_detected = []
        previous_states = {}
        previous_times = {}
        binary_transitions = {}
        binary_durations = {}

        while chunk_start < end_time:
            # Calculate chunk end (either 1 day later or end_time, whichever is earlier)
            chunk_end = min(chunk_start + timedelta(days=1), end_time)
            chunk_count += 1

            print(
                f"Executing chunk {chunk_count}: {format_timestamp(chunk_start)} to {format_timestamp(chunk_end)}"
            )

            query = f"""
            from(bucket: "{bucket}")
              |> range(start: {chunk_start.isoformat()}, stop: {chunk_end.isoformat()})
              |> filter(fn: (r) =>
                  (r["_field"] == "value" or r["_field"] == "state") 
              )
              |> map(fn: (r) => ({{
                  r with entity_id: r.domain + "." + r.entity_id
              }}))
              |> drop(columns: ["domain", "_measurement", "_start", "_stop"])
            """

            df = query_api.query_data_frame(query)
            chunk_df = pd.DataFrame()

            if isinstance(df, list):
                if df:
                    chunk_df = pd.concat(df)
            else:
                chunk_df = df

            # Process this chunk immediately if we have data
            if not chunk_df.empty:
                print(f"Processing chunk {chunk_count} with {len(chunk_df)} rows")

                # Process this chunk with state from previous chunks
                (
                    chunk_data,
                    chunk_entities,
                    previous_states,
                    previous_times,
                    binary_transitions,
                    binary_durations,
                ) = process_data_chunk(
                    chunk_df,
                    previous_states=previous_states,
                    previous_times=previous_times,
                    binary_transitions=binary_transitions,
                    binary_durations=binary_durations,
                )

                # Detect anomalies with the updated model
                all_processed_data, all_feature_names, all_anomalies_detected = (
                    detect_anomalies(
                        chunk_data,
                        chunk_entities,
                        anomaly_model,
                        processed_data=all_processed_data,
                        feature_names=all_feature_names,
                        anomalies_detected=all_anomalies_detected,
                    )
                )

                # Train the model after processing this chunk
                anomaly_model = train_model_with_data(
                    all_processed_data, anomaly_model, save_path=ANOMALY_MODEL_PATH
                )

                print(f"Completed processing chunk {chunk_count}")
            else:
                print(f"Chunk {chunk_count} has no data, skipping processing")

            # Move to next chunk
            chunk_start = chunk_end

        print(f"Completed {chunk_count} daily chunks")
        print(f"Total processed data points: {len(all_processed_data)}")
        print(f"Total anomalies detected: {len(all_anomalies_detected)}")

        # We've already processed all data directly in the loop
        return pd.DataFrame()


def process_and_train_chunk(chunk_df):
    """
    Process a single chunk of data and train the model on it.

    Args:
        chunk_df: DataFrame containing the chunk data
    """
    global anomaly_model

    if chunk_df.empty:
        print("Chunk has no data, skipping processing")
        return

    print(f"Processing chunk with {len(chunk_df)} rows")

    try:
        # Process this chunk
        (
            chunk_data,
            chunk_entities,

        ) = process_data_chunk(chunk_df)

        # Detect anomalies
        all_processed_data, feature_names, all_anomalies_detected = detect_anomalies(
            chunk_data, chunk_entities, anomaly_model
        )

        # Train the model after processing this chunk
        if len(all_processed_data) > SEQUENCE_LENGTH + PREDICTION_LENGTH:
            anomaly_model = train_model_with_data(
                all_processed_data, anomaly_model, save_path=ANOMALY_MODEL_PATH
            )

            print(
                f"Processed {len(all_processed_data)} data points, detected {len(all_anomalies_detected)} anomalies"
            )
        else:
            print(
                f"Not enough data for training: {len(all_processed_data)} points (need {SEQUENCE_LENGTH + PREDICTION_LENGTH})"
            )
    except Exception as e:
        print(f"Error in process_and_train_chunk: {e}")
        import traceback

        traceback.print_exc()

    return


# === MODEL INITIALIZATION ===
# --- Create or Load Anomaly Model ---
if os.path.exists(ANOMALY_MODEL_PATH) and not FORCE_RETRAIN:
    try:
        # Load existing model.
        print(f"Loading existing model from {ANOMALY_MODEL_PATH}")
        anomaly_model = create_anomaly_model(ANOMALY_MODEL_PATH)
    except Exception as e:
        print(f"Error loading anomaly model: {e}")
        # Fall back to creating a new model.
        print("Creating new model instead")
        anomaly_model = create_anomaly_model()
else:
    # Create new model.
    print("Creating new anomaly detection model")
    anomaly_model = create_anomaly_model()

# === DATA RETRIEVAL ===
# --- Find Latest Data ---
# Query to find the most recent data in InfluxDB.
latest_query = f"""
from(bucket: "{BUCKET}")
  |> range(start: -{DEFAULT_DAYS}d)
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["_measurement"] == "state")
  |> first()
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
stop_time = now.replace(microsecond=0)
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

# --- InfluxDB Query using time chunks ---
# Query for all state and value data in the time range
# The query function now handles processing each chunk after retrieval
df = query_influxdb_with_time_chunks(
    start_time=last_cutoff.replace(microsecond=0),
    end_time=stop_time,
    bucket=BUCKET,
    max_chunk_days=1,  # Split queries into daily chunks
)

print("Processing complete.")


def main():
    """
    Main entry point for the Home Assistant anomaly detection system.
    Determines time range, initializes model, and processes data in daily chunks.
    """
    global anomaly_model

    print("Starting Home Assistant Anomaly Detection System...")

    # === INITIALIZATION ===
    # --- Directory Setup ---
    # Ensure model directory exists.
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Model Initialization ---
    # Load or create the anomaly detection model
    if os.path.exists(ANOMALY_MODEL_PATH) and not FORCE_RETRAIN:
        try:
            # Load existing model.
            print(f"Loading existing model from {ANOMALY_MODEL_PATH}")
            anomaly_model = create_anomaly_model(ANOMALY_MODEL_PATH)
        except Exception as e:
            print(f"Error loading anomaly model: {e}")
            # Fall back to creating a new model.
            print("Creating new model instead")
            anomaly_model = create_anomaly_model()
    else:
        # Create new model.
        print("Creating new anomaly detection model")
        anomaly_model = create_anomaly_model()

    # --- Time Range Determination ---
    # Determine the start and end time for processing
    end_time = datetime.now(timezone.utc).replace(microsecond=0)

    if FORCE_RETRAIN:
        print("Forcing model retraining - using full data range")
        start_time = end_time - timedelta(days=DEFAULT_DAYS)
        if os.path.exists(ANOMALY_MODEL_PATH):
            print("Deleting existing model for retraining")
            os.remove(ANOMALY_MODEL_PATH)
    else:
        # Find the last data point we processed
        latest_time = find_latest_data_time()
        if latest_time:
            # Start from the last data point we processed
            start_time = latest_time
            print(f"Starting from last processed time: {format_timestamp(start_time)}")
        else:
            # Default to DEFAULT_DAYS ago
            start_time = end_time - timedelta(days=DEFAULT_DAYS)
            print(
                f"No previous data point found, using default range: {DEFAULT_DAYS} days"
            )

    # --- Minimum Time Check ---
    # Ensure we have at least 5 minutes of data to process
    time_difference = end_time - start_time
    if time_difference.total_seconds() < 300:  # 300 seconds = 5 minutes
        print(
            f"Only {round(time_difference.total_seconds())} seconds since last update."
        )
        print("Need at least 300 seconds (5 minutes) for a complete chunk.")
        print("Skipping model update to avoid processing incomplete chunks.")
        return

    print(
        f"Processing data from {format_timestamp(start_time)} to {format_timestamp(end_time)}"
    )

    # --- Process the data range ---
    process_data_range(start_time, end_time)

    print("Processing complete.")


def process_data_range(start_time, end_time):
    """
    Process data from InfluxDB over a specified time range.
    For ranges over a day, splits into daily chunks.

    Args:
        start_time: Start time for data processing (datetime with timezone)
        end_time: End time for data processing (datetime with timezone)
    """
    # Calculate total duration
    total_duration = end_time - start_time
    total_days = total_duration.total_seconds() / (24 * 60 * 60)

    print(f"Processing {total_days:.2f} days of data")


    # Process one day at a time
    current_start = start_time
    day_count = 0

    while current_start < end_time:
        day_count += 1
        # Calculate the end time for this chunk (either one day later or the overall end time)
        current_end = min(current_start + timedelta(days=1), end_time)

        print(
            f"Processing day {day_count}: {format_timestamp(current_start)} to {format_timestamp(current_end)}"
        )

        # Query data for this day
        data = query_day_from_influxdb(current_start, current_end)

        if not data.empty:
            # Process this day's data
            try:
                train_model(data)

                )
                print(f"Completed processing day {day_count}")
            except Exception as e:
                print(f"Error processing day {day_count}: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"No data for day {day_count}, skipping")

        # Move to next day
        current_start = current_end

    print(f"Completed processing {day_count} days")


def find_latest_data_time():
    """
    Query InfluxDB to find the timestamp of the most recent data.

    Returns:
        datetime: The timestamp of the most recent data, or None if no data found
    """
    try:
        latest_query = f"""
        from(bucket: "{BUCKET}")
          |> range(start: -{DEFAULT_DAYS}d)
          |> filter(fn: (r) => r["_field"] == "value")
          |> filter(fn: (r) => r["_measurement"] == "state")
          |> first()
        """

        latest_df = query_api.query_data_frame(latest_query)

        if isinstance(latest_df, list):
            if not latest_df:
                return None
            latest_df = pd.concat(latest_df)

        if latest_df.empty:
            return None

        latest_time = pd.to_datetime(latest_df["_time"].iloc[0])
        return latest_time
    except Exception as e:
        print(f"Error finding latest data time: {e}")
        return None


def query_day_from_influxdb(start_time, end_time):
    """
    Query InfluxDB for a single day's data.

    Args:
        start_time: Start time for the query (datetime with timezone)
        end_time: End time for the query (datetime with timezone)

    Returns:
        DataFrame: DataFrame containing the query results
    """
    print(
        f"Querying InfluxDB from {format_timestamp(start_time)} to {format_timestamp(end_time)}"
    )

    query = f"""
    from(bucket: "{BUCKET}")
      |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
      |> filter(fn: (r) =>
          (r["_field"] == "value" or r["_field"] == "state") 
      )
      |> map(fn: (r) => ({{
          r with entity_id: r.domain + "." + r.entity_id
      }}))
      |> drop(columns: ["domain", "_measurement", "_start", "_stop"])
    """

    try:
        results = query_api.query_data_frame(query)

        if isinstance(results, list):
            if not results:
                return pd.DataFrame()
            df = pd.concat(results)
        else:
            df = results

        print(f"Query returned {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame()


def train_model():
    pass


# Call the main function if this script is run directly
if __name__ == "__main__":
    main()
