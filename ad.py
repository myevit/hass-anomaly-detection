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
from autoformer_model import AnomalyDetector
import math
from sklearn.decomposition import PCA

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


def query_data_from_influxdb(start_time, end_time):
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
    # Calculate number of days needed (rounded up)
    # Calculate total duration

    # Convert to integer (rounding up to ensure we cover all data)
    days_to_process = math.ceil(time_difference.total_seconds() / (24 * 60 * 60))

    # Initialize state tracking variables that will persist across days
    previous_states = {}
    previous_times = {}
    binary_transitions = {}
    binary_durations = {}
    all_processed_data = []

    # Process one day at a time
    current_start = start_time
    for day in range(days_to_process):
        # Calculate the end time for this chunk
        current_end = min(current_start + timedelta(days=1), end_time)

        # Process the day...
        day_data = query_data_from_influxdb(current_start, current_end)

        if not day_data.empty:
            print(f"Processing day {day + 1}")
            process_data(
                day_data,
                previous_states,
                previous_times,
                binary_transitions,
                binary_durations,
                all_processed_data,
            )
        else:
            print(f"No data found for day {day + 1}, skipping")

        # Move to next day
        current_start = current_end

    print(f"Completed processing {days_to_process} days")
    print(f"Total processed data points: {len(all_processed_data)}")


# === DATA PROCESSING FUNCTIONS ===
def process_data(
    day_data,
    previous_states,
    previous_times,
    binary_transitions,
    binary_durations,
    all_processed_data,
):
    """
    Process a single day's data, detect anomalies, and train the model.

    Args:
        day_data: DataFrame containing the day's data
        previous_states: Dictionary of previous states (persisted across days)
        previous_times: Dictionary of previous times (persisted across days)
        binary_transitions: Dictionary of transition counts (persisted across days)
        binary_durations: Dictionary of durations (persisted across days)
        all_processed_data: List of processed data (persisted across days)
    """
    global anomaly_model

    # Process this day's data
    (
        chunk_data,
        chunk_entities,
        new_states,
        new_times,
        new_transitions,
        new_durations,
    ) = process_data_chunk(
        day_data,
        previous_states=previous_states,
        previous_times=previous_times,
        binary_transitions=binary_transitions,
        binary_durations=binary_durations,
    )

    # Update the state tracking variables for the next day
    previous_states.update(new_states)
    previous_times.update(new_times)
    binary_transitions.update(new_transitions)
    binary_durations.update(new_durations)

    if not chunk_data:
        print("No data to process after chunking")
        return

    # Detect anomalies
    day_processed_data, feature_names, anomalies = detect_anomalies(
        chunk_data, chunk_entities, anomaly_model, processed_data=all_processed_data
    )

    # Update all_processed_data with this day's processed data
    all_processed_data.extend(day_processed_data)

    # Keep a maximum history size to prevent memory issues (optional)
    max_history = 1000  # Adjust based on your memory constraints
    if len(all_processed_data) > max_history:
        # Remove oldest data points, keeping the most recent ones
        all_processed_data[:] = all_processed_data[-max_history:]

    # Train the model with all processed data
    if len(all_processed_data) > SEQUENCE_LENGTH + PREDICTION_LENGTH:
        print(f"Training model with {len(all_processed_data)} data points")
        anomaly_model = train_model_with_data(
            all_processed_data, anomaly_model, save_path=ANOMALY_MODEL_PATH
        )
        print(f"Model training complete. Detected {len(anomalies)} anomalies.")
    else:
        print(
            f"Not enough data points for training: {len(all_processed_data)} (need {SEQUENCE_LENGTH + PREDICTION_LENGTH})"
        )


def clean_value(value):
    """
    Clean and normalize a value for formatting.
    Handles nulls, empty strings, and formatting.
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
        # Replace problematic characters
        value = value.replace("|", "_").replace(":", "_").replace(" ", "_")

    return str(value).strip()


def is_valid_value(value):
    """Simplified validity check that works with clean_value"""
    return clean_value(value) != "NA"


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
    Track changes between two consecutive data points.

    Args:
        current_data: Dictionary of current data values
        previous_data: Dictionary of previous data values
        entity_cols: Set of entity column names

    Returns:
        dict: Dictionary of changes detected
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

        # Skip invalid values
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
                    # Just one value - store with zero change metrics
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

    if not timestamps:
        print("No timestamps to process")
        return processed_data, feature_names, anomalies_detected

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
    feature_count = len(all_feature_names)
    print(f"Total feature count: {feature_count}")

    # If we have too many features, select the most important ones based on prior data
    max_features = 500  # Limit for memory efficiency
    if feature_count > max_features and len(processed_data) > 0:
        try:
            # Get most recently used feature names
            if len(feature_names) > 0:
                # Keep consistent with previously used features if possible
                common_features = [
                    f for f in feature_names if f in all_possible_features
                ]
                if len(common_features) >= max_features // 2:
                    # If we have enough common features with previous data, prioritize those
                    remaining_count = max_features - len(common_features)
                    new_features = [
                        f for f in all_feature_names if f not in common_features
                    ][:remaining_count]
                    all_feature_names = common_features + new_features
                else:
                    # Otherwise, just take the first max_features
                    all_feature_names = all_feature_names[:max_features]
            else:
                # If no previous feature names, just limit to max_features
                all_feature_names = all_feature_names[:max_features]

            print(f"Limited features to {len(all_feature_names)} for efficiency")
        except Exception as e:
            print(f"Error limiting features: {e}")
            # Fall back to using all features if there's an error
            pass

    # Flag to track if Autoformer model is having issues
    model_failure_count = 0
    use_fallback_method = False

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
                        try:
                            idx = all_feature_names.index(name)
                            feature_vector[idx] = float(value)
                            feature_dict[name] = float(value)
                        except (ValueError, IndexError) as e:
                            print(f"Error adding numeric feature {name}: {e}")

            # Add context features
            for name, value in context_features.items():
                feature_name = f"ctx_{name}"
                if (
                    isinstance(value, (int, float))
                    and not pd.isna(value)
                    and feature_name in all_possible_features
                ):
                    try:
                        idx = all_feature_names.index(feature_name)
                        feature_vector[idx] = float(value)
                        feature_dict[feature_name] = float(value)
                    except (ValueError, IndexError) as e:
                        print(f"Error adding context feature {feature_name}: {e}")

            # Convert binary transitions to numeric features
            for entity, count in current_time_data["binary_transitions"].items():
                feature_name = f"trans_{entity}"
                if count > 0 and feature_name in all_possible_features:
                    try:
                        idx = all_feature_names.index(feature_name)
                        feature_vector[idx] = float(count)
                        feature_dict[feature_name] = float(count)
                    except (ValueError, IndexError) as e:
                        print(f"Error adding transition feature {feature_name}: {e}")

            # Store processed data
            processed_data.append(feature_vector)

            # Set feature names if not already set
            if not feature_names:
                feature_names = all_feature_names

            # Calculate global index (including previously processed data)
            global_index = len(processed_data) - 1

            # Evaluate with existing model (if we have enough data and model is working)
            if (
                global_index >= SEQUENCE_LENGTH
                and anomaly_model is not None
                and not use_fallback_method
            ):
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

                    # Reset failure count on success
                    model_failure_count = 0

                except Exception as e:
                    print(f"[{format_timestamp(ts)}] Error in model inference: {e}")
                    model_failure_count += 1

                    if model_failure_count >= 3:
                        print(
                            "Multiple model failures detected. Switching to fallback method."
                        )
                        use_fallback_method = True

                    # Use fallback method
                    is_anomaly, anomaly_score, normalized_score = (
                        fallback_anomaly_detection(
                            feature_vector, processed_data, global_index
                        )
                    )

            # Use fallback method if model is having issues or not enough data
            elif global_index >= 10:  # Need at least some history for fallback
                is_anomaly, anomaly_score, normalized_score = (
                    fallback_anomaly_detection(
                        feature_vector, processed_data, global_index
                    )
                )
            else:
                # Not enough data for any detection
                continue

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
                        change_str = f"'{change['previous']}' → '{change['current']}'"
                        if "diff" in change:
                            change_str += f" (diff: {change['diff']:.4f})"
                        if "transitions" in change:
                            change_str += f" [transitions: {change['transitions']}]"
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
            print(f"[{format_timestamp(ts)}] Error processing data: {e}")
            import traceback

            traceback.print_exc()

    return processed_data, feature_names, anomalies_detected


def fallback_anomaly_detection(current_features, processed_data, index):
    """
    Simple statistical fallback method for anomaly detection when the model fails.
    Uses z-score based approach to detect outliers.

    Args:
        current_features: Feature vector for current timestamp
        processed_data: All processed feature vectors
        index: Index of current features in processed_data

    Returns:
        tuple: (is_anomaly, score, normalized_score)
    """
    # Need some history for this to work
    if index < 10:
        return False, 0.0, 0.0

    try:
        # Get historical data (excluding current point)
        history = processed_data[max(0, index - 50) : index]
        history_array = np.array(history)

        # Calculate mean and std for each feature
        means = np.mean(history_array, axis=0)
        stds = (
            np.std(history_array, axis=0) + 1e-10
        )  # Add small value to avoid division by zero

        # Calculate z-scores for current features
        z_scores = np.abs((current_features - means) / stds)

        # Get maximum z-score as anomaly score
        max_z_score = np.max(z_scores)

        # Normalize score between 0 and 1 using sigmoid function
        normalized_score = 1.0 / (1.0 + np.exp(-(max_z_score - 3)))

        # Determine if it's an anomaly
        is_anomaly = normalized_score > ANOMALY_THRESHOLD

        return is_anomaly, max_z_score, normalized_score

    except Exception as e:
        print(f"Error in fallback anomaly detection: {e}")
        return False, 0.0, 0.0


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

        try:
            # Convert to numpy array
            train_data = np.array(processed_data)

            # Get feature count
            n_features = train_data.shape[1]
            print(f"Original feature count: {n_features}")

            # Apply dimensionality reduction if feature count is too high
            max_features = 64  # Maximum features the model can handle
            if n_features > max_features:
                print(
                    f"Applying dimensionality reduction from {n_features} to {max_features} features"
                )

                try:
                    # Use PCA for dimensionality reduction
                    pca = PCA(n_components=max_features)
                    train_data = pca.fit_transform(train_data)
                    print(f"Data shape after reduction: {train_data.shape}")
                    n_features = max_features
                except ImportError:
                    # Fallback if sklearn is not available: simple feature selection
                    print(
                        "sklearn not available, using simple feature selection instead"
                    )

                    # Calculate variance of each feature
                    feature_variance = np.var(train_data, axis=0)

                    # Get indices of features with highest variance
                    top_feature_indices = np.argsort(feature_variance)[-max_features:]

                    # Select only those features
                    train_data = train_data[:, top_feature_indices]
                    print(f"Data shape after selection: {train_data.shape}")
                    n_features = max_features

            # Update model dimensions
            print(f"Creating model with {n_features} features")
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
        except Exception as e:
            print(f"Error training model: {e}")
            print("Using default model instead")
            return anomaly_model
    else:
        print(
            f"Not enough data for training. Need at least {SEQUENCE_LENGTH + PREDICTION_LENGTH} points, got {len(processed_data)}."
        )
        return anomaly_model


# Call the main function if this script is run directly
if __name__ == "__main__":
    main()
