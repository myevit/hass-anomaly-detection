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
import holidays
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
import math

# Home Assistant Anomaly Detection Configuration

# Connection parameters for InfluxDB
INFLUX_URL = "http://192.168.1.4:8086"  # Use Docker service name
TOKEN = "Vhh7Rq9_wMaWr0xG9K5hcc8U1-PjLED9XAO9aTWSMfcDk-wyD9U8B9DNKsro7aOy3lQzOtr5Oh0spnnL2k0ksw=="
ORG = "myeHome"
BUCKET = "home-assistant"

# Model storage paths
MODELS_DIR = "models"
ANOMALY_MODEL_PATH = f"{MODELS_DIR}/autoformer_model.pt"


# Data processing parameters
CHUNK_SIZE = "5min"  # Time interval for aggregating data
DEFAULT_DAYS = 30  # Default number of days to look back for historical data

# Anomaly detection thresholds
ANOMALY_THRESHOLD = 0.7  # Primary anomaly score threshold (0-1 scale)
# META_ANOMALY_THRESHOLD = 0.5  # Threshold for meta model anomaly confirmation

# Training control
FORCE_RETRAIN = False  # Set to True to force model retraining from scratch

# Autoformer model parameters
SEQUENCE_LENGTH = 96  # Number of time steps to use for input sequence
PREDICTION_LENGTH = 24  # Number of time steps to predict
TRAINING_EPOCHS = 10  # Number of epochs for training
BATCH_SIZE = 16  # Batch size for training
LEARNING_RATE = 0.0001  # Learning rate for Adam optimizer
SIGMA_THRESHOLD = 3.0  # Number of standard deviations for anomaly threshold

# Verbosity level
DEBUG = False  # Set to True for more detailed logs


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

    # Process one day at a time
    current_start = start_time
    for day in range(days_to_process):
        # Calculate the end time for this chunk
        current_end = min(current_start + timedelta(days=1), end_time)

        # Process the day...
        data = query_data_from_influxdb(current_start, current_end)

        if not data.empty:
            print(f"Processing day {day + 1}")

            process_data()
        else:
            print(f"No data found for day {day + 1}, skipping")

        # Move to next day
        current_start = current_end

    print(f"Completed processing {days_to_process} days")


# === DATA PROCESSING FUNCTIONS ===
def process_data():
    pass


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


# Call the main function if this script is run directly
if __name__ == "__main__":
    main()
