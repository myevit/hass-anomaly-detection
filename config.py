# Home Assistant Anomaly Detection Configuration

# Connection parameters for InfluxDB
INFLUX_URL = "http://192.168.1.4:8086"  # Use Docker service name
TOKEN = "LY86Tqy1cg5-UYTYPMmHI5opIxC2_NtLiZexyHehiqmL7YLGyHOyEeosm9JXAnoVuNaZT5TYNNcMW1eQK3qW3g=="
ORG = "myeHome"
BUCKET = "home-assistant"

# Model storage paths
MODELS_DIR = "models"
ANOMALY_MODEL_PATH = f"{MODELS_DIR}/anomaly_model.vw"
# META_MODEL_PATH = f"{MODELS_DIR}/meta_model.vw"
# ANOMALY_HISTORY_PATH = f"{MODELS_DIR}/anomaly_history.pkl"

# Data processing parameters
CHUNK_SIZE = "5min"  # Time interval for aggregating data
DEFAULT_DAYS = 30  # Default number of days to look back for historical data

# Anomaly detection thresholds
ANOMALY_THRESHOLD = 0.7  # Primary anomaly score threshold (0-1 scale)
# META_ANOMALY_THRESHOLD = 0.5  # Threshold for meta model anomaly confirmation

# Training control
FORCE_RETRAIN = False  # Set to True to force model retraining from scratch

# Verbosity level
DEBUG = False  # Set to True for more detailed logs
