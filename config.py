# Home Assistant Anomaly Detection Configuration

# Connection parameters for InfluxDB
INFLUX_URL = "http://192.168.1.4:8086"  # Use Docker service name
TOKEN = "Vhh7Rq9_wMaWr0xG9K5hcc8U1-PjLED9XAO9aTWSMfcDk-wyD9U8B9DNKsro7aOy3lQzOtr5Oh0spnnL2k0ksw=="
ORG = "myeHome"
BUCKET = "home-assistant"

# Model storage paths
MODELS_DIR = "models"
ANOMALY_MODEL_PATH = f"{MODELS_DIR}/autoformer_model.pt"
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

# Autoformer model parameters
SEQUENCE_LENGTH = 96  # Number of time steps to use for input sequence
PREDICTION_LENGTH = 24  # Number of time steps to predict
TRAINING_EPOCHS = 10  # Number of epochs for training
BATCH_SIZE = 16  # Batch size for training
LEARNING_RATE = 0.0001  # Learning rate for Adam optimizer
SIGMA_THRESHOLD = 3.0  # Number of standard deviations for anomaly threshold

# Verbosity level
DEBUG = False  # Set to True for more detailed logs
