# Home Assistant Anomaly Detection with Autoformer

This project provides an anomaly detection system for Home Assistant data using the Autoformer deep learning model. The system analyzes time-series data from Home Assistant and identifies potential anomalies in smart home behavior.

## Features

- Connects to InfluxDB to retrieve Home Assistant time-series data
- Uses Autoformer, a state-of-the-art time series forecasting model for anomaly detection
- Considers contextual factors such as time of day, day of week, and seasonality
- Learns patterns of normal behavior and identifies deviations
- Provides detailed information about detected anomalies

## Requirements

- Python 3.8 or higher
- Home Assistant with InfluxDB integration
- PyTorch (CPU or GPU)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hass-anomaly-detection.git
   cd hass-anomaly-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `config.py` file with your InfluxDB connection details:
   ```python
   # Example configuration
   INFLUX_URL = "http://your-influxdb-host:8086"
   TOKEN = "your-influxdb-token"
   ORG = "your-organization"
   BUCKET = "home-assistant"
   
   MODELS_DIR = "models"
   ANOMALY_MODEL_PATH = f"{MODELS_DIR}/autoformer_model.pt"
   
   CHUNK_SIZE = "5min"
   DEFAULT_DAYS = 30
   ANOMALY_THRESHOLD = 0.7
   FORCE_RETRAIN = False
   
   # Autoformer parameters
   SEQUENCE_LENGTH = 96
   PREDICTION_LENGTH = 24
   TRAINING_EPOCHS = 10
   BATCH_SIZE = 16
   LEARNING_RATE = 0.0001
   SIGMA_THRESHOLD = 3.0
   ```

## Usage

Run the anomaly detection script:

```bash
python ha-data.py
```

The script will:
1. Connect to your InfluxDB instance and retrieve Home Assistant data
2. Process the data into time-series chunks
3. Train the Autoformer model if no existing model is found
4. Detect anomalies in the data and output details of any findings
5. Save the trained model for future use

## Understanding Anomaly Detection Results

When an anomaly is detected, the script will output:
- The timestamp of the anomaly
- An anomaly score (higher values indicate stronger anomalies)
- Details of which entities changed and how they changed
- Duration and transition information for binary state entities

## Advanced Configuration

You can adjust various parameters in the `config.py` file to customize the behavior:

- `SEQUENCE_LENGTH`: Length of input sequence for the model
- `PREDICTION_LENGTH`: Length of prediction sequence
- `ANOMALY_THRESHOLD`: Threshold for considering an observation anomalous (0-1)
- `TRAINING_EPOCHS`: Number of training epochs
- `SIGMA_THRESHOLD`: Number of standard deviations for anomaly threshold

## How It Works

The system uses Autoformer, a transformer-based model designed specifically for time series forecasting. Autoformer combines auto-correlation mechanisms with a deep decomposition architecture to effectively model time series data.

For anomaly detection, the system:
1. Processes raw Home Assistant data into feature vectors
2. Trains the Autoformer model on these sequences
3. Uses the trained model to predict expected values
4. Calculates anomaly scores based on the difference between actual and predicted values
5. Flags time points that exceed the anomaly threshold

https://victoriametrics.com/blog/victoriametrics-anomaly-detection-handbook-chapter-3/