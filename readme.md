# Home Assistant Anomaly Detection

This Docker-based solution provides context-aware anomaly detection for Home Assistant smart home systems, analyzing time-series data to detect abnormal patterns.

## Features

- Monitors Home Assistant entities via InfluxDB data
- Detects anomalies using machine learning (Vowpal Wabbit)
- Provides context-aware detection considering time, day, season
- Identifies recurring patterns and anomaly frequencies
- Maintains history of detected anomalies
- Suggests optimal threshold settings based on historical data

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed on your host system
- InfluxDB instance with Home Assistant data
- Network access between the container and InfluxDB

### Configuration

1. Clone this repository to your server:
   ```
   git clone [repository-url]
   cd hass-anomaly-detection
   ```

2. **Important:** Create `config.py` file with your connection details:
   ```
   cp config.py.example config.py
   nano config.py  # Edit with your settings
   ```

   Make sure to update these essential settings:
   ```python
   # Most important settings to change
   INFLUX_URL = "http://your-influxdb-host:8086"  # Use container name if on same network
   TOKEN = "your-influxdb-token"
   ORG = "your-organization"
   BUCKET = "home-assistant"
   ```

   The Docker setup REQUIRES a valid config.py file to run.

### Building and Running

1. Build and start the container:
   ```
   docker-compose up -d
   ```

2. Check logs to verify proper execution:
   ```
   docker-compose logs -f
   ```

3. The container will automatically restart unless explicitly stopped.

## Data Persistence

The system stores model data in the `./models` directory which is mounted as a volume. This ensures that:

- Trained models persist between container restarts
- Anomaly history is maintained
- No retraining is needed when the container restarts

## Scheduling

The script runs continuously within the container, processing data in incremental chunks. It will:

1. Check for new data since the last run
2. Process only if there's at least 5 minutes of new data
3. Update models and anomaly history
4. Wait for the next execution (15 minutes by default)

## Troubleshooting

- If the container stops unexpectedly, check logs with `docker-compose logs ha-anomaly`
- Make sure config.py exists and has valid connection information
- Verify InfluxDB connectivity from within the container
- Ensure proper permissions on the models directory

## Advanced Configuration

Additional settings can be modified in the `config.py` file:

- `CHUNK_SIZE`: Time interval for aggregating data (default: "5min")
- `DEFAULT_DAYS`: Default lookback period when no history exists (default: 7)
- `ANOMALY_THRESHOLD`: Primary detection threshold (default: 0.9)
- `FORCE_RETRAIN`: Force model retraining from scratch (default: False)

You can also adjust Docker-specific settings in docker-compose.yml:
- `LOOP_INTERVAL`: Time between script executions in seconds (default: 900)
- `SLEEP_AFTER_ERROR`: Time to wait after an error before retry (default: 300)

Based on the latest research and industry practices for time series anomaly detection, here are some best practices that could improve your Home Assistant anomaly detection system:
Chunking Improvements
Variable window sizes: While your 5-minute chunking provides a good baseline, consider implementing multiple time windows (5-min, 1-hour, 24-hour) to detect anomalies at different scales. This helps catch both rapid fluctuations and slow pattern changes.
Binary State Handling
Count state changes: Rather than just keeping the last state in a 5-minute window, count the number of state transitions (e.g., door opened 3 times in 5 minutes) to better detect unusual activity patterns.
Duration in states: Track how long a binary device stays in each state within the window.
Model Improvements
Implement forecasting: Add time series forecasting to predict expected values and calculate anomaly scores based on deviations from predictions.
Semi-supervised approach: Your Home Assistant system naturally has long periods of "normal" behavior - use this to your advantage with a semi-supervised model that learns normal patterns.
Threshold Optimization
Dynamic thresholds: Your fixed threshold of 0.9 is likely too high. Implement adaptive thresholds that vary by:
Time of day (what's normal at 3PM differs from 3AM)
Day of week
Device type (temperature sensors vs. motion detectors)
Feature Engineering
Context-aware features: Add more contextual features like:
Correlation between devices (bathroom light + bathroom motion sensor)
Sequences of activations (front door → hallway motion → living room light)
External data (weather, sunrise/sunset times)
The VictoriaMetrics article emphasizes that there's "no one-size-fits-all" approach - your solution should combine multiple techniques based on your specific smart home environment and most common anomaly types.


https://victoriametrics.com/blog/victoriametrics-anomaly-detection-handbook-chapter-3/