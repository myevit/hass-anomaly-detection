#!/bin/bash

# docker-loop.sh - Run anomaly detection script in a loop within Docker container

# Default sleep time between runs (in seconds)
SLEEP_TIME=${LOOP_INTERVAL:-900}  # Default to 15 minutes if not set

echo "Starting Home Assistant Anomaly Detection loop"
echo "Will run script every ${SLEEP_TIME} seconds"

# Check if config.py exists
if [ ! -f "/app/config.py" ]; then
  echo "ERROR: config.py not found! Make sure to mount it as a volume."
  echo "Checking if a default config.py was included in the image..."
  ls -la /app/
  exit 1
fi

# Run in a continuous loop
while true; do
  echo "========================================================"
  echo "Running anomaly detection at $(date)"
  echo "========================================================"
  
  # Run the main script
  python /app/ha-data.py
  
  # Capture exit code
  EXIT_CODE=$?
  
  # If the script exited with an error, log it but continue
  if [ $EXIT_CODE -ne 0 ]; then
    echo "Script exited with error code ${EXIT_CODE}"
    # Optional: reduce sleep time after errors to retry sooner
    SLEEP_AFTER_ERROR=${SLEEP_AFTER_ERROR:-300}  # 5 minutes
    echo "Will retry in ${SLEEP_AFTER_ERROR} seconds"
    sleep ${SLEEP_AFTER_ERROR}
  else
    echo "Script completed successfully"
    echo "Next run in ${SLEEP_TIME} seconds"
    sleep ${SLEEP_TIME}
  fi
done 