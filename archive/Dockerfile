FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration file first
COPY config.py .

# Copy application code
COPY ha-data.py .

# Copy the loop script
COPY docker-loop.sh .
RUN chmod +x docker-loop.sh

# Create directory for models
RUN mkdir -p models

# Run with unbuffered output for better logging
ENV PYTHONUNBUFFERED=1

# Default loop interval (15 min)
ENV LOOP_INTERVAL=900

# Command to run the script in a loop
CMD ["./docker-loop.sh"] 