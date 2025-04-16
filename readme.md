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