# 01 – Integration & Data-Ingestion Layer

Status: ✅ **Done**

## Purpose

All raw telemetry is first collected by **Home Assistant (HA)**.

• The most recent **30 days** are retained in HA's built-in Recorder database (SQLite/PostgreSQL).
• For **long-term retention**, HA's native **InfluxDB integration** streams every state-change to InfluxDB **with no expiration**.

The ingestion service subscribes to HA's WebSocket API (and/or MQTT topics) to mirror real-time events into a unified message bus (Kafka, Redis Streams, etc.) for downstream processing, while relying on InfluxDB for historical back-fill and replay.

## Milestones

- [x] Configure HA ↔︎ InfluxDB integration with unlimited retention
- [x] Decide on message-bus technology & schema contract

- Parent overview: ../../README.md
- Architecture context: ../../Achitecture/project_overview.md
