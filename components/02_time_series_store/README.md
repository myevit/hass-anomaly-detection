# 02 – Time-Series Storage & Feature Store

Status: ✅ **Done**

## Purpose

Optimised database for sensor telemetry (InfluxDB 2.x, TimescaleDB, or Prometheus + Thanos) and separate feature/metadata store for engineered features & labels.

## TODO (next sprint)

- [ ] Select storage technology
- [ ] Define retention and downsampling strategy
- [ ] Expose read/write API contract

## Milestones (all completed)

- [x] Provision InfluxDB 2.x instance (add-on or external)
- [x] Configure Home Assistant ↔︎ InfluxDB stream (no expiry)
- [x] Document bucket/retention and authentication token location

## Artifacts

none

## Links

- Parent overview: ../../README.md

Optimised database uses **InfluxDB 2.x** running outside this repo (add-on or managed instance). Home Assistant's native InfluxDB integration is already configured to stream every state-change with unlimited retention.

This component therefore simply documents the connection details and query conventions used by downstream services.
