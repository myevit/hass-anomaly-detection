# Home Assistant Anomaly-Detection Suite

A modular, micro-service-oriented toolkit that ingests raw Home Assistant telemetry, detects anomalies, reasons about context with LLMs, and proposes (or executes) remediations.

## Component Directory Map

| #   | Folder                              | Purpose                                                                                        | Status         |
| --- | ----------------------------------- | ---------------------------------------------------------------------------------------------- | -------------- |
| 01  | `components/01_ingestion`           | Integrate Home Assistant event stream, MQTT topics, and REST pulls into a unified message bus. | 🚧 In Progress |
| 02  | `components/02_time_series_store`   | Optimised time-series DB & feature store.                                                      | 🗓 Planned      |
| 03  | `components/03_feature_engineering` | Data cleansing, resampling, and feature derivation.                                            | Planned        |
| 04  | `components/04_anomaly_detection`   | Rule-based & ML models run hourly to flag anomalies.                                           | Planned        |
| 05  | `components/05_media_analysis`      | Optional worker for image/video understanding.                                                 | Planned        |
| 06  | `components/06_llm_reasoner`        | LLM-powered explanation & action planning.                                                     | Planned        |
| 07  | `components/07_notification_ux`     | Push notifications and feedback capture.                                                       | Planned        |
| 08  | `components/08_automation_executor` | Executes approved actions via Home Assistant services.                                         | Planned        |
| 09  | `components/09_model_training`      | Offline model training & evaluation pipelines.                                                 | Planned        |
| 10  | `components/10_observability`       | Metrics, logs, and health dashboards.                                                          | Planned        |
| 11  | `components/11_devops`              | CI/CD pipelines and deployment manifests.                                                      | Planned        |
| 12  | `components/12_docs_config`         | Centralised docs, schemas, and config templates.                                               | Planned        |

---

📚 High-level architecture lives in `architecture/project_overview.md`.

🔖 Current sprint focus lives in `.llm_context/global_status.yml`.
