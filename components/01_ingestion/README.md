# 01 – Integration & Data-Ingestion Layer

Status: 🚧 **In Progress**

## Purpose

Home Assistant's event stream, MQTT topics, and direct REST/GraphQL pulls are funnelled into a unified message bus (Kafka, Redis Streams, or HA WebSocket). Provides both real-time and hourly snapshot delivery to downstream services.

## Milestones

- [ ] Decide on message-bus technology & schema contract
- [ ] Scaffold producer/consumer code
- [ ] Dockerfile + local dev compose stack
- [ ] CI job with basic unit tests

## Artifacts

| File/Dir            | Description                                |
| ------------------- | ------------------------------------------ |
| `src/`              | Python/Go/TS implementation will live here |
| `tests/`            | Unit tests                                 |
| `docs/sequence.mmd` | Sequence diagram (to-be)                   |

## Links

- Parent overview: ../../README.md
- Architecture context: ../../Achitecture/project_overview.md
