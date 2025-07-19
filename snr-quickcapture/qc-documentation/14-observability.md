# Observability

## Overview

Observability in QuickCapture provides deep visibility into system health, performance, and behavior. This document describes the observability stack, metrics, logging, and alerting strategies.

## Observability Components

### 1. Metrics Collection
- **metrics_collector.py**: Collects and aggregates system and application metrics
- **performance_tracker.py**: Tracks performance metrics for key operations
- **Custom Metrics**: Processing time, throughput, error rates, resource usage

#### Example Metrics
- `note_processing_time_seconds`
- `embedding_generation_time_seconds`
- `classification_accuracy`
- `storage_write_latency_seconds`
- `system_cpu_usage_percent`
- `system_memory_usage_percent`

### 2. Logging
- **Structured Logging**: JSON logs for all major events and errors
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Rotation**: Automatic log file rotation and archival
- **Sensitive Data Redaction**: Remove PII from logs

#### Example Log Entry
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "component": "embedding_layer",
  "event": "embedding_generated",
  "note_id": "abc123",
  "duration": 0.85
}
```

### 3. Health Monitoring
- **health_monitor.py**: Monitors system health and component status
- **health checks**: Storage, embedding model, classification, vector store
- **Status endpoints**: `/health` API endpoint for system status

#### Example Health Check
```python
from observability.health_monitor import HealthMonitor
monitor = HealthMonitor()
status = monitor.get_system_health()
print(status)
```

### 4. Alerting
- **Threshold-based Alerts**: Error rate, processing time, resource usage
- **Notification Channels**: Email, Slack, dashboards
- **Alert Rules**: Configurable in `observability.yaml`

#### Example Alert Rule
```yaml
alerts:
  - name: "high_error_rate"
    condition: "error_rate > 0.05"
    severity: "warning"
    notification_channels:
      - type: "email"
        recipients: ["admin@example.com"]
```

## Observability Best Practices
- Monitor all critical system components
- Use structured, searchable logs
- Set actionable alert thresholds
- Regularly review metrics and logs
- Test health checks and alerting
- Document all observability configurations 
noteId: "1890968064c111f0970d05fa391d7ad1"
tags: []

---

 