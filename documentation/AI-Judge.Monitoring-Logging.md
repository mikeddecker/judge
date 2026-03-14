# AI Judge Monitoring & Logging Strategy

**Document Status**: Complete (March 2026)  
**Last Updated**: 2026-03-14

## Overview

Comprehensive monitoring and structured logging are essential for multi-region operations. This document outlines health checks, metrics exposure, centralized logging, and alerting.

## Health Endpoints

The API exposes health checks at standard paths:

### `GET /health`

Basic liveness probe. Returns 200 if API is running.

```bash
curl http://localhost:5555/health
```

**Response** (200 OK):
```json
{"status": "healthy"}
```

**Used by**: Docker healthcheck, load balancers

### `GET /health/readiness`

Readiness probe. Returns 200 if API can serve traffic (DB connected, external services reachable).

```bash
curl http://localhost:5555/health/readiness
```

**Response** (200 OK):
```json
{
  "ready": true,
  "database": "connected",
  "cache": "connected",
  "external_services": [
    {"name": "sync-gateway", "status": "healthy"},
    {"name": "storage", "status": "healthy"}
  ]
}
```

**Used by**: Kubernetes readiness probes, deploy checks before routing traffic

### `GET /health/database-replica-lag`

**Production-only endpoint**. Returns replication lag from primary MySQL.

```bash
curl http://localhost:5555/health/database-replica-lag
```

**Response** (200 OK):
```json
{
  "replica_lag_seconds": 2.5,
  "primary": "belgium",
  "is_synced": true,
  "timestamp": "2026-03-14T15:32:45Z"
}
```

**Response** (503 Service Unavailable) - if lag > 5 seconds:
```json
{
  "replica_lag_seconds": 127.3,
  "primary": "belgium",
  "is_synced": false,
  "error": "Replica lag exceeds 5 seconds"
}
```

**Used by**: Monitoring systems to detect replication issues, circuit breaker logic

## Metrics Endpoint

The API exports Prometheus-compatible metrics at `/metrics`.

### Registration

```bash
curl http://localhost:5555/metrics
```

### Metric Types

**Request Latency**:
```
http_request_duration_seconds{method="GET",endpoint="/videos",status="200"} 0.245
http_request_duration_seconds{method="POST",endpoint="/labels",status="201"} 1.823
```

**Request Rate**:
```
http_requests_total{method="GET",endpoint="/videos",status="200"} 15234
http_requests_total{method="POST",endpoint="/labels",status="201"} 4521
```

**Database Metrics**:
```
mysql_replication_lag_seconds 2.5
mysql_queries_total 582341
mysql_slow_queries_total 23
mysql_connection_errors_total 0
```

**Job Queue**:
```
job_queue_length{category="AI"} 12
job_queue_length{category="SYNC"} 3
job_queue_length{category="BACKUP"} 0
job_processing_duration_seconds{category="AI",status="completed"} 1234.5
```

**Sync Operations**:
```
sync_operations_total{type="labeled_videos",status="success"} 1024
sync_operations_total{type="generated_data",status="failed"} 12
sync_transfer_bytes{direction="belgium_to_usa"} 5368709120
```

**Conflict Tracking**:
```
conflicts_total{entity_type="FrameLabel",auto_resolved="true"} 8
conflicts_total{entity_type="FrameLabel",auto_resolved="false"} 2
conflicts_unresolved 2
```

**Storage**:
```
storage_bytes_used{path="STORAGE_DIR_VIDEOS"} 1099511627776
storage_bytes_available{path="STORAGE_DIR_GENERATED_DATA"} 536870912000
backup_count 87
```

### Prometheus Configuration

Add to `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-judge-api'
    static_configs:
      - targets: ['localhost:5555']
    metrics_path: '/metrics'
    scrape_interval: 30s  # More frequent for production
```

## Structured Logging

All application logs use `structlog` with JSON format for programmatic parsing.

### Log Format

```json
{
  "timestamp": "2026-03-14T15:32:45.123Z",
  "level": "INFO",
  "service": "ai-judge-api",
  "region": "belgium",
  "correlation_id": "req-abc123xyz",
  "event": "job_completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "PREDICT",
  "duration_ms": 1234,
  "status": "success"
}
```

### Log Levels & Examples

**DEBUG**:
```json
{"level": "DEBUG", "event": "database_query", "query": "SELECT * FROM Videos", "duration_ms": 2}
```

**INFO**:
```json
{"level": "INFO", "event": "sync_job_started", "data_type": "labeled_videos", "target_region": "usa"}
```

**WARNING**:
```json
{"level": "WARNING", "event": "slow_query_detected", "query_id": "q123", "duration_ms": 534}
```

**ERROR**:
```json
{"level": "ERROR", "event": "sync_failed", "reason": "connection_timeout", "retry_count": 3}
```

**CRITICAL**:
```json
{"level": "CRITICAL", "event": "database_replication_failed", "ping_response": null, "action": "escalate"}
```

### Log Directory Structure

```
${STORAGE_DIR_GENERATED_DATA}/logs/
├── api/
│   ├── belgium/
│   │   ├── api-2026-03-14.log
│   │   ├── sync-manager-2026-03-14.log
│   │   └── backup-rotation-2026-03-14.log
│   └── usa/
│       ├── api-2026-03-14.log
│       ├── sync-manager-2026-03-14.log
│       └── backup-rotation-2026-03-14.log
└── database/
    ├── slow-queries-2026-03-14.log
    └── replication-2026-03-14.log
```

**Auto-cleanup**: Logs older than `LOG_RETENTION_DAYS` (default 30) are automatically deleted.

## Correlation IDs

Each request/job gets a unique correlation ID to trace across services:

```
Request → Belgium API → Sync Job → USA API → Database replica
  ↓
req-abc123xyz (traced across all components)
```

**In logs**:
```json
{"correlation_id": "req-abc123xyz", "service": "api", "event": "started"}
{"correlation_id": "req-abc123xyz", "service": "sync-manager", "event": "transferring", "bytes": 2048000}
{"correlation_id": "req-abc123xyz", "service": "api", "endpoint": "/metrics", "event": "completed", "status": 200}
```

## Alerting Rules

### Critical Alerts (page on-call)

1. **API Down**: No response from `/health` for 5 minutes
2. **Database Replication Lag > 5 seconds**: Data divergence risk
3. **Disk Usage > 90%**: Storage critical
4. **Backup Missing**: No backup in last 3 hours
5. **Sync Queue Backlog**: > 100 pending jobs for 1+ hour

### Warning Alerts (ticket, resolved within SLA)

1. **High Error Rate**: >5% of requests returning 5xx
2. **Slow Queries**: >10 queries >200ms in last 5 minutes
3. **Memory Usage > 80%**: Potential OOM kill incoming
4. **Disk Usage > 80%**: Plan cleanup
5. **Unresolved Conflicts > 10**: Users need attention

### Info Alerts (logged only, no notification)

1. **Job Completed**: Track processing patterns
2. **Backup Rotation Completed**: Logs freed space
3. **Replication Synced**: Confirms all is well

### AlertManager Configuration

```yaml
groups:
  - name: ai_judge_critical
    rules:
      - alert: APIDown
        expr: up{job="ai-judge-api"} == 0
        for: 5m
        annotations:
          summary: "AI Judge API is down"
          
      - alert: ReplicationLagHigh
        expr: mysql_replication_lag_seconds > 5
        for: 2m
        annotations:
          summary: "Database replication lag > 5s"
          
      - alert: DiskCritical
        expr: storage_bytes_available / storage_bytes_total < 0.1
        for: 5m
        annotations:
          summary: "Disk usage > 90%, clean up now"
```

## Grafana Dashboards

### Dashboard 1: API Health

Panels:
- Request rate (req/s) over time
- Response latency (p50, p95, p99) by endpoint
- Error rate (5xx, 4xx by endpoint)
- CPU/memory usage
- Connected clients

### Dashboard 2: Database Replication

Panels:
- Replication lag (seconds)
- Binary log position (bytes)
- Write IOPS by region
- Query count by type
- Slow query count

### Dashboard 3: Sync Operations

Panels:
- Sync job queue depth by type
- Transfer rate (bytes/sec)
- Success/failure rate by data type
- Duration histogram by operation
- Conflict count (auto-resolved vs user-resolved)

### Dashboard 4: Backup & Recovery

Panels:
- Backup count and total size
- Last backup timestamp
- Backup age distribution (hours, days, weeks, months tiers)
- Rotation metrics (deleted count, freed space)
- Storage trend

### Dashboard 5: Conflicts

Panels:
- Conflict count by entity type
- Conflict distribution (auto-resolved % vs user-resolved %)
- Average resolution time
- Conflicts per region
- Unresolved conflict list

## Log Analysis Queries

### ELK / Loki

**Find all errors in last hour**:
```
level: ERROR AND timestamp: [now-1h TO now]
```

**Find sync latency p95 by data type**:
```
event: "sync_completed"
| stats percentile(duration_ms, 95) by data_type
```

**Find unresolved conflicts by account**:
```
event: "conflict_logged" AND auto_resolved: false
| stats count by resolved_by
```

**Track replication lag over time**:
```
event: "replication_lag" 
| stats avg(lag_seconds), max(lag_seconds) by region, 5m
```

## Monitoring Best Practices

1. **Alert Fatigue Prevention**:
   - Set thresholds based on historical baselines, not arbitrary numbers
   - Use `for` durations to reduce false positives (e.g., 5m sustained high lag vs spikes)
   - Group related alerts (don't page for every small issue)

2. **Correlation ID Propagation**:
   - Every API request gets a unique ID
   - Pass ID to background jobs, sync operations, database queries
   - Include in error messages for debugging

3. **Log Retention**:
   - Raw logs: 30 days (GDPR auto-delete)
   - Aggregated metrics: 1 year
   - Alert history: 1 year
   - Escalations/incidents: 3 years (compliance)

4. **Access Control**:
   - Metrics endpoint: Protected by IP whitelist (internal only)
   - Health endpoints: Public (load balancers need access)
   - Logs: Role-based access (ops team only for production logs)

5. **Privacy**:
   - Don't log video content or user data
   - Log only metadata (video IDs, account IDs, sizes, timestamps)
   - Sanitize error messages (remove file paths from user-facing responses)

## Quick Troubleshooting

### API is slow

```bash
# Check metrics
curl http://localhost:5555/metrics | grep http_request_duration

# Check logs for slow queries
grep "slow_query_detected\|duration_ms.*[0-9]{4}" logs/api/*/api-*.log

# Check database replication lag
curl http://localhost:5555/health/database-replica-lag
```

### Sync jobs not running

```bash
# Check sync queue length
curl http://localhost:5555/metrics | grep 'job_queue_length{category="SYNC"}'

# Check if sync manager is alive
docker ps | grep sync

# Check sync logs for errors
tail -f logs/api/*/sync-manager-*.log | grep ERROR

# Verify SSH connectivity to remote region
ssh sync-user@remote-host "echo 'OK'"
```

### High conflict rate

```bash
# Find unresolved conflicts
echo 'SELECT * FROM ConflictLogs WHERE is_resolved = FALSE;' | mysql

# Check auto-resolved conflicts (should be >90%)
echo 'SELECT auto_resolved, COUNT(*) FROM ConflictLogs GROUP BY auto_resolved;' | mysql

# Review conflict details
tail -f logs/api/*/api-*.log | grep "conflict_logged"
```

### Replication lag increasing

```bash
# Check binary log on primary
mysql -e "SHOW MASTER STATUS;"

# Check replica status
mysql -e "SHOW SLAVE STATUS\G" | grep Seconds_Behind_Master

# Check for long-running queries
mysql -e "SELECT * FROM INFORMATION_SCHEMA.PROCESSLIST WHERE TIME > 300;"

# If stuck, may need manual intervention
# See: documentation/AI-Judge.Disaster-Recovery.md
```

## Related Documents

- [AI-Judge.Disaster-Recovery.md](AI-Judge.Disaster-Recovery.md) - Escalation procedures
- [AI-Judge.Multi-Region.md](AI-Judge.Multi-Region.md) - Architecture details
- [PROJECT_RULES.md](../PROJECT_RULES.md) - Testing and CI requirements

