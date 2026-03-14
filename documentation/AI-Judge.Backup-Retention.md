# AI Judge Backup Retention Strategy

**Document Status**: Complete (March 2026)  
**Last Updated**: 2026-03-14

## Overview

Backups are the foundation of disaster recovery. The AI Judge system implements an automated 2-hour backup rotation strategy with granular retention tiers to balance storage costs against recovery window flexibility.

## Design Principles

1. **Frequent Recent Backups**: Keep hourly-to-2-hour resolution for recent data (24h window)
2. **Coarser Long-term Backups**: Reduce frequency for older backups using daily, weekly, monthly tiers
3. **Perpetual Retention of Milestone Backups**: Monthly backups preserved indefinitely for compliance/audit
4. **Automated Execution**: APScheduler in Flask handles scheduled backups; no external cron required
5. **Clean Shutdown Backups**: Additional labeled backup on app shutdown, automatically retained

## Backup Schedule

### Execution

- **Primary Scheduled Backups**: Every 2 hours (APScheduler in `api/app.py`)
- **Shutdown Backups**: When application terminates (signal handlers in `api/app.py`)
- **Rotation Execution**: Manual or scheduled (via job queue or external cron) using `backup-rotation.py`

### Filename Format

All backups stored in `${STORAGE_DIR_GENERATED_DATA}/backups/`:

```
judge_db_{label}_{YYYYMMDD_HHMMSS}.sql
```

Examples:
- `judge_db_scheduled_20260314_101500.sql` - Regular 2-hour backup
- `judge_db_shutdown_20260314_155430.sql` - App shutdown backup

### Retention Tiers

| Tier | Frequency | Window | Max Count | Use Case |
|------|-----------|--------|-----------|----------|
| **Recent** | Every 2h | Last 24h | 12 | Point-in-time recovery, last-minute issues |
| **Daily** | 1 per day | 2 weeks (14 days) | 14 | Weekly data loss scenarios |
| **Weekly** | 1 per week | 6 months | ~24 | Monthly/quarterly historical recovery |
| **Monthly** | 1 per month | Forever | Unbounded | Compliance, audit trails, long-term archive |
| **Shutdown** | On app exit | 30 days | N/A | Graceful shutdown checkpoints |

## Implementation

### 2-Hour Scheduled Backups

> **File**: `api/app.py` (lines 248-275)

```python
scheduler.add_job(
    backup_mysql_db,
    "interval",
    hours=2,
    id="mysql_scheduled_backup",
    replace_existing=True
)
scheduler.start()
```

**Key Features**:
- APScheduler handles execution (no cron external dependency)
- Timestamp format includes hours/minutes: `%Y%m%d_%H%M%S`
- Self-contained within Flask app lifecycle
- Logs backup location to stdout for monitoring

### Shutdown Backups

> **File**: `api/app.py` (lines 277-290)

```python
def shutdown_handler(*args):
    print("⚠️ Shutting down Flask app, creating backup...")
    try:
        backup_mysql_db(label="shutdown")
    except Exception as e:
        print(e)
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, shutdown_handler)  # kill
atexit.register(shutdown_handler)                # Python exit
```

**Triggered by**:
- `SIGINT` (Ctrl+C in terminal)
- `SIGTERM` (Docker stop, systemd stop)
- `atexit` (Python interpreter shutdown)

### Backup Rotation

> **Script**: `api/scripts/backup-rotation.py`

Implements tiered retention logic:

```bash
# Dry run (preview what would be deleted)
python api/scripts/backup-rotation.py --backup-dir /data/generated

# Execute deletion
python api/scripts/backup-rotation.py --backup-dir /data/generated --execute
```

**Algorithm**:

1. Parse backup filename for timestamp and label
2. For each backup, calculate age in hours/days
3. Apply decision tree:
   - If `age <= 24 hours`: **KEEP** (recent tier)
   - If `24h < age <= 14 days`: **KEEP** if unique to day (daily tier)
   - If `14 < age <= 180 days`: **KEEP** if Sunday backup (weekly tier)
   - If `age > 180 days`: **KEEP** if day-of-month == 1 (monthly tier)
   - Shutdown backups: **KEEP** if `age <= 30 days`
   - All others: **DELETE**

**Example Output**:

```
🔍 Found 156 backup files in /data/generated/backups
✅ Keep: judge_db_scheduled_20260314_101500.sql (342.5MB) - recent_24h_tier
✅ Keep: judge_db_scheduled_20260312_235959.sql (341.2MB) - daily_tier_day_2026-03-12
✅ Keep: judge_db_scheduled_20260308_000000.sql (339.8MB) - weekly_tier_week_10
✅ Keep: judge_db_scheduled_20260301_000000.sql (336.1MB) - monthly_tier_2026-03
🗑️  Delete: judge_db_scheduled_20260310_153000.sql (340.1MB) - intermediate_day_not_week_end

📊 Summary: Keep 87 backups, delete 69 backups
💾 Space that would be freed: 23,451.3MB (22.9GB)
```

## Storage Management

### Disk Space Estimation

Assuming:
- Average backup size: 350 MB
- Retention: ~87 backups at any time

**Total disk footprint**: ~30.5 GB

> **Recommendation**: Allocate `50-100 GB` for `${STORAGE_DIR_GENERATED_DATA}` to accommodate backups plus generated data (training results, predictions).

### Volume Configuration

Backups stored on the mounted external SSD:

```yaml
volumes:
  - ${STORAGE_DIR_GENERATED_DATA}:${STORAGE_DIR_GENERATED_DATA}
```

**Critical**: Ensure `${STORAGE_DIR_GENERATED_DATA}` is:
- External to container (mounted SSD or NAS)
- Sufficient free space (alert if <20% remaining)
- Backed up regularly (e.g., daily to another location)
- Accessible to both Belgium and USA Docker hosts

## Execution Methods

### Method 1: Scheduled via APScheduler (Default)

No manual intervention required:

```
2-hour interval → mysqldump → ${STORAGE_DIR_GENERATED_DATA}/backups/
```

Status: Enabled by default in production.

### Method 2: Manual Script Execution

Run rotation script on-demand:

```bash
# From within container
docker exec ai-judge-api python api/scripts/backup-rotation.py --execute

# From host
python /path/to/api/scripts/backup-rotation.py --backup-dir=/mnt/data/backups --execute
```

### Method 3: Job Queue Integration

Queue rotation as a job (future enhancement):

```python
from services.jobService import JobService

job_service = JobService()
job_service.launch_backup_rotation_job()
```

## Monitoring & Alerts

### Health Checks

- **Backup Freshness**: Alert if no backup created within 3 hours
- **Disk Space**: Alert if backup volume >80% full or <100 GB free
- **Backup Validity**: Monthly integrity check (restore test to temporary database)
- **Process Health**: Monitor APScheduler thread health

### Log Locations

- **Backup Execution Logs**: Flask application stdout (Docker logs)
- **Rotation Audit Logs**: stdout from `backup-rotation.py`
- **Database Logs**: MySQL error log in container or external volume

### Recommended Monitoring

```bash
# Watch backup directory growth
watch "ls -lh /mnt/data/generated/backups | tail -20"

# Monitor backup file count and total size
watch "ls /mnt/data/generated/backups/*.sql | wc -l && du -sh /mnt/data/generated/backups"

# Check Flask app logs for backup status
docker logs -f ai-judge-api | grep -i backup
```

## Disaster Recovery Examples

### Scenario 1: Restore from Last 24 Hours

**Problem**: Accidental data modification 6 hours ago.  
**Solution**: 
1. Find the 2-hour backup closest to incident time
2. Restore to temporary database: `mysql < judge_db_scheduled_20260314_100000.sql`
3. Export affected tables, merge into production
4. Validate, commit

### Scenario 2: Weekly Rollback

**Problem**: Bad migration deployed 3 days ago, no recent backups available.  
**Solution**:
1. Find the daily backup from 2 days ago (in daily tier)
2. Full restore to temporary database
3. Compare schemas and data with production
4. Restore production using this backup

### Scenario 3: Compliance Audit

**Problem**: Need to demonstrate data state from 6 months ago.  
**Solution**:
1. Monthly tier retains backups forever
2. Restore the month-1 backup: `judge_db_scheduled_20250901_000000.sql`
3. Generate audit report showing historical state

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MYSQL_BACKUP` | Backup directory path (deprecated, use STORAGE_DIR_GENERATED_DATA) | *unset* |
| `STORAGE_DIR_GENERATED_DATA` | Base path for backups (backups go in `{path}/backups/`) | *required* |
| `DELETE_RECOVERY_DAYS` | Soft-delete recovery window (affects DeletedVideos, not backups) | 30 |

### APScheduler Configuration

No external config needed; hardcoded in `api/app.py`:

```python
scheduler.add_job(
    backup_mysql_db,           # Function to call
    "interval",                # Trigger type
    hours=2,                   # Interval
    id="mysql_scheduled_backup",  # Unique ID
    replace_existing=True      # Don't queue duplicates
)
```

## Maintenance & Operations

### Monthly Tasks

1. **Rotation Health Check**
   ```bash
   python api/scripts/backup-rotation.py --backup-dir=/mnt/data/generated --quiet
   # Expected: 85-90 backups kept, ~30 GB used
   ```

2. **Backup Integrity Test**
   ```bash
   # Restore latest backup to temp database
   mysql -u root -p < /mnt/data/generated/backups/$(ls -t | head -1)
   # Verify table counts, indexes, data samples
   ```

3. **Disk Space Audit**
   ```bash
   du -sh /mnt/data/generated/backups/
   df -h /mnt/data/generated
   ```

### Common Operations

**Emergency: Stop APScheduler backups temporarily**
```python
from api.app import scheduler
scheduler.pause()  # Pause all jobs
# Fix issue...
scheduler.resume()
```

**Emergency: Create immediate backup**
```bash
docker exec ai-judge-api python -c "
from api.app import backup_mysql_db
backup_mysql_db(label='emergency_fix')
"
```

**Cleanup: Remove all test backups**
```bash
rm /mnt/data/generated/backups/judge_db_scheduled_202603*_000000.sql
# Warning: Use with caution!
```

## Related Documents

- [AI-Judge.Disaster-Recovery.md](AI-Judge.Disaster-Recovery.md) - Recovery runbooks and escalation
- [AI-Judge.Multi-Region.md](AI-Judge.Multi-Region.md) - Replication architecture
- [AI-Judge.Bidirectional-Sync.md](AI-Judge.Bidirectional-Sync.md) - Data synchronization strategy

