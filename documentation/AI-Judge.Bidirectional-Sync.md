# AI Judge Bidirectional Sync Strategy

**Document Status**: Complete (March 2026)  
**Last Updated**: 2026-03-14

## Overview

Multi-region deployment requires synchronization of video files and generated data across regions. The **Bidirectional Sync** system uses rsync over SSH to keep regions in sync after processing completes.

Unlike MySQL replication (which handles metadata automatically), video files and training artifacts are too large to afford constant synchronization and require selective, event-driven syncing.

**Region Flexibility**: The system supports N regions (not just 2), with flexible configuration in `.env` using region-specific host definitions.

## Design Principles

1. **Selective Sync**: Only sync videos that have labels, not all videos
2. **Database-Driven**: Query database to determine sync eligibility, not directory structure
3. **Async Operation**: Sync via job queue, not blocking web requests
4. **Multi-Region**: Support 2+N regions with configurable hosts
5. **Health-Aware**: Check remote server accessibility before attempting sync
6. **Change Detection**: Detect and sync moved/renamed files
7. **Efficient Transfer**: Rsync checksums avoid duplicate transfers
8. **Documented Recovery**: Hard deletes sync after 30-day recovery window

## Storage Organization

### Account-Based Folder Structure

Videos are organized by account UUID in the root STORAGE_DIR:

```
${STORAGE_DIR_VIDEOS}/
├── {account-uuid-1}/          # Account that uploaded the videos
│   ├── folder-{uuid}/
│   │   ├── video-{uuid}.mp4
│   │   ├── video-{uuid}.mp4
│   │   └── ...
│   └── folder-{uuid}/
└── {account-uuid-2}/          # Different account's videos
    └── folder-{uuid}/
        └── video-{uuid}.mp4
```

**Benefits**:
- Permissions enforcement: Can sync only account's own videos
- Quota management: Track per-account storage usage
- Multi-tenancy: Organize by data owner (future grouping)
- Cross-region consistency: Same hierarchy in all regions

### Generated Data Structure

```
${STORAGE_DIR_GENERATED_DATA}/
├── logs/
│   ├── api/
│   │   ├── belgium/
│   │   │   ├── api-2026-03-14.log
│   │   │   └── sync-manager-2026-03-14.log
│   │   └── usa/
│   │       └── api-2026-03-14.log
│   └── backups/                # Rotation logs
├── backups/
│   ├── judge_db_scheduled_*.sql
│   └── judge_db_shutdown_*.sql
└── {video-uuid}/
    ├── predictions/            # Synced after PREDICT job
    │   ├── predictions.json
    │   └── confidence_map.tif
    └── training_results/       # Synced after TRAIN job
        ├── model.pt
        └── metrics.json
```

## Data Sync Triggers

### Labeled Videos Sync

**When**: When a video transitions from unlabeled to labeled (first frame label added)

**What**: Entire video directory from `${STORAGE_DIR_VIDEOS}/{account_id}/{video_directory}`

**Size**: 360 MB to 5 GB per 2-hour video

**Trigger Logic**:
```python
@app.route('/api/videos/<video_id>/labels', methods=['POST'])
def add_frame_label(video_id):
    label = create_label(video_id, request.json)
    
    # Check if this was the FIRST label for this video
    video = Video.query.get(video_id)
    label_count_before = FrameLabel.query.filter_by(video_id=video_id).count()
    
    if label_count_before == 1:  # This is the first label
        # Queue sync to all other regions
        for target_region in get_all_other_regions(REGION):
            job_service.launch_sync_job(
                source_region=REGION,
                target_region=target_region,
                data_type='labeled_videos',
                video_id=video_id,
                account_id=video.uploadedBy
            )
    
    return {"id": label.id}
```

### Generated Data Sync

**When**: After TRAIN or PREDICT job completes

**What**: Generated files in `${STORAGE_DIR_GENERATED_DATA}/{video_id}/`

**Size**: 50-500 MB per video depending on model outputs

**Trigger Logic**:
```python
def on_job_complete(job_id):
    job = Job.query.get(job_id)
    
    if job.status == 'Completed' and job.job_category == 'AI':
        video_id = job.job_arguments.get('video_id')
        
        # Sync generated data to all other regions
        for target_region in get_all_other_regions(REGION):
            job_service.launch_sync_job(
                source_region=REGION,
                target_region=target_region,
                data_type='generated_data',
                video_id=video_id
            )
```

### Hard Delete Sync

**When**: After 30-day soft-delete recovery window expires

**What**: Removal of video files and generated data

**Trigger Logic**:
```python
def process_expired_soft_deletes():
    """Batch job that checks for expired soft-deleted videos"""
    expired = DeletedVideo.query.filter(
        DeletedVideo.recovery_deadline < datetime.now(),
        DeletedVideo.is_hard_deleted == False
    ).all()
    
    for deleted_video in expired:
        # Hard delete local files
        hard_delete_video_files(deleted_video.video_id)
        
        # Update record
        deleted_video.is_hard_deleted = True
        deleted_video.hard_deleted_by = SYSTEM_ACCOUNT_ID
        
        # Queue sync to other regions (they'll hard delete too)
        for target_region in get_all_other_regions(REGION):
            job_service.launch_sync_job(
                source_region=REGION,
                target_region=target_region,
                data_type='deletions',
                video_id=deleted_video.video_id
            )
    
    db.session.commit()
```

## Database-Driven Sync Selection

### Why Not Folder-Based Filtering?

❌ **Folder-based approach** (REJECTED):
```bash
rsync --filter='+labeled_videos/' --filter='-unlabeled_videos/' ...
# Problems:
# - Requires duplicating folder structure to track state
# - Sync happens blindly ("if file exists, it's labeled")
# - Impossible to detect renames/moves
# - No cross-region consistency mechanism
```

✅ **Database-driven approach** (IMPLEMENTED):
```python
# Query database for labeled status
video = Video.query.get(video_id)
if video.has_frame_labels():  # Check database, not filesystem
    # Sync this video
    rsync(f"{account_id}/{video_folder}", remote_host, ...)
```

### Labeled Video Detection

```python
def should_sync_video(video_id: UUID) -> bool:
    """Determine if video should be synced to other regions"""
    video = Video.query.get(video_id)
    
    if not video:
        return False
    
    # Video is synced if it has ANY frame labels
    label_count = FrameLabel.query.filter_by(video_id=video_id).count()
    return label_count > 0

def get_videos_requiring_sync(account_id: UUID = None) -> List[Video]:
    """Get all labeled videos that haven't been synced yet"""
    labeled_videos = db.session.query(Video).join(FrameLabel).distinct()
    
    if account_id:
        labeled_videos = labeled_videos.filter(Video.uploadedBy == account_id)
    
    return labeled_videos.all()
```

### Detecting Moved/Renamed Videos

Videos may be moved or renamed locally. Sync manager detects and handles these:

```python
def detect_moved_videos(video_id: UUID) -> Optional[Path]:
    """Find where a video's file actually is"""
    # Query database for current account/folder
    video = Video.query.get(video_id)
    folder = Folder.query.get(video.folderId)
    account_id = video.uploadedBy
    
    expected_path = Path(
        STORAGE_DIR_VIDEOS, 
        str(account_id),
        str(folder.id),
        video.name
    )
    
    if expected_path.exists():
        return expected_path
    
    # File moved/renamed; search by video ID in metadata
    for candidate in Path(STORAGE_DIR_VIDEOS).rglob('*.mp4'):
        # Extract video ID from file metadata
        metadata = get_video_metadata(candidate)
        if metadata.get('video_id') == str(video_id):
            return candidate
    
    # Not found
    return None
```

## Multi-Region Configuration

### Flexible Region Setup

Define regions dynamically in `.env`:

```bash
# List of all regions (comma-separated)
ALL_REGIONS=belgium,usa,germany,japan

# Host for each region (SSH accessible)
REGION_BELGIUM_HOST=sync@belgium.example.com
REGION_USA_HOST=sync@usa.example.com
REGION_GERMANY_HOST=sync@germany.example.com
REGION_JAPAN_HOST=sync@japan.example.com

# Currently running in:
REGION=belgium
MYSQL_SERVER_ID=1
```

### Sending to All Other Regions

```python
def get_all_other_regions(current_region: str) -> List[str]:
    """Get list of all regions except current"""
    all_regions = os.getenv('ALL_REGIONS', 'belgium,usa').split(',')
    return [r.strip() for r in all_regions if r.strip() != current_region]

# Usage: When labeling video in belgium, sync to all others
for target_region in get_all_other_regions('belgium'):
    job_service.launch_sync_job(
        source_region='belgium',
        target_region=target_region,
        data_type='labeled_videos',
        video_id=video_id
    )
# Launches: belgium → usa, belgium → germany, belgium → japan
```

## Server Health Checks

### Pre-Sync Validation

Before attempting rsync, verify remote servers are accessible:

```python
def check_remote_accessibility(region: str, timeout_seconds: int = 10) -> bool:
    """Check if remote region is reachable via SSH"""
    host_var = f"REGION_{region.upper()}_HOST"
    remote_host = os.getenv(host_var)
    
    if not remote_host:
        logger.error(f"No host defined for region {region}")
        return False
    
    try:
        # Simple SSH connectivity check
        result = subprocess.run(
            ['ssh', '-o', f'ConnectTimeout={timeout_seconds}', 
             remote_host, 'echo "OK"'],
            capture_output=True,
            timeout=timeout_seconds + 2
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"SSH timeout connecting to {remote_host}")
        return False
    except Exception as e:
        logger.warning(f"Cannot reach {region} ({remote_host}): {e}")
        return False
```

### Graceful Queuing on Unavailability

```python
def execute_sync_job(job: Job) -> bool:
    """Execute sync with health checks"""
    args = job.job_arguments
    target_region = args['target_region']
    
    # Check if target region is reachable
    if not check_remote_accessibility(target_region):
        logger.info(f"Target region {target_region} unavailable, requeueing...")
        job.status = 'Pending'  # Requeue for retry
        job.status_details = f"Target region {target_region} unreachable"
        db.session.commit()
        return False
    
    # Proceed with rsync
    try:
        success = execute_rsync(args)
        job.status = 'Completed' if success else 'Failed'
        return success
    except Exception as e:
        job.status = 'Failed'
        job.status_details = str(e)
        logger.error(f"Sync failed: {e}")
        return False
    finally:
        db.session.commit()
```

## Data Sync Categories

### Category A: Labeled Videos

**When**: After first frame label added to a video  
**What**: `/STORAGE_DIR_VIDEOS/{video_id}/` directory  
**Size**: Varies (360 MB to 5 GB per video for 2-hour clips)  
**Priority**: High (needed for training)  
**Sync Mode**: Bidirectional (label could happen in either region)

**Trigger**:
```python
# In video label API endpoint
@app.route('/api/videos/<video_id>/labels', methods=['POST'])
def add_label(video_id):
    # ... create label ...
    
    # Launch sync job
    if not video_has_labels(video_id):  # First label
        job_service.launch_sync_job(
            source_region=REGION,  # 'belgium' or 'usa'
            target_region='usa' if REGION == 'belgium' else 'belgium',
            data_type='labeled_videos',
            video_ids=[video_id]
        )
    return {...}
```

### Category B: Generated Data

**When**: After training/prediction job completes  
**What**: `/STORAGE_DIR_GENERATED_DATA/{video_id}/predictions/`  
**Size**: 50-500 MB per video (depending on model outputs)  
**Priority**: High (needed for next stage of processing)  
**Sync Mode**: Unidirectional (generated in one region, synced to other)

**Trigger**:
```python
# In training/prediction job completion
def on_job_complete(job_id):
    job = Job.query.get(job_id)
    
    if job.type in ['TRAIN', 'PREDICT']:
        job_service.launch_sync_job(
            source_region=REGION,
            target_region='usa' if REGION == 'belgium' else 'belgium',
            data_type='generated_data',
            video_ids=[job.job_arguments['video_id']]
        )
```

### Category C: Hard Deletes (Post-Recovery)

**When**: After 30-day soft-delete recovery window expires  
**What**: Video files and generated data marked for deletion  
**Size**: Same as original (can free 500 MB to 5 GB per video)  
**Priority**: Medium (storage reclamation, not data access)  
**Sync Mode**: Unidirectional (source region initiates hard delete, syncs to target)

**Trigger**:
```python
# In scheduled job that checks DeletedVideo.recovery_deadline
def hard_delete_expired_videos():
    expired = DeletedVideo.query.filter(
        DeletedVideo.recovery_deadline < datetime.now()
    ).filter_by(is_hard_deleted=False)
    
    for deleted_video in expired:
        # Perform hard delete (remove files)
        hard_delete_video_files(deleted_video.video_id)
        
        # Mark as hard deleted
        deleted_video.is_hard_deleted = True
        
        # Sync deletion to other region
        job_service.launch_sync_job(
            source_region=REGION,
            target_region='usa' if REGION == 'belgium' else 'belgium',
            data_type='deletions'
        )
    
    db.session.commit()
```

## Sync Execution Model

### Job Queue Integration

Sync jobs are queued like any other job, with special filtering for executors:

> **File**: `api/repository/models.py` (line 370)

```python
class Jobs(DomainObject):
    __tablename__ = 'Jobs'
    type = db.Column(db.String(30), nullable=False)  # 'SYNC', 'TRAIN', 'PREDICT'
    job_category = db.Column(db.String(20), default='AI')  # 'AI', 'SYNC', or 'BACKUP'
    # ... other fields ...
```

**Three Executor Types**:

1. **AI Executor** (Computer Vision)
   - Filters: `job_category == 'AI'`
   - Processes: TRAIN, PREDICT jobs
   - Ignores: SYNC, BACKUP jobs

2. **Sync Executor** (rsync manager)
   - Filters: `job_category == 'SYNC'`
   - Processes: SYNC jobs
   - Ignores: AI, BACKUP jobs

3. **Backup Executor** (rotation manager)
   - Filters: `job_category == 'BACKUP'`
   - Processes: BACKUP jobs
   - Ignores: AI, SYNC jobs

### Launching a Sync Job

> **File**: `api/services/jobService.py` (lines 62-77)

```python
def launch_sync_job(self, source_region: str, target_region: str, 
                     data_type: str, video_ids: List[UUID] = None) -> Job:
    """Launch a bidirectional sync job"""
    sync_job = Job(
        type='SYNC',
        step=f'sync_{data_type}',
        status='Created',
        job_category='SYNC',  # Filtered to sync executor only
        job_arguments={
            'source_region': source_region,
            'target_region': target_region,
            'data_type': data_type,
            'video_ids': [str(vid) for vid in (video_ids or [])]
        }
    )
    self.launch_job(sync_job)
    return sync_job
```

## Synchronization Implementation

### Rsync-Based Transfer

> **File**: `api/scripts/sync-job-manager.py`

Rsync provides:
- **Checksum-based transfer**: Only modified files synchronized
- **Bandwidth efficient**: Compression during transfer
- **Partial file recovery**: Resume on network interruption
- **Historical preservation**: Timestamps and permissions maintained

**Rsync Command Structure**:

```bash
rsync \
  --archive              # Preserve attributes
  --verbose              # Show progress
  --delete               # Delete files not in source
  --compress             # Compression
  --checksum             # Use checksum (not mtime)
  --partial              # Keep partial files
  --progress             # Display transfer progress
  --exclude='*.tmp'      # Skip temporary files
  /source/path/          # Trailing slash = sync contents
  remote-host:/target/path/
```

**Example Sync Session**:

```
[belgium-api] Syncing labeled videos: belgium → usa
Building file list...
sending incremental file list
videos/vid-001/labeled_videos/
videos/vid-001/labeled_videos/frame_labels.csv
                4.3K 100%   42.0 MB/s   0:00:00
videos/vid-001/labeled_videos/video.mp4
            2.1G  92%   45.0 MB/s   0:01:30

Total: 2.1GB processed in 1:45
✅ Rsync completed: labeled_videos (belgium → usa)
```

### Sync Executor Loop

Sync executor runs independently, checking for SYNC jobs:

```python
# sync-job-manager.py main loop
while ENABLE_SYNC_JOBS:
    # Get pending SYNC jobs
    sync_jobs = job_service.get_sync_jobs()
    
    for job in sync_jobs:
        args = job.job_arguments
        
        if args['data_type'] == 'labeled_videos':
            success = sync_labeled_videos(
                args['source_region'],
                args['target_region'],
                args['video_ids']
            )
        elif args['data_type'] == 'generated_data':
            success = sync_generated_data(...)
        elif args['data_type'] == 'deletions':
            success = sync_hard_deletes(...)
        
        job.status = 'Completed' if success else 'Failed'
        db.session.commit()
    
    sleep(60)  # Check every minute
```

**Execution**: Run as separate service (container or process)

```bash
# Start sync executor
python api/scripts/sync-job-manager.py --mode=jobs --execute

# Or in docker-compose
services:
  sync-executor:
    container_name: ai-judge-sync
    build: ./api
    env_file: .env
    depends_on: [api, mysqldb]
    networks: [ai-judge-network]
    volumes: [${STORAGE_DIR_VIDEOS}:${STORAGE_DIR_VIDEOS}, ...]
    command: python api/scripts/sync-job-manager.py --mode=jobs --execute
```

## Wait Time for Async Processing

### Problem

Sync jobs launch after AI jobs complete, but files still be written:

```
14:30:00 - Predict job starts → video.mp4 created
14:30:30 - Predict job completes → triggers sync job
14:30:31 - Sync job checks files → only 50% written (concurrent write)
14:30:35 - Video write finishes
14:35:00 - Next sync finds new files, re-syncs (inefficient)
```

### Solution: Configurable Wait

> **Environment**: `.env`

```
ENABLE_SYNC_JOBS=true
SYNC_WAIT_MINUTES_FOR_AI_JOBS=5
```

**Implementation**:

```python
def should_launch_sync_after_ai(job_type: str) -> bool:
    """Check if enough time passed since AI job completion"""
    wait_minutes = int(os.getenv('SYNC_WAIT_MINUTES_FOR_AI_JOBS', 5))
    
    if job_type in ['TRAIN', 'PREDICT']:
        created_ago = (datetime.now() - job.updated_at).total_seconds() / 60
        
        if created_ago < wait_minutes:
            # Too soon, don't sync yet
            return False
    
    return True
```

**Benefits**:
- Ensures files fully written before rsync starts
- Reduces network traffic from partial syncs
- Configurable for different workload patterns

## Conflict-Free Transfers

### Scenario: Network Partition During Sync

**Timeline**:
```
14:30:00 Belgium   User A labels frame 100
14:30:05 Belgium   Sync job starts → rsync begins
14:30:10 USA       User B labels same frame 100
14:30:15 Belgium   Video file partially synced to USA
14:30:20 Network disconnects Belgium-USA
14:30:25 USA       Rsync timeout, sync job fails
14:30:30 Belgium   continues (unaware of USA update)
14:35:00 Network reconnects
```

**Resolution via ConflictLog**:
1. When USA update reaches Belgium via replication → ConflictLog records timestamp difference
2. Next successful sync only transfers missing files (rsync checksum detects partial)
3. Users resolve conflict via dashboard (First-Write-Wins)

**No data loss**: Both versions preserved in ConflictLog, users can inspect and decide.

## Monitoring Sync Jobs

### SQL Queries

**Pending sync jobs**:
```sql
SELECT id, type, step, status, job_arguments, created_at
FROM Jobs
WHERE job_category = 'SYNC'
AND status = 'Created'
ORDER BY created_at ASC;
```

**Recent sync completions**:
```sql
SELECT id, type, step, status, updated_at, 
       JSON_EXTRACT(job_arguments, '$.data_type') as data_type
FROM Jobs
WHERE job_category = 'SYNC'
AND status = 'Completed'
ORDER BY updated_at DESC
LIMIT 10;
```

**Failed syncs**:
```sql
SELECT id, type, step, status_details, updated_at
FROM Jobs  
WHERE job_category = 'SYNC'
AND status = 'Failed'
ORDER BY updated_at DESC;
```

### Dashboard Metrics

- **Sync Queue Depth**: Pending jobs by data type
- **Sync Latency**: Time from upload to both-regions synchronized  
- **Transfer Rate**: Bytes transferred per minute
- **Error Rate**: Failed syncs per 1000 attempts

### Example Monitoring Script

```bash
#!/bin/bash
# Monitor sync job health

while true; do
    echo "=== Sync Job Status ==="
    docker exec ai-judge-api mysql -e "
    SELECT 
        JSON_EXTRACT(job_arguments, '$.data_type') as data_type,
        status,
        COUNT(*) as count,
        MAX(updated_at) as latest
    FROM Jobs
    WHERE job_category = 'SYNC'
    GROUP BY data_type, status
    ORDER BY data_type, status;
    "
    
    echo ""
    echo "Total queue depth: $(docker exec ai-judge-api mysql -sNe "
    SELECT COUNT(*) FROM Jobs WHERE job_category='SYNC' AND status='Created'
    ")"
    
    sleep 60
done
```

## Configuration Reference

### Environment Variables

| Variable                        | Purpose                                           | Default       |
| ------------------------------- | ------------------------------------------------- | ------------- |
| `ALL_REGIONS`                   | Comma-separated list of all regions               | `belgium,usa` |
| `REGION_{REGION}_HOST`          | SSH host for sync ({REGION} = belgium, usa, etc.) | *unset*       |
| `ENABLE_SYNC_JOBS`              | Enable/disable sync executor                      | `false`       |
| `SYNC_WAIT_MINUTES_FOR_AI_JOBS` | Wait before syncing after PREDICT/TRAIN           | `5`           |
| `STORAGE_DIR_VIDEOS`            | Video storage path (organized by account UUID)    | *required*    |
| `STORAGE_DIR_GENERATED_DATA`    | Training results & logs path                      | *required*    |

### Example for 4-Region Setup

```bash
ALL_REGIONS=belgium,usa,germany,japan
REGION=belgium
MYSQL_SERVER_ID=1

REGION_BELGIUM_HOST=sync@be-gateway.jump.local
REGION_USA_HOST=sync@usa-gateway.jump.local
REGION_GERMANY_HOST=sync@de-gateway.jump.local
REGION_JAPAN_HOST=sync@jp-gateway.jump.local
```

### Environment Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `ENABLE_SYNC_JOBS` | Enable/disable sync executor | false | true |
| `SYNC_WAIT_MINUTES_FOR_AI_JOBS` | Wait before syncing after PREDICT/TRAIN | 5 | 10 |
| `REGION_BELGIUM_HOST` | SSH host for Belgium region | *unset* | sync@be.jump.local |
| `REGION_USA_HOST` | SSH host for USA region | *unset* | sync@usa.jump.local |
| `STORAGE_DIR_VIDEOS` | Video storage path (synced after label) | *required* | /data/videos |
| `STORAGE_DIR_GENERATED_DATA` | Training results path (synced after predict) | *required* | /data/generated |

### Rsync-Specific Settings

Hardcoded in `sync-job-manager.py`:

```python
RSYNC_ARGS = {
    'archive': True,        # Preserve permissions, timestamps
    'compress': True,       # Enable compression
    'checksum': True,       # Use content checksums (slower but accurate)
    'partial': True,        # Keep partial files
    'timeout': 3600,        # 1 hour per rsync invocation
    'exclude': [
        '*.tmp',            # Skip temp files
        '*.lock',           # Skip lock files
        '__pycache__',      # Skip Python cache
    ]
}
```

## Manual Sync Operations

### Dry-Run Sync (Preview What Would Transfer)

```bash
python api/scripts/sync-job-manager.py \
  --mode labeled-videos \
  --source-region belgium \
  --target-region usa
  # (No --execute flag = dry run)
```

### Execute Specific Sync

```bash
python api/scripts/sync-job-manager.py \
  --mode generated-data \
  --source-region usa \
  --target-region belgium \
  --execute
```

### Check Sync Status

```bash
# In-database
mysql -e "
SELECT id, step, status, updated_at, status_details
FROM Jobs
WHERE job_category='SYNC'
ORDER BY updated_at DESC LIMIT 5;
"

# From logs
docker logs -f ai-judge-sync | grep -i sync
```

## Performance Characteristics

### Transfer Speed

- **Network Speed**: 1 Gigabit/s (typical cloud inter-region)
- **Rsync Overhead**: ~15-20% due to checksums
- **Effective Speed**: ~85-100 MB/s

### Time Estimates

| Data Size | Time |
|-----------|------|
| 500 MB (small video) | ~5-6 min |
| 2 GB (typical 2h video) | ~20-25 min |
| 10 GB (long training results) | ~100-120 min |

### Storage Requirements

**Both regions must have**:
- `${STORAGE_DIR_VIDEOS}`: Sum of all video files (~500 GB-2 TB)
- `${STORAGE_DIR_GENERATED_DATA}`: Training/prediction artifacts (~100-300 GB)

**Total per region**: ~1-2.5 TB minimum

## Related Documents

- [AI-Judge.Backup-Retention.md](AI-Judge.Backup-Retention.md) - Backup strategy, different from file sync
- [AI-Judge.Conflict-Resolution.md](AI-Judge.Conflict-Resolution.md) - Handling simultaneous updates
- [AI-Judge.Multi-Region.md](AI-Judge.Multi-Region.md) - MySQL replication (metadata sync)
- [AI-Judge.Disaster-Recovery.md](AI-Judge.Disaster-Recovery.md) - Recovery procedures if sync fails

