# AI Judge Conflict Resolution Strategy

**Document Status**: Complete (March 2026)  
**Last Updated**: 2026-03-14

## Overview

In a dual-primary MySQL replication environment, simultaneous writes to the same entity can occur when:
- Both regionsacknowledge writes before the other region receives the update
- Network latency causes temporary partition
- Users in Belgium and USA label the same video simultaneously

The AI Judge system uses a **First-Write-Wins** (FWW) strategy to resolve conflicts while maintaining data integrity and providing visibility to users.

## Design Philosophy

1. **Deterministic Resolution**: Timestamp-based winner selection eliminates arbitrary choices
2. **Transparent Audit Trail**: Every conflict logged with full context (ConflictLog table)
3. **User Awareness**: Conflicts visible on video label dashboard, resolvable by users
4. **Data Preservation**: Losing writes stored for inspection, not silently discarded
5. **Graceful Degradation**: System continues operating even if one region is unreachable

## Conflict Detection

### Trigger Scenarios

Conflicts are detected and logged in these situations:

**Scenario A: Simultaneous Frame Label Creation**
```
Belgium:  User A labels frame 100 at 14:32:15.123 UTC
USA:      User B labels frame 100 at 14:32:14.999 UTC
          ↓
Conflict: Same frame, different accounts, simultaneous timestamp windows
Winner:   USA (earlier timestamp by 124ms) keeps label
Loser:    Belgium label rejected, stored in ConflictLog
```

**Scenario B: Conflicting Video Metadata Updates**
```
Belgium:  Admin marks video training=true at 14:45:00
USA:      Admin marks video training=false at 14:44:55
          ↓
Conflict: Video.training has divergent states
Winner:   USA (earlier timestamp) sets training=false
Loser:    Belgium's training=true recorded, user can inspect/resolve
```

**Scenario C: Skill/Prediction Overwrites**
```
Belgium:  Training algorithm generates skill at 15:10:30
USA:      Different ML model generates skill for same timerange at 15:10:25
          ↓
Conflict: Two different skills for overlapping frame ranges
Winner:   USA (earlier timestamp by 5 seconds)
Loser:    Belgium skillinfo JSON archived in ConflictLog
```

## First-Write-Wins Implementation

### Algorithm

```python
# Pseudocode executed on receiving remote update
def apply_remote_update(remote_entity, local_entity):
    if local_entity is None:
        # No local version, apply remote
        return apply(remote_entity)
    
    # Both exist: compare timestamps
    if remote_entity.updatedAt < local_entity.updatedAt:
        # Local is newer → local wins, log conflict
        log_conflict(
            entity_type=type(remote_entity),
            entity_id=remote_entity.id,
            winning_region=LOCAL_REGION,
            winning_timestamp=local_entity.updatedAt,
            losing_region=REMOTE_REGION,
            losing_timestamp=remote_entity.updatedAt,
            losing_data=remote_entity.to_dict()
        )
        return keep_local(local_entity)
    else:
        # Remote is newer or equal → remote wins
        log_conflict(...)
        return apply(remote_entity)
```

### Data Model: ConflictLog

> **File**: `api/repository/models.py` (lines 314-355)

```python
class ConflictLog(DomainObject):
    """Tracks first-write-wins conflicts in dual-primary replication"""
    __tablename__ = 'ConflictLogs'
    
    entity_type = db.Column(db.String(50), nullable=False)  # 'Video', 'FrameLabel', 'Skill'
    entity_id = db.Column(UUIDType, nullable=False, index=True)
    
    # Winning update (kept)
    winning_account_id = db.Column(UUIDType, nullable=False)
    winning_region = db.Column(db.String(50), nullable=False)
    winning_timestamp = db.Column(db.DateTime, nullable=False)
    
    # Losing update (archived)
    losing_account_id = db.Column(UUIDType, nullable=True)
    losing_region = db.Column(db.String(50), nullable=False)
    losing_timestamp = db.Column(db.DateTime, nullable=False)
    losing_data = db.Column(JSON, nullable=False)  # Full entity as JSON
    
    # Support for user resolution
    conflict_description = db.Column(db.String(255), nullable=False)
    is_resolved = db.Column(db.Boolean, default=False)
    resolved_by = db.Column(UUIDType, nullable=True)  # Account that resolved
    resolved_at = db.Column(db.DateTime, nullable=True)
    resolution_note = db.Column(db.String(255), nullable=True)
```

## User Experience

### Conflict Notification

**Video Label Dashboard** displays unresolved conflicts:

```
⚠️  CONFLICT: Frame Label Sync Issue

  Entity: FrameLabel in Video "jump_technique_01.mp4"
  Frame #: 100
  
  KEPT (Belgium, 14:32:15):
  - Account: john.smith@belgium.ai-judge.be
  - Label Position: x=245, y=120, w=80, h=150
  - Jumper Visible: true
  
  REJECTED (USA, 14:32:14):
  - Account: jane.doe@usa.ai-judge.be
  - Label Position: x=242, y=118, w=75, h=145
  - Jumper Visible: true
  
  ACTION REQUIRED:
  □ Reject Belgium version, use USA  
  □ Keep Belgium version (current)
  □ Manually merge (edit now)
  
  Account Dashboard → Conflict Resolution (1 pending)
```

### Resolution Workflow

1. **Identify**: System flags conflict, notifies involved users
2. **Review**: Users inspect both versions on dashboard
3. **Decide**: Choose winner, or manually edit to merge
4. **Mark Resolved**: Click "Resolved" button with optional note
5. **Audit**: ConflictLog table records user decision

## Implementation in Code

### Logging Conflicts

> **File**: `api/services/jobService.py` (lines 82-99)

```python
def log_conflict(self, entity_type: str, entity_id: UUID, 
                  winning_account_id: UUID, winning_region: str, 
                  winning_timestamp: datetime, losing_account_id: UUID,
                  losing_region: str, losing_timestamp: datetime, 
                  losing_data: dict, description: str) -> ConflictLog:
    """Log a first-write-wins conflict"""
    conflict = ConflictLog(
        entity_type=entity_type,
        entity_id=entity_id,
        winning_account_id=winning_account_id,
        winning_region=winning_region,
        winning_timestamp=winning_timestamp,
        losing_account_id=losing_account_id,
        losing_region=losing_region,
        losing_timestamp=losing_timestamp,
        losing_data=losing_data,
        conflict_description=description,
        is_resolved=False
    )
    db.session.add(conflict)
    db.session.commit()
    return conflict
```

### Retrieving Unresolved Conflicts

```python
def get_unresolved_conflicts(self, account_id: UUID = None) -> List[ConflictLog]:
    """Get unresolved conflicts, optionally filtered by account"""
    query = db.session.query(ConflictLog).filter_by(is_resolved=False)
    if account_id:
        query = query.filter_by(winning_account_id=account_id)
    return query.all()
```

### Integration with Sync Jobs

Sync job manager executes conflict checks before applying remote updates:

> **File**: `api/scripts/sync-job-manager.py`

```python
def apply_remote_update(entity):
    """Before syncing remote entity, check for conflicts"""
    local = db.session.query(Video).filter_by(id=entity.id).first()
    
    if local and local.updatedAt > entity.updatedAt:
        # Local is newer, keep local, log conflict
        job_service.log_conflict(
            entity_type='Video',
            entity_id=entity.id,
            winning_region='belgium',
            winning_timestamp=local.updatedAt,
            losing_region='usa',
            losing_timestamp=entity.updatedAt,
            losing_data=entity.to_dict(),
            description=f"Video '{entity.name}' updated in both regions"
        )
        return False  # Don't apply remote
    
    # Remote is newer or no conflict
    apply(entity)
    return True
```

## Conflict Statistics & Monitoring

### Query Unresolved Conflicts

```sql
SELECT 
    entity_type,
    COUNT(*) as conflict_count,
    MIN(created_at) as oldest_conflict
FROM ConflictLogs
WHERE is_resolved = FALSE
GROUP BY entity_type
ORDER BY conflict_count DESC;
```

### Example Output

```
entity_type     conflict_count  oldest_conflict
FrameLabel      3               2026-03-12 14:32:15
Skill           1               2026-03-11 10:15:00
Video           0               NULL
```

### Dashboard Metrics

- **Unresolved Conflicts**: Count by region and account
- **Resolution Time**: Median time from conflict to resolution
- **Conflict Rate**: Conflicts per 1000 sync operations
- **Busiest Hours**: When conflicts most frequently occur

## Edge Cases & Handling

### Case 1: Cascading Deletes

**Scenario**: Video deleted in Belgium, but being sync'd to USA
```
Belgium: DELETE Video X
USA:     Video X.updatedAt > Belgium DELETE timestamp
→
Result: USA keeps Video X (FWW applies to updates, not deletes)
Expected: Allow USA to keep, let deletion replicate normally
```

### Case 2: NULL vs Value Conflict

**Scenario**: One region sets field, other sets to NULL
```
Belgium: Video.training_model = "MViT"
USA:     Video.training_model = NULL
→
Result: Specific value wins (longer timestamp = more recent edit)
```

### Case 3: Rapid Re-edits

**Scenario**: User in Belgium edits, USA rejects, Belgium re-edits
```
Time 14:30:00  Belgium: FrameLabel.x = 100
Time 14:30:05  USA:     FrameLabel.x = 110 (wins, timestamp-based)
Time 14:30:10  Belgium: FrameLabel.x = 105 (conflicts with USA again)
→
Result: Each conflict independently logged, user decides final state
```

## Performance Implications

### Database Impact

**New Tables**: 
- `ConflictLogs`: Typically 0.1-1% of Video/FrameLabel volume
- **Storage**: ~300 bytes per conflict, negligible overhead

**Query Performance**:
- Conflict logging: O(1) insert with foreign key indexes
- Conflict retrieval: O(n) scan filtered by account, typical n<100

### Replication Impact

- **None**: FWW resolution happens post-sync, doesn't affect MySQL replication
- Losing data stored in JSON, not replicated

## Related Concepts

### vs. Merge-Based Conflict Resolution

**Why NOT automatic merge?**
- Frame labels can't be semantically merged (can't combine two positions)
- Unsafe to assume both versions correct
- User intent important (which version did they want?)

**Why First-Write-Wins?**
- Simple, deterministic, auditable
- Preserves data intent (earlier decision takes precedence)
- Fails gracefully (always a clear winner)

### vs. Operational Transformation (OT)

**Why NOT OT/CRDTs?**
- OT requires coordination layer (defeats multi-region async model)
- CRDTs suitable for text editing (frame positions aren't commutative)
- Complexity not justified for infrequent conflicts

## Operational Runooks

### Inspect a Conflict

```sql
SELECT 
    id,
    entity_type,
    entity_id,
    winning_region,
    losing_region,
    losing_data,
    conflict_description,
    created_at
FROM ConflictLogs
WHERE is_resolved = FALSE
LIMIT 5;
```

### Review Losing Data

```python
import json
conflict = ConflictLog.query.first()
print(json.dumps(conflict.losing_data, indent=2))
```

### Manually Resolve

```python
conflict = ConflictLog.query.filter_by(id='...).first()
conflict.is_resolved = True
conflict.resolved_by = current_user.id
conflict.resolution_note = "User confirmed Belgium version was stale, kept USA"
conflict.resolved_at = datetime.now()
db.session.commit()
```

## Future Enhancements

1. **Auto-Resolution Rules**: Define patterns (e.g., "always keep USA timestamps")
2. **Conflict Prediction**: Identify high-risk entities before conflict occurs
3. **Weighted Win**: Consider account privilege (admin wins over contributor)
4. **Semantic Merge**: For non-conflicting fields in same entity
5. **User Prompts**: Real-time notifications when editing involves conflict risk

## Related Documents

- [AI-Judge.Multi-Region.md](AI-Judge.Multi-Region.md) - Primary-primary replication setup  
- [AI-Judge.Bidirectional-Sync.md](AI-Judge.Bidirectional-Sync.md) - How conflicts are detected during sync
- [AI-Judge.Disaster-Recovery.md](AI-Judge.Disaster-Recovery.md) - Escalation if conflict resolution fails

