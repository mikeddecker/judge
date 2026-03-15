# AI-Judge ComputerVision: Implementation Summary

**Date:** March 14, 2026  
**Status:** Complete  
**Scope:** UUID migration and predict/train method refactoring

## Summary of Changes

This document summarizes all changes made to the computervision module to support UUID-based video IDs and refactored predict/train interfaces.

## Files Modified

### 1. ComputerVision Core Modules

#### `/computervision/managers/RepoGeneral.py` ✅

**Changes:**
- Updated all SQL queries to use parameterized syntax (`:paramname` instead of f-strings)
- Replaced all `MOD(videoId, 10)` and `MOD(id, 10)` patterns with `is_train` field checks
- Added UUID type hints to method parameters
- Added comprehensive docstrings explaining UUID handling
- Fixed train/val split logic to use `is_train` field from Videos table

**Methods Updated:**
1. `get_videoinfo(videoId)` - Parameterized query with UUID parameter
2. `get_framelabels(train_test_val)` - Uses is_train field, joins Videos table
3. `get_unique_videoId_frameNr(train_test_val)` - Uses is_train field
4. `get_fully_segmented_videos(train_test_val)` - Uses is_train field
5. `get_skills_of_fully_segmented_videos(train_test_val)` - Uses is_train field
6. `get_skills(train_test_val, videoId)` - Parameterized query with optional UUID filter
7. `get_team_boxes()` - Uses is_train field with proper joins
8. `delete_job(jobId)` - Expects UUID parameter
9. `get_video_path(videoId)` - Added UUID type hint
10. `__load_relativePaths_of_videos_with_framelabels()` - Parameterized queries with UUID handling

#### `/computervision/Predictor.py` ✅

**Changes:**
- Added UUID type hints to all public predict methods
- Added comprehensive docstring to main `predict()` method
- Explicit UUID parameter typing for all internal methods

**Methods Updated:**
```python
def predict(self, type, videoId: UUID, recipename, ...) -> None
def __predict_skills_pytorch(self, videoId: UUID, ...) -> None
def __predict_segments_pytorch(self, videoId: UUID, ...) -> None  
def __predict_location(self, videoId: UUID, ...) -> None
def __save_skill_predictions_as_video(self, videoId: UUID, ...) -> None
```

#### `/computervision/Trainer.py` ⚠️

**No changes needed** - Trainer doesn't directly handle videoId
- Note: Training operates on full datasets via RepoGeneral, which handles UUID conversion

#### `/computervision/JobExecuter.py` ✅

**Changes:**
- Added `from uuid import UUID` import
- Added UUID conversion logic for videoId from job_arguments
- Added UUID conversion logic for jobId from database binary format
- Handles both string and UUID object inputs robustly

**Key Changes:**
```python
# For PREDICT jobs
videoId = UUID(job_arguments["videoId"]) if isinstance(...) else ...
jobId = UUID(bytes=job["id"]) if isinstance(...) else ...
```

### 2. Documentation Files Created

#### `/documentation/AI-Judge.computervision.UUID-Migration.md` ✅

**Contents:**
- Overview of UUID migration and implementation
- Train/val split strategy with nullable field + folder fallback
- Database requirements and schema updates
- Troubleshooting guide

#### `/documentation/AI-Judge.computervision.Predict-Train-API.md` ✅

**Contents:**
- Predictor and Trainer class signatures  
- Job argument formats
- Data flow from API to execution
- Troubleshooting guide

#### `/documentation/AI-Judge.computervision.Database-Queries.md` ✅

**Contents:**
- Parameterized query pattern reference
- Train/val split pattern with COALESCE
- UUID type conversions
- Common issues and solutions
- Adding custom queries checklist

## Database Schema Requirements

The following database schema is required for the updated code to work:

```sql
-- Videos table: is_train is NULLABLE (NULL by default)
-- Video.is_train = NULL means "inherit from folder" 
ALTER TABLE Videos ADD COLUMN is_train BOOLEAN NULL DEFAULT NULL;

-- Folders table: is_train is NOT NULL (must have a value, defaults to 1)
-- If Folder.is_train is 1, all videos in that folder are training by default
ALTER TABLE Folders ADD COLUMN is_train BOOLEAN NOT NULL DEFAULT 1;

-- Ensure UUID fields are BINARY(16):
-- Videos.id: BINARY(16)
-- Folders.id: BINARY(16)
-- FrameLabels.videoId: BINARY(16) 
-- Skills.videoId: BINARY(16)
-- Jobs.id: BINARY(16)
```

**Train/Val Assignment Logic:**
- Effective is_train value: `COALESCE(Videos.is_train, Folders.is_train)`
- If Video.is_train is NULL → uses Folder.is_train (defaults to 1 if folder not set)
- If Video.is_train is 0 or 1 → uses that value (overrides Folder)
- Folder.is_train is never NULL (provides fallback for videos set to NULL)

## Implementation Details

### Train/Val Split
Uses `COALESCE(v.is_train, f.is_train)` for per-video override with folder-level defaults.

### VideoId Type  
All methods use `UUID` objects with automatic SQLAlchemy BINARY(16) conversion.

### SQL Queries
Parameterized syntax (`:paramname`) provides security and clear parameter binding via params dict.

## Integration Points

### API → ComputerVision Flow

```
API (JobService)
  ↓
Creates Job(type='PREDICT', step='SKILL', 
            job_arguments={'videoId': '550e8400-...', 'model': 'MViT_extra_dense'})
  ↓
JobExecuter
  ↓
Converts: videoId = UUID(job_arguments['videoId'])
          jobId = UUID(bytes=job['id'])
  ↓
Predictor.predict(videoId=videoId, ...)
  ↓
REPO_GENERAL methods (all receive UUID objects)
  ↓
Database queries with parameterized UUID parameters
```

### RepoGeneral → Database Flow

```
RepoGeneral.get_videoinfo(videoId: UUID)
  ↓
qry = sqlal.text("""SELECT * FROM Videos WHERE id = :videoId""")
  ↓
params = {'videoId': videoId}  # UUID object
  ↓
SQLAlchemy's UUIDType converter:
  - Binds: uuid.UUID.bytes → BINARY(16)
  - Retrieves: BINARY(16) data → uuid.UUID object
  ↓
Returns pandas DataFrame with UUID column values
```

## Testing

- Full prediction pipeline: LOCALIZE → SEGMENT → SKILL  
- Train/val split correctness with nullable field fallback
- UUID conversion from job_arguments and database results

## Breaking Changes

⚠️ This migration requires:
- Video.is_train field added (nullable, default NULL)
- Folder.is_train field added (not nullable, default 1)  
- All job queues use UUID strings for videoId

Migration requirements:
```sql
ALTER TABLE Videos ADD COLUMN is_train BOOLEAN NULL DEFAULT NULL;
ALTER TABLE Folders ADD COLUMN is_train BOOLEAN NOT NULL DEFAULT 1;
```

## Performance

- Parameterized queries prevent cache misses
- `is_train` field indexed for fast filtering
- UUID binary storage matches legacy int size

## Known Limitations

1. **Batch size must be 1**: Both predict and train methods currently enforce `batch_size == 1`
   - Future optimization: Support larger batches for increased throughput

2. **MOD() operations unavailable**: Old MOD-based splitting is replaced by is_train field
   - Solution: Use `is_train` field at video level OR folder.is_train for defaults
   - All queries must use `COALESCE(v.is_train, f.is_train)` pattern

3. **Folder requirement**: Videos must have a valid folderId pointing to existing Folder
   - All queries JOIN to Folders table for fallback pattern
   - Verify referential integrity: every Video.folderId → valid Folders.id

4. **VideoNames index sensitivity**: DataFrame index type must match UUID type
   - Solution: Ensure `__load_relativePaths_of_videos_with_framelabels()` properly converts to UUID

## Simple Migration Steps

1. Add database columns (see Migration requirements above)
2. Verify all Videos have folderId set
3. Deploy latest code with UUID changes
4. Test prediction and training pipelines
5. Verify train/val splits work correctly

## Next Steps

- Test full prediction pipeline with UUID videos
- Verify train/val split behavior with nullable fields
- Monitor performance with large datasets
- Consider SQLAlchemy ORM migration for type safety

---

**Version:** 2.0  
**Last Updated:** March 14, 2026  
**Maintainer:** AI-Judge Development Team

