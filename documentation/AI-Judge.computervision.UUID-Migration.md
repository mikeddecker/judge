# AI-Judge ComputerVision: UUID Migration Guide

**Date:** March 14, 2026  
**Status:** Complete  
**Impact:** Breaking changes to video ID handling in computervision module

## Overview

The computervision module has been migrated from integer video IDs to UUID (Universally Unique Identifiers) to match the API/database layer. This document outlines the changes and their implications.

## Current Implementation

### VideoId Type
All methods now use `UUID` type from Python's `uuid` module. SQLAlchemy automatically handles conversion to/from BINARY(16) format in database.

### Database Queries
All queries use parameterized syntax with SQLAlchemy:
```python
qry = sqlal.text("""SELECT * FROM Videos WHERE id = :videoId""")
df = pd.read_sql(qry, con=connection, params={'videoId': videoId})
```

### Train/Val Split
Uses `COALESCE(v.is_train, f.is_train)` pattern:
```sql
SELECT v.* FROM Videos v
JOIN Folders f ON v.folderId = f.id
WHERE COALESCE(v.is_train, f.is_train) = 1  -- Training set
```

**Key Differences:**
- `Video.is_train` is **nullable** (NULL by default)
- `Folder.is_train` is **NOT NULL** (defaults to 1/true, meaning training)
- **Effective train value** uses `COALESCE(Video.is_train, Folder.is_train)`
- If a Video has `is_train = NULL`, it inherits the training assignment from its parent Folder
- This allows flexible per-folder defaults with optional per-video overrides

## Migration Impact

### RepoGeneral Methods Updated

| Method | Change | Status |
|--------|--------|--------|
| `get_videoinfo(videoId)` | Now uses parameterized query with UUID | ✅ |
| `get_framelabels(train_test_val)` | Uses `is_train` field, joins with Videos table | ✅ |
| `get_unique_videoId_frameNr(train_test_val)` | Uses `is_train` field | ✅ |
| `get_fully_segmented_videos(train_test_val)` | Uses `is_train` field | ✅ |
| `get_skills()` | Uses `is_train` field, accepts UUID parameter | ✅ |
| `get_skills_of_fully_segmented_videos()` | Uses `is_train` field | ✅ |
| `get_team_boxes()` | Uses `is_train` field with join | ✅ |
| `delete_job(jobId)` | Now requires UUID parameter | ✅ |
| `get_video_path(videoId)` | Now expects UUID parameter | ✅ |
| `__load_relativePaths_of_videos_with_framelabels()` | Uses parameterized queries | ✅ |

### Predictor Methods
All methods include UUID type hints and are updatedto use UUID objects.

### JobExecuter
Converts UUID strings (from job_arguments) to UUID objects before passing to Predictor/Trainer methods.

## Migration Checklist

- [x] Update all SQL queries in RepoGeneral to use parameterized queries
- [x] Replace MOD(videoId, 10) with is_train field checks
- [x] Add UUID type hints to Predictor methods
- [x] Update JobExecuter to convert string UUIDs to UUID objects
- [x] Update JobExecuter to handle job ID conversion from bytes
- [x] Add docstrings explaining UUID handling
- [ ] Test full prediction pipeline end-to-end
- [ ] Test full training pipeline end-to-end
- [ ] Verify database queries work with UUID binary fields

## Database Requirements

Ensure your Videos and Folders tables have:
- **Videos.id** column: BINARY(16) for UUID storage
- **Videos.is_train** column: BOOLEAN NULL (nullable, defaults to NULL)
- **Folders.id** column: BINARY(16) for UUID storage  
- **Folders.is_train** column: BOOLEAN NOT NULL DEFAULT 1 (defaults to true/training)

```sql
ALTER TABLE Videos ADD COLUMN is_train BOOLEAN NULL DEFAULT NULL;
ALTER TABLE Folders ADD COLUMN is_train BOOLEAN NOT NULL DEFAULT 1;
```

**Behavior:**
- When `Videos.is_train = NULL`: Use `Folders.is_train` value
- When `Videos.is_train = 1 or 0`: Use the Video's explicit value, ignoring Folder
- All queries use `COALESCE(Videos.is_train, Folders.is_train)` to determine effective training assignment

## Troubleshooting

- **KeyError on VideoNames:** Ensure VideoNames index uses UUID objects, not binary bytes
- **Incorrect train/val splits:** Verify Folder.is_train is NOT NULL and all Videos have folderId set
- **SQL parameter errors:** Use parameterized queries (`:paramname` syntax) with params dict

