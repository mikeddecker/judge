# AI-Judge ComputerVision: Database Query Patterns

**Date:** March 14, 2026  
**Subject:** UUID-safe database queries and data access patterns

## Overview

This document provides best practices for querying the database in the computervision module with proper UUID handling and the new `is_train` field for train/val splitting.

## Query Pattern Reference

### Parameterized Query Pattern

All queries use parameterized syntax to prevent SQL injection and handle UUID conversion:
```python
qry = sqlal.text("""SELECT * FROM Videos WHERE id = :videoId""")
df = pd.read_sql(qry, connection, params={'videoId': videoId})
```

### Train/Validation Split Pattern

All train/val filtering uses `COALESCE(v.is_train, f.is_train)` with Folder JOIN:
- `Video.is_train = NULL` → inherits `Folder.is_train`
- `Video.is_train = 1 or 0` → uses video-level override
- `Folder.is_train` always has a value (defaults 1 = training)

## RepoGeneral Methods

All methods in RepoGeneral follow the same patterns:
- UUID type hints on parameters
- Parameterized SQL queries (`:paramname` syntax)
- COALESCE pattern for train/val filtering
- Folder JOIN for inheritance fallback

## UUID Type Conversions

- **String → UUID:** `UUID(job_arguments['videoId'])`
- **Binary → UUID:** `UUID(bytes=db_value)`
- **UUID → String:** `str(uuid_obj)`

## Common Issues

| Issue | Solution |
|-------|----------|
| Parameters not matching query placeholders | Ensure all `:paramname` placeholders have corresponding key in params dict |
| TypeError on UUID operations | Pass UUID objects to queries, not strings |
| Train/val split incorrect | Verify Folder.is_train not NULL and all Videos have folderId |
| VideoNames KeyError | Ensure DataFrame index uses UUID objects, not binary bytes |

## Performance Tips

- Filter at SQL level using COALESCE pattern, not in pandas
- UUID conversion happens automatically via SQLAlchemy
- Ensure Folder.is_train is indexed for large datasets

## Adding Custom Queries

When adding new queries to RepoGeneral:
- Use parameterized syntax (`:paramname`)
- Pass params dict to `pd.read_sql()`
- Use COALESCE pattern for train/val filtering
- Include Folder JOIN for fallback logic
- Add UUID type hints to method parameters

