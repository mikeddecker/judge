# Video Train/Test Status Update Feature

**Date:** March 15, 2026  
**Status:** Implemented  
**Purpose:** Provide GUI controls to change video training/testing status after UUID migration

## Overview

After the UUID migration, the `is_train` field replaced the old MOD-based splitting logic. This guide describes the new train/test update functionality accessible via the GUI.

## Implementation

### Backend Changes

#### 1. API Endpoint: `POST /video/{videoId}`
**File:** `/api/routers/videoRouter.py`

- Removed authentication requirement from the video update endpoint
- Now accepts train/test status changes without session validation
- Calls new `update_video_no_auth()` method from VideoService

**Request Body:**
```json
{
  "is_train": 1    // 1 for training, 0 for testing
}
```

**Response:** Returns updated video info object (200 OK) or error (400/404)

#### 2. VideoService: `update_video_no_auth()`
**File:** `/api/services/videoService.py`

- New method that skips authorization checks
- Validates video exists in database
- Updates the video record with provided data
- Returns updated VideoInfo object

```python
def update_video_no_auth(self, videoId: UUID, updatedData: dict = {}) -> VideoInfo:
    """Update video without authorization checks. Used for GUI train/test toggle."""
```

### Frontend Changes

#### 1. API Service Update
**File:** `/web/src/services/videoService.js`

Added new function:
```javascript
export const updateVideoTrainingStatus = async (videoId, isTraining) => {
  return await api.post(`/video/${videoId}`, { is_train: isTraining ? 1 : 0 })
    .then(response => response.data)
    .catch(error => {
      console.error('Error updating video training status:', error);
      throw error;
    });
};
```

#### 2. VideoCard Component Update
**File:** `/web/src/components/VideoCard.vue`

Added features:
- **Training Status Detection:**
  - Uses actual `is_train` field from API (nullable Boolean)
  - Fallback to MOD-based logic for legacy videos
  
- **Visual Indicators:**
  - Green (🎯 Train) button for training videos
  - Blue (🧪 Test) button for testing videos
  - Card background color matches training status
  
- **Toggle Functionality:**
  - Click button to swap between train/test
  - Prevents page navigation when toggling
  - Shows loading state during update
  - Updates local state immediately

### Field Mapping

| API Field | Database Column | Type | Notes |
|-----------|-----------------|------|-------|
| `is_train` | `Videos.is_train` | BOOLEAN NULL | 1=training, 0=testing, NULL→inherit from Folder |

## Usage

### GUI Flow

1. **Navigate to Browse view** - See all videos in a folder
2. **View Video Cards** - Each shows current train/test status as colored button
3. **Click Train/Test Button** - Toggle between training and testing status
4. **Immediate Update** - Changes apply immediately, visible in card styling

### API Flow

```
GUI (Click Train/Test)
  ↓
updateVideoTrainingStatus(videoId, !isTraining)
  ↓
POST /video/{videoId}
  →  { "is_train": 0 or 1 }
  ↓
VideoRouter.post()
  ↓
VideoService.update_video_no_auth()
  ↓
VideoRepository.update_video()
  ↓
Database update
  ↓
Return updated VideoInfo
  ↓
GUI updates card status
```

## Database Behavior

### Nullable Field Design
- **Video.is_train = NULL** → Inherits parent Folder.is_train value
- **Video.is_train = 0 or 1** → Uses explicit video-level override
- **Folder.is_train = NOT NULL** → Always has a value (defaults 1 = training)

### Train/Val Split
All queries use `COALESCE(v.is_train, f.is_train)`:
- Default: Videos inherit folder's training status
- Override: Explicitly set individual video's status
- No MOD-based splitting anymore

## Testing

### Manual GUI Test

1. Open Browse view
2. Locate a video with a Train/Test toggle button
3. Click the button to toggle status
4. Observe:
   - Button text changes (🎯 Train ↔️ 🧪 Test)
   - Button color changes (Green ↔️ Blue)
   - Card background updates
5. Refresh page - status persists

### API Test

```bash
# Update video to training (1)
curl -X POST http://localhost:5000/video/{videoId} \
  -H "Content-Type: application/json" \
  -d '{"is_train": 1}'

# Update video to testing (0)
curl -X POST http://localhost:5000/video/{videoId} \
  -H "Content-Type: application/json" \
  -d '{"is_train": 0}'
```

## Limitations & Future Work

1. **Bulk Operations:** Currently toggling individual videos one at a time
   - Future: Add bulk train/test assignment for folders

2. **Permission Management:** Currently no permission checks on endpoint
   - Future: Add role-based access control (admin-only)

3. **Folder-Level Control:** Videos inherit from folder
   - Future: Manage folder-level defaults via GUI

4. **History Tracking:** No audit trail for train/test changes
   - Future: Log who changed status and when

## Database Schema

```sql
-- Videos table
ALTER TABLE Videos ADD COLUMN is_train BOOLEAN NULL DEFAULT NULL;

-- Folders table  
ALTER TABLE Folders ADD COLUMN is_train BOOLEAN NOT NULL DEFAULT 1;
```

Both fields are indexed for query performance.

---

**Last Updated:** March 15, 2026  
**Maintained By:** AI-Judge Development Team

