# Exit Strategy & Data Portability Plan

**Goal**: Enable accounts to retrieve their data in human-readable, portable format when they leave AI-Judge.

**Date**: March 14, 2026
**Status**: Ready for implementation

---

## Overview

Currently, all video and generated data is stored using UUID-based folder structure for system efficiency:

```
/storage/videos/{account_uuid}/{folder_uuid}/{video_uuid}/{videofile.mp4}
/storage/generated_data/{account_uuid}/{training_uuid}/weights/yolo/best.pt
```

This structure is excellent for internal operations but opaque for account holders who want to retrieve their data. The exit plan enables accounts to export or retrieve their data with **human-readable folder names**.

---

## Two Implementation Paths

### Path A: On-Demand Export (Recommended)
Accounts request their data and receive a **portable archive** with readable names.

**Advantages**:
- No permanent storage overhead (temp file only)
- Accounts decide when to export
- Can be done without system downtime
- GDPR-compliant data portability

**Implementation**:
- `POST /api/account/{account_id}/export-data` endpoint
- Returns: ZIP archive with readable folder structure
- Storage: Temp files (auto-delete after 24h)

### Path B: Readable Folder Migration (Opt-in)
Accounts choose to migrate their storage to **human-readable folder names**.

**Advantages**:
- Permanent readable structure
- Direct file access via SFTP/SCP
- No archive size limits

**Disadvantages**:
- Requires downtime during migration (1-10GB per 1-2 hours)
- System rewrite of all path references
- One-way operation (reverting is complex)

---

## Implementation: On-Demand Export (Path A)

### Endpoint Specification

```
POST /api/account/{account_id}/export-data
Query Parameters:
  - include_metadata: bool (default: true) - Include CSV/JSON metadata
  - with_weights: bool (default: false) - Include model weights (increases size)

Response:
  - 202 Accepted + export_id (if > 5GB, async job)
  - 200 OK + file download (if < 5GB, immediate)

Response Headers:
  - Content-Disposition: attachment; filename="judge-export-{date}.zip"
  - X-Export-Size-GB: 4.5
```

### API Implementation

```python
# api/routers/accountRouter.py (NEW)

@api.route('/account/<account_id>/export-data', methods=['POST'])
@require_auth
def export_account_data(account_id):
    """Export account data in readable, portable format
    
    Creates ZIP with structure:
      
      judge-export-{date}/
      ├── README.txt
      ├── videos/
      │   ├── 2026-03/uploaded-interview-001.mp4
      │   ├── 2026-03/game-footage-round-2.mp4
      │   └── labels/
      │       └── uploaded-interview-001.json  (frame labels)
      ├── training-results/
      │   ├── yolo_v8_full_2026-02-10/
      │   │   ├── weights/best.pt
      │   │   ├── metrics/
      │   │   │   ├── confusion_matrix.png
      │   │   │   └── results.csv
      │   │   └── config.yaml
      │   └── skills_mvit_2026-01-15/
      │       └── weights/best.pth
      ├── account_metadata.json
      │   {
      │     "name": "Company Name",
      │     "videos_count": 2,
      │     "total_labels": 150,
      │     "trainings_count": 2,
      │     "export_date": "2026-03-14T10:30:00Z",
      │     "data_size_gb": 4.5
      │   }
      └── videos_manifest.csv
          video_name,video_id,frame_count,labeled_frames,upload_date
          uploaded-interview-001,uuid-xxx,300,150,2026-03-01
          game-footage-round-2,uuid-yyy,600,0,2026-03-02
    """
    
    account = Account.query.get(account_id)
    if not account or request.user.id != account.user_id:
        return {"error": "Unauthorized"}, 403
    
    include_metadata = request.args.get('include_metadata', 'true').lower() == 'true'
    with_weights = request.args.get('with_weights', 'false').lower() == 'true'
    
    try:
        export = ExportService(account_id)
        
        # Check size - if > 5GB, make it async
        estimated_size_gb = export.estimate_size(with_weights=with_weights)
        
        if estimated_size_gb > 5.0:
            # Async job
            job = export.create_async_export_job(
                include_metadata=include_metadata,
                with_weights=with_weights
            )
            return {
                "status": "queued",
                "export_id": job.id,
                "estimated_size_gb": estimated_size_gb,
                "check_status_url": f"/api/account/{account_id}/export-status/{job.id}"
            }, 202
        else:
            # Immediate export
            zip_bytes = export.create_zip(
                include_metadata=include_metadata,
                with_weights=with_weights
            )
            
            filename = f"judge-export-{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            return send_file(
                io.BytesIO(zip_bytes),
                mimetype='application/zip',
                as_attachment=True,
                download_name=filename
            )
    
    except Exception as e:
        return {"error": str(e)}, 500

@api.route('/account/<account_id>/export-status/<export_id>', methods=['GET'])
@require_auth
def get_export_status(account_id, export_id):
    """Check status of async export job"""
    job = ExportJob.query.get(export_id)
    if not job or job.account_id != account_id:
        return {"error": "Not found"}, 404
    
    return {
        "status": job.status,  # queued, processing, completed, failed
        "progress_percent": job.progress,
        "estimated_completion": job.eta,
        "download_url": f"/api/account/{account_id}/export-download/{export_id}" if job.status == "completed" else None
    }, 200
```

### Backend Service

```python
# api/services/exportService.py (NEW)

class ExportService:
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.account = Account.query.get(account_id)
        self.storage_base = os.getenv('STORAGE_DIR_VIDEOS')
        self.generated_base = os.getenv('STORAGE_DIR_GENERATED_DATA')
    
    def estimate_size(self, with_weights: bool = False) -> float:
        """Estimate export size in GB without creating ZIP"""
        account_video_path = os.path.join(self.storage_base, str(self.account_id))
        account_gen_path = os.path.join(self.generated_base, str(self.account_id))
        
        size_gb = 0.0
        for path in [account_video_path, account_gen_path]:
            if os.path.exists(path):
                for dirpath, dirnames, filenames in os.walk(path):
                    if not with_weights and 'weights' in dirpath:
                        continue
                    for filename in filenames:
                        try:
                            size_gb += os.path.getsize(os.path.join(dirpath, filename)) / (1024**3)
                        except:
                            pass
        
        return size_gb
    
    def create_zip(self, include_metadata: bool = True, with_weights: bool = False) -> bytes:
        """Create ZIP archive with readable folder structure
        
        Returns ZIP bytes (suitable for send_file response)
        """
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add README
            zf.writestr('README.txt', self._create_readme())
            
            # Videos with readable names
            self._add_videos_to_zip(zf, with_readable_names=True)
            
            # Generated data (training results)
            self._add_generated_data_to_zip(zf, with_weights=with_weights, with_readable_names=True)
            
            # Metadata files
            if include_metadata:
                zf.writestr('account_metadata.json', self._create_metadata_json())
                zf.writestr('videos_manifest.csv', self._create_videos_manifest_csv())
        
        zip_buffer.seek(0)
        return zip_buffer.read()
    
    def _get_video_readable_name(self, video_id: UUID) -> str:
        """Get human-readable name for video
        
        Strategy:
        1. Use video.name if set
        2. Fallback to "video-{upload_date}"
        """
        video = Video.query.get(video_id)
        if video and video.name:
            sanitized = video.name.replace(' ', '-').replace('_', '-').replace('/', '-')[:50]
            return sanitized
        else:
            # Fallback to date-based name
            if video:
                date_str = video.created_at.strftime('%Y-%m-%d')
                return f"video-{date_str}-{str(video_id)[:8]}"
            return f"video-{str(video_id)[:8]}"
    
    def _add_videos_to_zip(self, zf: zipfile.ZipFile, with_readable_names: bool = True):
        """Add videos to ZIP with readable names"""
        account_video_path = os.path.join(self.storage_base, str(self.account_id))
        
        if not os.path.exists(account_video_path):
            return
        
        # Find all videos for this account
        videos = Video.query.filter_by(account_id=self.account_id).all()
        
        for video in videos:
            video_name = self._get_video_readable_name(video.id) if with_readable_names else str(video.id)
            
            # Add video file
            video_path = video.file_path  # Full path to video
            if video_path and os.path.exists(video_path):
                arc_name = f"videos/{video_name}.mp4"  # or detect extension
                zf.write(video_path, arcname=arc_name)
            
            # Add frame labels as JSON
            labels = FrameLabel.query.filter_by(video_id=video.id).all()
            if labels:
                labels_json = json.dumps([
                    {
                        "frame_number": l.frame_number,
                        "label_type": l.label_type,
                        "coordinates": l.coordinates,
                        "confidence": l.confidence,
                        "timestamp_created": l.created_at.isoformat()
                    }
                    for l in labels
                ], indent=2)
                arc_name = f"videos/labels/{video_name}.json"
                zf.writestr(arc_name, labels_json)
    
    def _add_generated_data_to_zip(self, zf: zipfile.ZipFile, with_weights: bool = False, with_readable_names: bool = True):
        """Add training results with readable names"""
        account_gen_path = os.path.join(self.generated_base, str(self.account_id))
        
        if not os.path.exists(account_gen_path):
            return
        
        # Find all training jobs for this account
        training_jobs = Jobs.query.filter_by(
            account_id=self.account_id,
            type='TRAIN',
            status='Completed'
        ).all()
        
        for job in training_jobs:
            job_name = f"{job.step or 'training'}_{job.created_at.strftime('%Y-%m-%d')}" if with_readable_names else str(job.id)
            base_path = os.path.join(account_gen_path, str(job.id))
            
            if os.path.exists(base_path):
                for dirpath, dirnames, filenames in os.walk(base_path):
                    if not with_weights and 'weights' in dirpath:
                        dirnames[:] = []  # Don't descend into weights
                        continue
                    
                    for filename in filenames:
                        file_full_path = os.path.join(dirpath, filename)
                        rel_path = os.path.relpath(file_full_path, base_path)
                        arc_name = f"training-results/{job_name}/{rel_path}"
                        zf.write(file_full_path, arcname=arc_name)
    
    def _create_metadata_json(self) -> str:
        """Create account metadata file"""
        videos = Video.query.filter_by(account_id=self.account_id).all()
        total_labels = db.session.query(func.count(FrameLabel.id)).filter(
            FrameLabel.video_id.in_([v.id for v in videos])
        ).scalar() or 0
        
        training_jobs = Jobs.query.filter_by(
            account_id=self.account_id,
            type='TRAIN'
        ).all()
        
        metadata = {
            "account_name": self.account.name or "Unknown",
            "export_date": datetime.now(timezone.utc).isoformat(),
            "videos_count": len(videos),
            "total_frame_labels": total_labels,
            "training_jobs_count": len(training_jobs),
            "estimated_size_gb": self.estimate_size(),
            "ai_judge_version": "import package_version; package_version.get_version()",
            "portability_note": "This is your complete data export. You can import this data into another AI-Judge instance or use for analysis."
        }
        
        return json.dumps(metadata, indent=2)
    
    def _create_videos_manifest_csv(self) -> str:
        """Create CSV listing all videos"""
        videos = Video.query.filter_by(account_id=self.account_id).all()
        
        rows = ["video_name,video_id,frame_count,labeled_frames,upload_date"]
        for video in videos:
            labeled_count = FrameLabel.query.filter_by(video_id=video.id).count()
            video_name = self._get_video_readable_name(video.id)
            upload_date = video.created_at.strftime('%Y-%m-%d') if hasattr(video, 'created_at') else "unknown"
            rows.append(f'{video_name},{video.id},{video.frame_length or 0},{labeled_count},{upload_date}')
        
        return '\n'.join(rows)
    
    def _create_readme(self) -> str:
        return """# AI-Judge Data Export

Generated: {date}
Account: {account_name}

## Contents

- **videos/**: All uploaded videos with their frame labels (JSON)
- **training-results/**: Training job outputs (models, metrics, configs)
- **account_metadata.json**: Summary of your account data
- **videos_manifest.csv**: List of all videos and labels

## Using This Data

### Import into another AI-Judge instance:
1. Upload videos using the web interface
2. Use /api/labels/import endpoint to restore frame labels
3. Place model weights into training directory

### Analyze locally:
- Frame labels are in frame_number, label_type, coordinates format
- Videos are standard MP4 files
- Model weights are in PyTorch (.pth) or YOLO (.pt) format
- Training metrics are in CSV and PNG format

## Data Format Examples

### Frame Label (JSON)
{{
  "frame_number": 45,
  "label_type": "person",
  "coordinates": {{"x": 100, "y": 150, "w": 50, "h": 80}},
  "confidence": 0.95,
  "timestamp_created": "2026-03-10T14:30:00Z"
}}

### Training Metadata
Results include:
- weights/best.pt: Best model checkpoint
- metrics/confusion_matrix.png: Classification metrics
- results.csv: Training history (loss, accuracy, etc.)
- config.yaml: Training hyperparameters

## Technical Details

- Export date: {date}
- Region: {region}
- Total data size: {size_gb} GB
- Compression: ZIP with DEFLATE compression
- Character encoding: UTF-8

## Questions?

See our knowledge base: https://docs.aijudge.app/data-export
Contact support: support@aijudge.app
""".format(
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            account_name=self.account.name or "Unknown",
            region=os.getenv('REGION', 'unknown'),
            size_gb=self.estimate_size()
        )
```

---

## Implementation: Readable Folder Migration (Path B - Future)

If accounts request permanent readable folder structure, implement:

### Migration Function

```python
# api/services/migrationService.py

def migrate_account_to_readable_folders(account_id: str) -> dict:
    """
    One-time migration: Rename UUID folders to readable names
    
    WARNING: One-way operation! Requires system downtime.
    
    Process:
    1. Stop all jobs processing this account
    2. Create mapping: old_uuid → new_readable_name
    3. Rename folders on disk
    4. Update all database path references
    5. Validate completeness
    6. Resume operations
    
    Returns: {status, renamed_count, failed_count, mapping}
    """
    pass
```

**When to use**: If account requests direct file access, infrequent operations, or large data volumes.

---

## Database Schema (ExportJob)

```python
class ExportJob(db.Model):
    """Track async export jobs"""
    __tablename__ = 'export_jobs'
    
    id = db.Column(db.UUID, primary_key=True, default=uuid4)
    account_id = db.Column(db.UUID, db.ForeignKey('Account.id'), nullable=False)
    
    status = db.Column(db.String(20), default='queued')  # queued, processing, completed, failed, expired
    progress = db.Column(db.Integer, default=0)  # 0-100 percent
    
    include_metadata = db.Column(db.Boolean, default=True)
    with_weights = db.Column(db.Boolean, default=False)
    
    file_path = db.Column(db.String(500))  # Temp file location
    file_size_bytes = db.Column(db.BigInteger)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    expires_at = db.Column(db.DateTime)  # 24 hours after completion
    
    error_message = db.Column(db.Text)
```

---

## Deployment Roadmap

| Phase | Timeline | Feature | FYI |
|-------|----------|---------|-----|
| **Phase 1** | March 2026 | On-demand export (< 5GB instant) | This document |
| **Phase 2** | April 2026 | Async export jobs (> 5GB) | ExportJob model |
| **Phase 3** | May 2026 | Label import API recovery | Restore labels on new instance |
| **Phase 4** | Q3 2026 | Readable folder option (opt-in) | Path B implementation |
| **Phase 5** | Optional | Automatic exports (weekly/monthly) | User preference |

---

## GDPR & Compliance

✅ **Data Portability**: Export function complies with GDPR Article 20 right to data portability  
✅ **Format**: Open, machine-accessible ZIP format (not proprietary)  
✅ **Completeness**: Includes all account data (videos, labels, training results)  
✅ **Timeliness**: Generates within 45 days (usually < 5 minutes)  
✅ **Security**: Exports only to authenticated account holder  

---

## Testing Checklist

- [ ] Export endpoint requires authentication (account owner only)
- [ ] Small export (< 5GB) returns immediately with correct ZIPstructure
- [ ] Large export (> 5GB) creates async job and returns 202
- [ ] Video readable names match database names or date-based fallback
- [ ] Frame labels exported as JSON with all metadata
- [ ] Training results include metrics, config, (optionally) weights
- [ ] Metadata JSON is valid and complete
- [ ] Videos manifest CSV is parseable
- [ ] Export can be re-imported to new instance (future)
- [ ] Temp files auto-delete after 24 hours
- [ ] Export size estimate matches actual ZIP size (±5%)
- [ ] Large exports show progress status tracking

