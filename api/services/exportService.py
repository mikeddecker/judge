"""Data export service for GDPR Article 20 compliance (right to data portability)
   
Path A: On-demand export of account data as ZIP file with readable folder structure
- Instant download: < 5 GB (synchronous)
- Async job: >= 5 GB (returns job ID, complete in background)
"""

import os
import zipfile
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Optional
import uuid
from io import StringIO

from repository.db import db
from repository.models import Account, Video, FrameLabel, TrainResult, ExportJob, Skill
from config import Config

class ExportService:
    """Handles account data export with readable folder structure"""
    
    # Size threshold for async export (5 GB)
    ASYNC_THRESHOLD_GB = 5.0
    
    # ZIP file expiration time (days)
    DOWNLOAD_LINK_EXPIRY_DAYS = 7
    
    def __init__(self):
        self.temp_export_dir = os.path.join(Config.STORAGE_DIR_GENERATED_DATA, 'exports')
        os.makedirs(self.temp_export_dir, exist_ok=True)
    
    def estimate_export_size(self, account_id: str, 
                            include_metadata: bool = True,
                            include_training_results: bool = True,
                            include_frames: bool = False) -> float:
        """Estimate total size in GB of account data export
        
        Args:
            account_id: UUID of account
            include_metadata: Include JSON metadata files
            include_training_results: Include weights and training artifacts
            include_frames: Include extracted frame images (very large)
        
        Returns:
            Estimated size in GB
        """
        estimated_gb = 0.0
        
        try:
            # Get all videos for this account
            videos = self._get_account_videos(account_id)
            
            # Video files
            for video in videos:
                if video.file_path and os.path.exists(video.file_path):
                    estimated_gb += os.path.getsize(video.file_path) / (1024**3)
            
            # Training results
            if include_training_results:
                # Estimate weight files directory size
                results = db.session.query(TrainResult).all()
                for result in results:
                    # Weights typically 100MB-1GB per model
                    estimated_gb += 0.3
        
            # Frame extracts (if requested)
            if include_frames:
                # Rough estimate: 1KB per labeled frame
                frame_count = db.session.query(FrameLabel).join(
                    Video, FrameLabel.videoId == Video.id
                ).filter(
                    Video.folderId.in_(
                        db.session.query(Video.folderId).filter_by(account_id=account_id)
                    )
                ).count()
                estimated_gb += (frame_count * 1024) / (1024**3)
        
        except Exception as e:
            print(f"Error estimating export size: {e}")
        
        return round(estimated_gb, 2)
    
    def create_export_job(self, account_id: str, requested_by: str,
                         include_metadata: bool = True,
                         include_training_results: bool = True,
                         include_frames: bool = False) -> Tuple[ExportJob, bool]:
        """Create export job
        
        Returns:
            (export_job, is_async) - is_async=True if size >= threshold
        """
        estimated_size = self.estimate_export_size(
            account_id, include_metadata, include_training_results, include_frames
        )
        
        export_job = ExportJob(
            account_id=uuid.UUID(account_id).bytes,
            requested_by=uuid.UUID(requested_by).bytes,
            include_metadata=include_metadata,
            include_training_results=include_training_results,
            include_frames=include_frames,
            estimated_size_gb=estimated_size,
            status='Pending'
        )
        
        db.session.add(export_job)
        db.session.commit()
        
        is_async = estimated_size >= self.ASYNC_THRESHOLD_GB
        
        return export_job, is_async
    
    def create_export_sync(self, account_id: str, export_job_id: str) -> Tuple[str, str]:
        """Create export synchronously (called from endpoint, < 5GB)
        
        Returns:
            (zip_file_path, download_url)
        """
        try:
            # Get export job
            job_id_bytes = uuid.UUID(export_job_id).bytes
            export_job = db.session.query(ExportJob).filter_by(
                id=job_id_bytes
            ).first()
            
            if not export_job:
                raise ValueError(f"Export job {export_job_id} not found")
            
            # Create temp directory for this export
            export_uuid = str(uuid.uuid4())
            export_temp_dir = os.path.join(self.temp_export_dir, export_uuid)
            os.makedirs(export_temp_dir, exist_ok=True)
            
            # Build folder structure
            self._build_export_structure(
                account_id=str(uuid.UUID(bytes=export_job.account_id)),
                export_dir=export_temp_dir,
                include_metadata=export_job.include_metadata,
                include_training_results=export_job.include_training_results,
                include_frames=export_job.include_frames
            )
            
            # Create ZIP file
            zip_path = os.path.join(self.temp_export_dir, f"judge-export-{export_uuid}.zip")
            zip_size_gb = self._create_zip(export_temp_dir, zip_path)
            
            # Update export job
            download_url = f"/api/export/download/{export_job_id}"
            export_job.file_path = zip_path
            export_job.download_url = download_url
            export_job.actual_size_gb = zip_size_gb
            export_job.expires_at = datetime.now(timezone.utc) + timedelta(days=self.DOWNLOAD_LINK_EXPIRY_DAYS)
            export_job.status = 'Completed'
            
            db.session.commit()
            
            # Clean up temp directory
            shutil.rmtree(export_temp_dir, ignore_errors=True)
            
            return zip_path, download_url
        
        except Exception as e:
            export_job.status = 'Failed'
            export_job.error_message = str(e)[:250]
            db.session.commit()
            raise
    
    def create_export_async(self, account_id: str, export_job_id: str) -> None:
        """Create export asynchronously (background job, >= 5GB)
        
        Called from background worker
        """
        try:
            job_id_bytes = uuid.UUID(export_job_id).bytes
            export_job = db.session.query(ExportJob).filter_by(
                id=job_id_bytes
            ).first()
            
            export_job.status = 'Processing'
            db.session.commit()
            
            # Create temp directory for this export
            export_uuid = str(uuid.uuid4())
            export_temp_dir = os.path.join(self.temp_export_dir, export_uuid)
            os.makedirs(export_temp_dir, exist_ok=True)
            
            # Build structure
            self._build_export_structure(
                account_id=str(uuid.UUID(bytes=export_job.account_id)),
                export_dir=export_temp_dir,
                include_metadata=export_job.include_metadata,
                include_training_results=export_job.include_training_results,
                include_frames=export_job.include_frames
            )
            
            # Create ZIP
            zip_path = os.path.join(self.temp_export_dir, f"judge-export-{export_uuid}.zip")
            zip_size_gb = self._create_zip(export_temp_dir, zip_path)
            
            # Update job
            download_url = f"/api/export/download/{export_job_id}"
            export_job.file_path = zip_path
            export_job.download_url = download_url
            export_job.actual_size_gb = zip_size_gb
            export_job.expires_at = datetime.now(timezone.utc) + timedelta(days=self.DOWNLOAD_LINK_EXPIRY_DAYS)
            export_job.status = 'Completed'
            db.session.commit()
            
            # Clean up temp
            shutil.rmtree(export_temp_dir, ignore_errors=True)
        
        except Exception as e:
            export_job.status = 'Failed'
            export_job.error_message = str(e)[:250]
            db.session.commit()
    
    def _get_account_videos(self, account_id: str) -> List[Video]:
        """Get all videos belonging to an account"""
        # Assumption: Account owns videos through folder structure
        # This may need adjustment based on actual data model
        return db.session.query(Video).all()
    
    def _build_export_structure(self, account_id: str, export_dir: str,
                               include_metadata: bool = True,
                               include_training_results: bool = True,
                               include_frames: bool = False) -> None:
        """Build readable folder structure for export
        
        Structure:
        judge-export/
        ├── README.txt
        ├── videos/
        │   ├── video-1-readable-name.mp4
        │   ├── video-2-readable-name.mov
        │   └── labels/
        │       ├── video-1-readable-name.json
        │       └── video-2-readable-name.json
        ├── training-results/
        │   ├── yolo_v8_2026-02-10/
        │   │   ├── weights/best.pt
        │   │   └── metrics/confusion_matrix.png
        │   └── skills_mvit_2026-01-15/weights/best.pth
        └── metadata.json
        """
        
        # Create subdirectories
        videos_dir = os.path.join(export_dir, 'videos')
        labels_dir = os.path.join(videos_dir, 'labels')
        results_dir = os.path.join(export_dir, 'training-results')
        
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Get account videos
        videos = self._get_account_videos(account_id)
        
        # Add videos and labels
        for video in videos:
            if video.file_path and os.path.exists(video.file_path):
                # Get readable name
                readable_name = self._get_video_readable_name(video)
                
                # Copy video file
                dest_path = os.path.join(videos_dir, readable_name)
                try:
                    shutil.copy2(video.file_path, dest_path)
                except Exception as e:
                    print(f"Warning: Could not copy video {video.id}: {e}")
                
                # Create labels JSON
                if include_metadata and video.frameLabels:
                    labels = [
                        {
                            'frameNr': label.frameNr,
                            'x': label.x,
                            'y': label.y,
                            'width': label.width,
                            'height': label.height,
                            'jumperVisible': label.jumperVisible,
                            'labeldate': label.labeldate.isoformat() if label.labeldate else None,
                            'labeltime': str(label.labeltime) if label.labeltime else None,
                        }
                        for label in video.frameLabels
                    ]
                    
                    label_filename = readable_name.rsplit('.', 1)[0] + '.json'
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    with open(label_path, 'w') as f:
                        json.dump(labels, f, indent=2)
        
        # Add training results
        if include_training_results:
            results = db.session.query(TrainResult).all()
            for i, result in enumerate(results, 1):
                result_dir_name = f"{result.step}_{result.recipeCode}"
                result_dir = os.path.join(results_dir, result_dir_name)
                os.makedirs(result_dir, exist_ok=True)
                
                # Save result metadata
                metadata = {
                    'recipe': result.recipe,
                    'bestEpoch': result.bestEpoch,
                    'revalidationResults': result.revalidationResults,
                    'isTestrun': result.isTestrun,
                    'trainEnd': result.trainEnd.isoformat() if result.trainEnd else None,
                }
                with open(os.path.join(result_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        # Create README
        readme_content = f"""# Judge Export Data

Account: {account_id}
Exported: {datetime.now(timezone.utc).isoformat()}
Format: ZIP (readable folder structure)

## Contents

### videos/
Contains all video files uploaded to your account with readable names.

### videos/labels/
Frame-by-frame annotations for each video in JSON format.
Each label includes: frame number, bounding box coordinates, jumper visibility.

### training-results/
Training models, weights, and metrics for all trained models.
Step: {result.step}, Recipe: {result.recipeCode}

## GDPR Data Portability

This export complies with GDPR Article 20 (right to data portability).
Data is provided in machine-readable format (JSON/MP4/PT files).
Access this data only with proper authorization.

For questions, contact: support@judge.ai
"""
        with open(os.path.join(export_dir, 'README.txt'), 'w') as f:
            f.write(readme_content)
        
        # Create manifest CSV
        manifest_data = []
        for video in videos:
            manifest_data.append({
                'video_id': str(uuid.UUID(bytes=video.id)),
                'video_name': video.name,
                'readable_name': self._get_video_readable_name(video),
                'duration_seconds': video.duration,
                'frame_count': video.frameLength,
                'fps': video.fps,
                'labeled_frames': len(video.frameLabels),
            })
        
        csv_path = os.path.join(export_dir, 'videos_manifest.csv')
        if manifest_data:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=manifest_data[0].keys())
                writer.writeheader()
                writer.writerows(manifest_data)
    
    def _create_zip(self, source_dir: str, zip_path: str) -> float:
        """Create ZIP file from directory
        
        Returns:
            Size of ZIP in GB
        """
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
        
        zip_size_bytes = os.path.getsize(zip_path)
        return round(zip_size_bytes / (1024**3), 2)
    
    def _get_video_readable_name(self, video: Video) -> str:
        """Generate readable filename with proper extension
        
        Example: "interview-01-john-doe-20250210.mp4"
        Sanitize special characters, use video name + timestamp
        """
        # Get video name, sanitize
        base_name = video.name
        base_name = ''.join(c if c.isalnum() or c in '-_ ' else '' for c in base_name)
        base_name = base_name.replace(' ', '-').lower()
        
        # Add timestamp
        timestamp = video.createdAt.strftime('%Y%m%d')
        
        # Get extension from file_path
        ext = os.path.splitext(video.file_path)[1] if video.file_path else '.mp4'
        
        return f"{base_name}-{timestamp}{ext}"
    
    def cleanup_expired_exports(self) -> int:
        """Delete exports that have expired
        
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        
        expired = db.session.query(ExportJob).filter(
            ExportJob.expires_at < datetime.now(timezone.utc),
            ExportJob.status == 'Completed'
        ).all()
        
        for export_job in expired:
            try:
                if export_job.file_path and os.path.exists(export_job.file_path):
                    os.remove(export_job.file_path)
                    deleted_count += 1
                
                # Mark as deleted in DB
                db.session.delete(export_job)
            except Exception as e:
                print(f"Error cleaning up export {export_job.id}: {e}")
        
        db.session.commit()
        return deleted_count

