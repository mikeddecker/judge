from flask import request
from flask_restful import Resource
from services.videoService import VideoService
from services.statsService import StatsService
import os
from repository.db import db
from datetime import datetime, timezone

class StatsRouter(Resource):
    def __init__(self, **kwargs):
        self.videoService = VideoService()
        self.statsService = StatsService(self.videoService)
        super().__init__(**kwargs)

    def get(self):
        step = request.args.get('step')

        match step:
            case 'general':
                return self.statsService.general(), 200
            case 'localization':
                return self.statsService.localization(), 200
            case 'segmentation':
                return self.statsService.segmentation(), 200
            case 'recognition':
                return self.statsService.recognition(), 200
            case 'monitoring':
                return self._get_monitoring_stats(), 200
            case _:
                return f'Forbidden step: {step}', 404
    
    def _get_monitoring_stats(self) -> dict:
        """Return monitoring dashboard statistics for admin panel
        
        Includes sync queue depth, transfer rates, conflict tracking, etc.
        Designed for admin dashboard - separate tab in stats view.
        """
        try:
            from repository.models import Jobs, ConflictLog, DeletedVideos, FrameLabel, Video
            
            # Job queue metrics
            pending_ai_jobs = db.session.query(Jobs).filter_by(
                job_category='AI', status='Created'
            ).count()
            pending_sync_jobs = db.session.query(Jobs).filter_by(
                job_category='SYNC', status='Created'
            ).count()
            pending_backup_jobs = db.session.query(Jobs).filter_by(
                job_category='BACKUP', status='Created'
            ).count()
            
            completed_jobs = db.session.query(Jobs).filter_by(status='Completed').count()
            failed_jobs = db.session.query(Jobs).filter_by(status='Failed').count()
            
            # Conflict metrics
            unresolved_conflicts = db.session.query(ConflictLog).filter_by(
                is_resolved=False, auto_resolved=False
            ).count()
            auto_resolved_conflicts = db.session.query(ConflictLog).filter_by(
                is_resolved=True, auto_resolved=True
            ).count()
            
            # Data metrics
            total_videos = db.session.query(Video).count()
            labeled_videos = db.session.query(Video).join(
                FrameLabel
            ).filter(FrameLabel.video_id == Video.id).distinct().count()
            
            soft_deleted_videos = db.session.query(DeletedVideos).count()
            
            # Storage metrics
            storage_path = os.getenv('STORAGE_DIR_VIDEOS', '')
            storage_gb = 0.0
            if storage_path and os.path.exists(storage_path):
                for dirpath, dirnames, filenames in os.walk(storage_path):
                    for filename in filenames:
                        try:
                            filepath = os.path.join(dirpath, filename)
                            storage_gb += os.path.getsize(filepath) / (1024**3)
                        except:
                            pass
            
            generated_path = os.getenv('STORAGE_DIR_GENERATED_DATA', '')
            generated_gb = 0.0
            if generated_path and os.path.exists(generated_path):
                for dirpath, dirnames, filenames in os.walk(generated_path):
                    for filename in filenames:
                        try:
                            filepath = os.path.join(dirpath, filename)
                            generated_gb += os.path.getsize(filepath) / (1024**3)
                        except:
                            pass
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "queue": {
                    "ai_jobs_pending": pending_ai_jobs,
                    "sync_jobs_pending": pending_sync_jobs,
                    "backup_jobs_pending": pending_backup_jobs,
                    "total_pending": pending_ai_jobs + pending_sync_jobs + pending_backup_jobs,
                    "jobs_completed_total": completed_jobs,
                    "jobs_failed_total": failed_jobs,
                },
                "conflicts": {
                    "unresolved_count": unresolved_conflicts,
                    "auto_resolved_count": auto_resolved_conflicts,
                    "total_conflicts": unresolved_conflicts + auto_resolved_conflicts,
                },
                "data": {
                    "total_videos": total_videos,
                    "labeled_videos": labeled_videos,
                    "labeling_percentage": round((labeled_videos / total_videos * 100) if total_videos > 0 else 0, 2),
                    "soft_deleted_videos": soft_deleted_videos,
                },
                "storage": {
                    "videos_gb": round(storage_gb, 2),
                    "generated_data_gb": round(generated_gb, 2),
                    "total_gb": round(storage_gb + generated_gb, 2),
                },
                "region": os.getenv('REGION', 'belgium'),
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

