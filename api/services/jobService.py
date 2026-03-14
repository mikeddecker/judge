from domain.job import Job
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from repository.jobRepo import JobRepository
from repository.models import ConflictLog, Jobs
from typing import List
from uuid import UUID
from datetime import datetime

VISION_MODELS = ['MViT'] # TODO : delete, rewrite this

class JobService:
    """Provides the video information of videos"""
    PROPERTIES = [
        "VideoRepo",
        "FolderRepo",
        "JobRepo",
    ]
    def __init__(self):
        self.VideoRepo = VideoRepository(db=db)
        self.FolderRepo = FolderRepository(db=db)
        self.JobRepo = JobRepository(db=db)
        
    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name in self.PROPERTIES:
                raise AttributeError(f"Cannot modify {name} once it's set")
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def __add(self, job: Job) -> None:
        """Adds the given job in the database
        TRAIN, PREDICT - L, S, R, FULL

        """
        assert isinstance(job, Job)
        if self.JobRepo.exists(job):
            raise ValueError("Job exists")
        else:
            self.JobRepo.add(job)
        
    def count(self) -> int:
        return self.JobRepo.count()
    
    def get(self) -> List[Job]:
        """Get video with the corresponding Id"""
        return self.JobRepo.get_all()
    
    def video_has_pending_job(self, videoId: UUID, model: str, step: str = 'FULL'):
        return self.JobRepo.exists_by_job_content(
            Job(type='PREDICT', step=step, job_arguments={'model': model, 'videoId':videoId})
        )
    
    def launch_job_predict_skills(self, step: str, model: str, videoId: UUID):
        job = Job(
            type = 'PREDICT',
            step = step,
            status = 'Created',
            job_arguments = { "videoId": videoId, "model": model, "save_mp4": True },
        )
        if not self.JobRepo.exists_by_job_content(job):
            self.__add(job)

    def launch_job(self, job:Job):
        if not self.JobRepo.exists_by_job_content(job):
            self.__add(job)

    def re_train_and_predict(self):
        trainjob = Job(
            type='TRAIN',
            step = 'FULL',
            status= 'Created',
            job_arguments = { "model": VISION_MODELS[0] }
        )
        if not self.JobRepo.exists_by_job_content(trainjob):
            self.__add(trainjob)

        videoIds = [1285, 1315, 1178, 1408, 2283, 2285, 2289, 2288, 2296, 2309, 2568,2569,2570,2571,2572,2573,2574,2575,2576,2577,2578,2579,2580,2581,2582,2583,2584,2585,2586,2587,2588,2589]
        for videoId in videoIds:
            predictJob = Job(
                type = 'PREDICT',
                step = 'SEGMENT_SKILL',
                status = 'Created',
                job_arguments = { "videoId": videoId, "model": "best", "save_mp4": True },
            )
            if not self.JobRepo.exists_by_job_content(predictJob):
                self.__add(predictJob)

    def get_ai_jobs(self) -> List[Job]:
        """Get all AI-related jobs for the computer vision executor"""
        return db.session.query(Jobs).filter_by(job_category='AI').all()
    
    def get_sync_jobs(self) -> List[Job]:
        """Get all sync jobs for the bidirectional sync executor"""
        return db.session.query(Jobs).filter_by(job_category='SYNC').all()
    
    def get_backup_jobs(self) -> List[Job]:
        """Get all backup jobs"""
        return db.session.query(Jobs).filter_by(job_category='BACKUP').all()
    
    def launch_sync_job(self, source_region: str, target_region: str, data_type: str, video_ids: List[UUID] = None) -> Job:
        """Launch a bidirectional sync job
        
        Args:
            source_region: Source region ('belgium' or 'usa')
            target_region: Target region ('belgium' or 'usa')
            data_type: 'labeled_videos', 'generated_data', or 'deletions'
            video_ids: Optional list of video IDs for targeted sync
        """
        sync_job = Job(
            type='SYNC',
            step=f'sync_{data_type}',
            status='Created',
            job_category='SYNC',
            job_arguments={
                'source_region': source_region,
                'target_region': target_region,
                'data_type': data_type,
                'video_ids': [str(vid) for vid in (video_ids or [])]
            }
        )
        self.launch_job(sync_job)
        return sync_job
    
    def log_conflict(self, entity_type: str, entity_id: UUID, winning_account_id: UUID, 
                      winning_region: str, winning_timestamp: datetime, winning_data: dict,
                      losing_account_id: UUID, losing_region: str, losing_timestamp: datetime, 
                      losing_data: dict, description: str, auto_resolved: bool = False) -> ConflictLog:
        """Log a first-write-wins conflict
        
        Args:
            auto_resolved: If True, conflict is automatically resolved (non-critical fields).
                          If False, requires user inspection.
        """
        conflict = ConflictLog(
            entity_type=entity_type,
            entity_id=entity_id,
            winning_account_id=winning_account_id,
            winning_region=winning_region,
            winning_timestamp=winning_timestamp,
            winning_data=winning_data,
            losing_account_id=losing_account_id,
            losing_region=losing_region,
            losing_timestamp=losing_timestamp,
            losing_data=losing_data,
            conflict_description=description,
            auto_resolved=auto_resolved,
            is_resolved=auto_resolved  # Auto-resolved means is_resolved becomes True immediately
        )
        db.session.add(conflict)
        db.session.commit()
        return conflict
    
    def get_unresolved_conflicts(self, account_id: UUID = None, include_auto_resolved: bool = False) -> List[ConflictLog]:
        """Get unresolved conflicts, optionally filtered by account
        
        Args:
            account_id: Filter by winning account (optional)
            include_auto_resolved: If False, only return conflicts requiring user action
        """
        query = self.db.session.query(ConflictLog).filter_by(is_resolved=False)
        
        if not include_auto_resolved:
            query = query.filter_by(auto_resolved=False)  # Only user-resolvable conflicts
        
        if account_id:
            query = query.filter_by(winning_account_id=account_id)
        
        return query.all()
    
    def verify_required_videos_exist_locally(self, video_ids: List[UUID]) -> tuple[bool, List[UUID]]:
        """Verify that all required videos exist on the local storage before training
        
        Called before TRAIN jobs to ensure all labeled videos are available.
        This prevents training from failing due to missing files from other regions.
        
        Args:
            video_ids: List of video UUIDs required for training
        
        Returns:
            (all_exist: bool, missing_ids: List[UUID])
            - all_exist: True if all videos exist locally
            - missing_ids: List of UUIDs that are missing from local storage
        """
        import os
        from config import ENVS
        
        missing_videos = []
        storage_path = ENVS.DIRS.VIDEOS
        
        if not storage_path or not os.path.exists(storage_path):
            return False, video_ids
        
        from repository.videoRepo import VideoRepository
        video_repo = VideoRepository(db=db)
        
        for video_id in video_ids:
            try:
                # Get video from database
                video = video_repo.get(video_id)
                if not video:
                    missing_videos.append(video_id)
                    continue
                
                # Check if video file exists on local storage
                # Path structure: {STORAGE}/{account_uuid}/{folder_uuid}/{video_filename}
                video_path = os.path.join(storage_path, video.file_path) if hasattr(video, 'file_path') and video.file_path else None
                
                if not video_path or not os.path.exists(video_path):
                    missing_videos.append(video_id)
            except Exception as e:
                # If we can't verify, mark as missing to be safe
                missing_videos.append(video_id)
        
        all_exist = len(missing_videos) == 0
        return all_exist, missing_videos

