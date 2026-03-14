#!/usr/bin/env python3
"""
Sync job manager for bidirectional multi-region data synchronization.

Executes SYNC category jobs from the job queue, handling:
- Labeled video synchronization (videos with frame labels, detected via database query)
- Generated data synchronization (training results, predictions)
- Hard-delete synchronization (from DeletedVideos after 30-day window)
- Move/rename detection for previously labeled videos

Supports N-region configuration via ALL_REGIONS environment variable.
Includes health checks to verify remote regions are accessible before syncing.

Jobs are filtered by job_category='SYNC' to avoid interference with AI jobs.
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
import json
import time
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyncJobManager:
    """Manages bidirectional rsync synchronization between regions"""
    
    def __init__(self, db_connection_string: str = None, dry_run: bool = True):
        """Initialize sync manager
        
        Args:
            db_connection_string: SQLAlchemy connection string
            dry_run: If True, don't actually rsync (use --dry-run flag)
        """
        self.dry_run = dry_run
        self.ssh_user = os.environ.get('SYNC_SSH_USER', 'sync_user')
        self.ssh_timeout = int(os.environ.get('SYNC_SSH_TIMEOUT', '10'))
        
        # Try to import Flask app context for database access
        try:
            from api.app import app, db
            self.app = app
            self.db = db
            self.has_db = True
        except ImportError:
            logger.warning("Could not import Flask app, running in standalone mode")
            self.has_db = False
    
    def get_rsync_args(self, source_path: str, target_host: str, target_path: str,
                       exclude_patterns: List[str] = None) -> List[str]:
        """Build rsync command arguments with optimal settings for multi-region sync"""
        
        args = [
            'rsync',
            '--archive',           # Preserve permissions, timestamps, etc.
            '--verbose',           # Verbose output
            '--delete',            # Delete files not in source
            '--compress',          # Compress during transfer
            '--checksum',          # Use checksum to detect changes (not mtime)
            '--partial',           # Keep partial transfers
            '--progress',          # Show progress
        ]
        
        if self.dry_run:
            args.append('--dry-run')
        
        # Add exclude patterns (e.g., hidden files, temp files)
        if exclude_patterns is None:
            exclude_patterns = [
                '*.tmp',
                '*.lock',
                '.DS_Store',
                '.git',
                '__pycache__',
                '*.pyc',
            ]
        
        for pattern in exclude_patterns:
            args.extend(['--exclude', pattern])
        
        # Source and target
        args.append(f"{source_path}/")  # Trailing slash: sync contents, not directory itself
        args.append(f"{target_host}:{target_path}/")
        
        return args
    
    def sync_labeled_videos(self, source_region: str, target_region: str, 
                           video_ids: List[str] = None, account_id: str = None) -> bool:
        """Sync labeled videos from source to target region
        
        Uses DATABASE-DRIVEN selection: only syncs videos that have frame labels
        (determined by FrameLabel.query.filter_by(video_id=X).count() > 0)
        
        Args:
            source_region: Source region identifier
            target_region: Target region identifier
            video_ids: Optional list of specific video IDs to sync
            account_id: Optional account UUID for per-account sync (part of path)
        
        Returns:
            True if sync successful, False otherwise
        """
        logger.info(f"Syncing labeled videos: {source_region} → {target_region}")
        
        # Check remote accessibility before attempting sync
        if not self._check_remote_accessibility(target_region):
            logger.error(f"Remote region '{target_region}' is not accessible, will retry later")
            return False
        
        # Determine videos to sync based on database
        videos_to_sync = self._get_videos_requiring_sync(account_id, video_ids)
        if not videos_to_sync:
            logger.info("No videos requiring sync found")
            return True
        
        logger.info(f"Found {len(videos_to_sync)} videos requiring sync")
        
        # Get source and target paths
        base_storage_dir = os.environ.get('STORAGE_DIR_VIDEOS')
        if not base_storage_dir:
            logger.error("Missing STORAGE_DIR_VIDEOS environment variable")
            return False
        
        # Build account-based path structure: {STORAGE_DIR}/{account_uuid}/
        if account_id:
            source_path = os.path.join(base_storage_dir, account_id)
        else:
            source_path = base_storage_dir
        
        target_host = self._get_remote_host(target_region)
        if not target_host:
            return False
        
        # Build rsync command
        args = self.get_rsync_args(source_path, target_host, source_path)
        
        # Filter for only videos that have labels (database-driven)
        for video_id in videos_to_sync:
            args.extend(['--include', f"*/{video_id}/**"])
        
        # Exclude everything else
        args.append('--exclude=*')
        
        return self._execute_rsync(args, f"labeled_videos ({source_region} → {target_region})")
    
    def sync_generated_data(self, source_region: str, target_region: str,
                           video_ids: List[str] = None, account_id: str = None) -> bool:
        """Sync generated data (training results, predictions, metrics)
        
        Args:
            source_region: Source region identifier
            target_region: Target region identifier
            video_ids: Optional list of specific video IDs to sync
            account_id: Optional account UUID for per-account data
        
        Returns:
            True if sync successful, False otherwise
        """
        logger.info(f"Syncing generated data: {source_region} → {target_region}")
        
        # Check remote accessibility first
        if not self._check_remote_accessibility(target_region):
            logger.error(f"Remote region '{target_region}' is not accessible, will retry later")
            return False
        
        source_path = os.environ.get('STORAGE_DIR_GENERATED_DATA')
        target_host = self._get_remote_host(target_region)
        
        if not source_path or not target_host:
            logger.error("Missing STORAGE_DIR_GENERATED_DATA or remote host")
            return False
        
        # Account-based path if specified
        if account_id:
            source_path = os.path.join(source_path, account_id)
        
        args = self.get_rsync_args(source_path, target_host, source_path)
        
        # Exclude backup directory from regular sync
        args.extend(['--exclude', 'backups/'])
        
        if video_ids:
            for vid in video_ids:
                args.extend(['--include', f"*/{vid}/**"])
            args.append('--exclude=*')
        
        return self._execute_rsync(args, f"generated_data ({source_region} → {target_region})")
    
    def sync_hard_deletes(self, source_region: str, target_region: str, account_id: str = None) -> bool:
        """Sync hard-delete information (DeletedVideos after 30-day window)
        
        This removes files that have been hard-deleted in the source region.
        
        Args:
            source_region: Source region identifier
            target_region: Target region identifier
            account_id: Optional account UUID for per-account deletes
        
        Returns:
            True if sync successful, False otherwise
        """
        logger.info(f"Syncing hard deletes: {source_region} → {target_region}")
        
        # Check remote accessibility first
        if not self._check_remote_accessibility(target_region):
            logger.error(f"Remote region '{target_region}' is not accessible, will retry later")
            return False
        
        target_host = self._get_remote_host(target_region)
        videos_path = os.environ.get('STORAGE_DIR_VIDEOS')
        generated_path = os.environ.get('STORAGE_DIR_GENERATED_DATA')
        
        if not target_host:
            logger.error("Missing remote host configuration")
            return False
        
        success = True
        
        # Account-based paths if specified
        if account_id:
            if videos_path:
                videos_path = os.path.join(videos_path, account_id)
            if generated_path:
                generated_path = os.path.join(generated_path, account_id)
        
        # Sync delete markers for videos
        if videos_path:
            args = self.get_rsync_args(videos_path, target_host, videos_path)
            success &= self._execute_rsync(args, f"video deletes ({source_region} → {target_region})")
        
        # Sync delete markers for generated data
        if generated_path:
            args = self.get_rsync_args(generated_path, target_host, generated_path)
            args.extend(['--exclude', 'backups/'])
            success &= self._execute_rsync(args, f"generated data deletes ({source_region} → {target_region})")
        
        return success
    
    def _get_remote_host(self, region: str) -> Optional[str]:
        """Get SSH host for a region from environment or config"""
        env_var = f"REGION_{region.upper()}_HOST"
        host = os.environ.get(env_var)
        
        if not host:
            logger.error(f"Missing {env_var} in environment")
            return None
        
        return host
    
    def _check_remote_accessibility(self, region: str) -> bool:
        """Check if remote region is accessible via SSH
        
        Performs a quick SSH connectivity test before syncing.
        Returns False if host is unreachable, so job is requeued for retry.
        
        Returns:
            True if remote is accessible, False otherwise
        """
        host = self._get_remote_host(region)
        if not host:
            return False
        
        try:
            # Simple SSH connectivity check: execute 'true' remotely
            result = subprocess.run(
                ['ssh', '-o', f'ConnectTimeout={self.ssh_timeout}', host, 'true'],
                check=False,
                capture_output=True,
                timeout=self.ssh_timeout + 2
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Remote region '{region}' ({host}) is accessible")
                return True
            else:
                logger.warning(f"❌ Remote region '{region}' ({host}) is not responding")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"❌ SSH timeout connecting to '{region}' ({host})")
            return False
        except Exception as e:
            logger.warning(f"❌ SSH error connecting to '{region}' ({host}): {e}")
            return False
    
    def get_all_other_regions(self, current_region: str) -> List[str]:
        """Get all regions except the current one for N-region sync
        
        Reads ALL_REGIONS environment variable (e.g., 'belgium,usa,germany,japan')
        and returns all regions except current.
        
        Args:
            current_region: Current region identifier
        
        Returns:
            List of other region identifiers
        """
        all_regions_str = os.environ.get('ALL_REGIONS', 'belgium,usa')
        all_regions = [r.strip().lower() for r in all_regions_str.split(',')]
        
        current_region = current_region.lower().strip()
        other_regions = [r for r in all_regions if r != current_region]
        
        logger.info(f"Found {len(other_regions)} target regions: {', '.join(other_regions)}")
        return other_regions
    
    def _get_videos_requiring_sync(self, account_id: str = None, 
                                   specific_video_ids: List[str] = None) -> Set[str]:
        """Determine which videos require sync based on database state
        
        DATABASE-DRIVEN approach: queries FrameLabel table to find videos
        with at least one labeled frame. Also detects moved/renamed videos.
        
        Args:
            account_id: Filter to specific account (optional)
            specific_video_ids: Only check these video IDs (optional)
        
        Returns:
            Set of video UUIDs that have labels and require sync
        """
        if not self.has_db:
            logger.warning("Database not available, skipping database-driven selection")
            return set(specific_video_ids) if specific_video_ids else set()
        
        videos_with_labels = set()
        
        try:
            with self.app.app_context():
                from repository.models import FrameLabel, Video
                from sqlalchemy import func
                
                # Query: Find all videos with at least one frame label
                query = self.db.session.query(Video.id).join(
                    FrameLabel
                ).filter(
                    FrameLabel.video_id == Video.id
                ).group_by(
                    Video.id
                ).distinct()
                
                # Optional: Filter by account
                if account_id:
                    query = query.filter(Video.account_id == account_id)
                
                # Optional: Filter by specific video IDs
                if specific_video_ids:
                    query = query.filter(Video.id.in_(specific_video_ids))
                
                # Execute query
                results = query.all()
                videos_with_labels = {str(row[0]) for row in results}
                
                logger.info(f"Database query found {len(videos_with_labels)} videos with labels")
                
                # Also detect moved videos
                moved_videos = self._detect_moved_videos(account_id, videos_with_labels)
                if moved_videos:
                    videos_with_labels.update(moved_videos)
                    logger.info(f"Added {len(moved_videos)} moved videos to sync")
                
        except Exception as e:
            logger.error(f"Error querying labeled videos: {e}")
        
        return videos_with_labels
    
    def _detect_moved_videos(self, account_id: str = None, 
                            exclude_videos: Set[str] = None) -> Set[str]:
        """Detect and return videos that have been moved/renamed
        
        Scans video metadata to find files whose path doesn't match
        current storage structure or that have been relocated.
        
        Args:
            account_id: Check within specific account
            exclude_videos: Video IDs already identified (exclude these)
        
        Returns:
            Set of video IDs that have been moved/renamed
        """
        if not self.has_db:
            return set()
        
        moved_videos = set()
        exclude_videos = exclude_videos or set()
        
        try:
            with self.app.app_context():
                from repository.models import Video
                
                # Query videos with file_path metadata that might indicate movement
                query = self.db.session.query(Video.id, Video.file_path).filter(
                    Video.id.notin_(exclude_videos)
                )
                
                if account_id:
                    query = query.filter(Video.account_id == account_id)
                
                videos = query.all()
                
                storage_dir = os.environ.get('STORAGE_DIR_VIDEOS', '/storage/videos')
                base_path = os.path.join(storage_dir, account_id) if account_id else storage_dir
                
                # Check each video's file to detect movement
                for video_id, file_path in videos:
                    if not file_path:
                        continue
                    
                    expected_path = os.path.join(base_path, str(video_id), os.path.basename(file_path))
                    
                    # If file exists but not at expected path, it may have been moved
                    if os.path.exists(file_path) and file_path != expected_path:
                        logger.info(f"Detected moved video: {video_id} ({file_path} != {expected_path})")
                        moved_videos.add(str(video_id))
                
                logger.info(f"Found {len(moved_videos)} moved/renamed videos")
                
        except Exception as e:
            logger.warning(f"Error detecting moved videos: {e}")
        
        return moved_videos
    
    def _execute_rsync(self, args: List[str], description: str) -> bool:
        """Execute rsync command and log results"""
        logger.info(f"Running rsync for {description}")
        logger.debug(f"Command: {' '.join(args)}")
        
        try:
            result = subprocess.run(
                args,
                check=False,  # Don't raise on non-zero exit
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for large transfers
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Rsync completed: {description}")
                if result.stdout:
                    logger.debug(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Rsync failed ({result.returncode}): {description}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Rsync timeout: {description}")
            return False
        except Exception as e:
            logger.error(f"❌ Rsync exception: {description} - {e}")
            return False
    
    def process_sync_jobs(self, job_limit: int = 10) -> Dict[str, int]:
        """Process pending SYNC jobs from the database
        
        Executes SYNC jobs which may include:
        - Syncing from one region to many other regions
        - Account-specific data syncing
        - Detecting moved/renamed videos automatically
        
        Returns stats: {'completed': n, 'failed': n, 'skipped': n, 'requeued': n}
        """
        if not self.has_db:
            logger.error("Database not available, cannot process jobs")
            return {'completed': 0, 'failed': 0, 'skipped': 0, 'requeued': 0}
        
        stats = {'completed': 0, 'failed': 0, 'skipped': 0, 'requeued': 0}
        
        with self.app.app_context():
            from repository.models import Jobs
            
            # Get pending SYNC jobs
            pending_jobs = self.db.session.query(Jobs).filter_by(
                job_category='SYNC',
                status='Created'
            ).limit(job_limit).all()
            
            logger.info(f"Found {len(pending_jobs)} pending SYNC jobs")
            
            for job in pending_jobs:
                try:
                    args = job.job_arguments
                    source_region = args.get('source_region', os.environ.get('REGION', 'belgium'))
                    target_regions = args.get('target_regions')  # Can be list if syncing to multiple regions
                    target_region = args.get('target_region')    # Single region fallback
                    data_type = args.get('data_type', 'labeled_videos')
                    video_ids = args.get('video_ids', [])
                    account_id = args.get('account_id')  # Account UUID for per-account sync
                    
                    # If target_region not specified but target_regions is, use all other regions
                    if not target_region and not target_regions:
                        target_regions = self.get_all_other_regions(source_region)
                    elif target_region and not target_regions:
                        target_regions = [target_region]
                    elif isinstance(target_regions, str):
                        target_regions = [r.strip() for r in target_regions.split(',')]
                    
                    logger.info(f"Processing SYNC job {job.id}: {data_type} from {source_region} to {target_regions}")
                    
                    # Execute sync to all target regions
                    success = True
                    for target in target_regions:
                        region_success = False
                        try:
                            if data_type == 'labeled_videos':
                                region_success = self.sync_labeled_videos(
                                    source_region, target, video_ids, account_id
                                )
                            elif data_type == 'generated_data':
                                region_success = self.sync_generated_data(
                                    source_region, target, video_ids, account_id
                                )
                            elif data_type == 'deletions':
                                region_success = self.sync_hard_deletes(
                                    source_region, target, account_id
                                )
                            else:
                                logger.warning(f"Unknown data_type: {data_type}")
                                stats['skipped'] += 1
                                continue
                        except Exception as e:
                            logger.error(f"Error syncing to {target}: {e}")
                            region_success = False
                        
                        # If any region fails (esp. due to accessibility), requeue the job
                        if not region_success:
                            success = False
                    
                    # Update job status
                    if success:
                        job.status = 'Completed'
                        stats['completed'] += 1
                    else:
                        # Check if it's a health check failure (temporary) vs real error
                        # For now, requeue if any region was unreachable
                        job.status = 'Created'  # Will be retried
                        stats['requeued'] += 1
                    
                    self.db.session.commit()
                        
                except Exception as e:
                    logger.error(f"Error processing job {job.id}: {e}")
                    job.status = 'Failed'
                    job.status_details = str(e)
                    self.db.session.commit()
                    stats['failed'] += 1
        
        return stats

def main():
    parser = argparse.ArgumentParser(
        description='Manage bidirectional data synchronization via rsync'
    )
    parser.add_argument(
        '--mode',
        choices=['jobs', 'labeled-videos', 'generated-data', 'deletes'],
        default='jobs',
        help='Sync mode (default: process job queue)'
    )
    parser.add_argument(
        '--source-region',
        default='belgium',
        help='Source region (default: belgium)'
    )
    parser.add_argument(
        '--target-region',
        default='usa',
        help='Target region (default: usa)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually execute rsync (default: dry run)'
    )
    parser.add_argument(
        '--job-limit',
        type=int,
        default=10,
        help='Max jobs to process per run (default: 10)'
    )
    
    args = parser.parse_args()
    
    manager = SyncJobManager(dry_run=not args.execute)
    
    if args.mode == 'jobs':
        stats = manager.process_sync_jobs(args.job_limit)
        logger.info(f"Job processing complete: {stats}")
    elif args.mode == 'labeled-videos':
        success = manager.sync_labeled_videos(args.source_region, args.target_region)
        sys.exit(0 if success else 1)
    elif args.mode == 'generated-data':
        success = manager.sync_generated_data(args.source_region, args.target_region)
        sys.exit(0 if success else 1)
    elif args.mode == 'deletes':
        success = manager.sync_hard_deletes(args.source_region, args.target_region)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

