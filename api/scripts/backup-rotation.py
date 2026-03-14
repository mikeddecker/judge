#!/usr/bin/env python3
"""
Backup retention rotation script for multi-region disaster recovery.

Implements a tiered retention strategy:
- Every 2-hour backup for 24 hours (12 backups max)
- One per day for 2 weeks (14 backups)
- One per week for 6 months (~24 backups)
- One per month forever

Can be run manually or scheduled via cron/job queue.
"""

import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
import argparse

def parse_backup_filename(filename: str) -> dict:
    """
    Parse backup filename to extract metadata.
    
    Expected format: judge_db_{label}_{YYYYMMDD_HHMMSS}.sql
    Example: judge_db_scheduled_20260314_101530.sql
    """
    try:
        # Remove extension
        base = filename.replace('.sql', '')
        parts = base.split('_')
        
        if len(parts) < 4:
            return None
        
        # judge_db_{label}_{timestamp}
        label = parts[2]  # e.g., 'scheduled' or 'shutdown'
        timestamp_str = f"{parts[3]}_{parts[4]}"  # YYYYMMDD_HHMMSS
        
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        
        return {
            'filename': filename,
            'label': label,
            'timestamp': timestamp,
            'age_days': (datetime.now() - timestamp).days,
            'age_hours': (datetime.now() - timestamp).total_seconds() / 3600,
        }
    except (ValueError, IndexError):
        return None

def should_keep_backup(parsed: dict, now: datetime = None) -> tuple:
    """
    Determine if backup should be kept based on retention policy.
    
    Returns: (should_keep, reason)
    """
    if now is None:
        now = datetime.now()
    
    age_hours = (now - parsed['timestamp']).total_seconds() / 3600
    age_days = (now - parsed['timestamp']).days
    
    # Always keep shutdown backups for at least 30 days
    if parsed['label'] == 'shutdown':
        if age_days < 30:
            return True, "shutdown_backup_24h_retention"
        else:
            return True, "shutdown_backup_permanent_retention"
    
    # Retention tiers for scheduled backups:
    
    # 1. Keep all backups from last 24 hours (2-hour resolution)
    if age_hours <= 24:
        return True, "recent_24h_tier"
    
    # 2. Keep one per day for 2 weeks (14-28 days)
    if 24 < age_days <= 14:
        # Keep first backup of each day
        # Check if this is the most recent backup from its day
        backup_date = parsed['timestamp'].date()
        return True, f"daily_tier_day_{backup_date}"
    
    # 3. Keep one per week for 6 months (15-180 days)
    if 14 < age_days <= 180:
        # Keep if it's the last backup of the week (Sunday)
        is_week_end = parsed['timestamp'].weekday() == 6  # Sunday
        if is_week_end or age_days > 180:  # Keep until we filter properly
            return True, f"weekly_tier_week_{parsed['timestamp'].isocalendar()[1]}"
        else:
            return False, "intermediate_day_not_week_end"
    
    # 4. Keep one per month forever (180+ days)
    if age_days > 180:
        # Keep first day of month backups
        is_month_start = parsed['timestamp'].day == 1
        if is_month_start:
            return True, f"monthly_tier_{parsed['timestamp'].strftime('%Y-%m')}"
        else:
            return False, "intermediate_day_not_month_start"
    
    return True, "default_keep"

def rotate_backups(backup_dir: str, dry_run: bool = True, verbose: bool = True):
    """
    Apply retention policy and delete old backups.
    
    Args:
        backup_dir: Directory containing backup files
        dry_run: If True, don't actually delete files
        verbose: If True, print details about each backup
    """
    
    if not os.path.isdir(backup_dir):
        print(f"❌ Backup directory not found: {backup_dir}")
        return
    
    # Find all backup files
    backup_files = sorted(glob.glob(os.path.join(backup_dir, "*.sql")))
    
    if not backup_files:
        print(f"ℹ️  No backup files found in {backup_dir}")
        return
    
    print(f"🔍 Found {len(backup_files)} backup files in {backup_dir}")
    print()
    
    to_delete = []
    to_keep = []
    
    # Analyze each backup
    for filepath in backup_files:
        filename = os.path.basename(filepath)
        parsed = parse_backup_filename(filename)
        
        if parsed is None:
            if verbose:
                print(f"⚠️  Skipped (invalid format): {filename}")
            continue
        
        should_keep, reason = should_keep_backup(parsed)
        
        if should_keep:
            to_keep.append({'filepath': filepath, 'parsed': parsed, 'reason': reason})
            if verbose:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"✅ Keep: {filename} ({size_mb:.1f}MB) - {reason}")
        else:
            to_delete.append({'filepath': filepath, 'parsed': parsed, 'reason': reason})
            if verbose:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"🗑️  Delete: {filename} ({size_mb:.1f}MB) - {reason}")
    
    print()
    print(f"📊 Summary: Keep {len(to_keep)} backups, delete {len(to_delete)} backups")
    
    total_freed_mb = 0
    for item in to_delete:
        filepath = item['filepath']
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        total_freed_mb += size_mb
        
        if dry_run:
            print(f"  [DRY RUN] Would delete: {os.path.basename(filepath)}")
        else:
            try:
                os.remove(filepath)
                print(f"  ✅ Deleted: {os.path.basename(filepath)}")
            except OSError as e:
                print(f"  ❌ Failed to delete {os.path.basename(filepath)}: {e}")
    
    if total_freed_mb > 0:
        print(f"💾 Space that would be freed: {total_freed_mb:.1f}MB")

def main():
    parser = argparse.ArgumentParser(
        description='Rotate MySQL backups based on retention policy'
    )
    parser.add_argument(
        '--backup-dir',
        default=os.environ.get('STORAGE_DIR_GENERATED_DATA', '/data/generated'),
        help='Directory containing backup files (default: $STORAGE_DIR_GENERATED_DATA/backups)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete files (default: dry run)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Append /backups if not already included
    backup_dir = args.backup_dir
    if not backup_dir.endswith('/backups'):
        backup_dir = os.path.join(backup_dir, 'backups')
    
    rotate_backups(
        backup_dir,
        dry_run=not args.execute,
        verbose=not args.quiet
    )

if __name__ == '__main__':
    main()

