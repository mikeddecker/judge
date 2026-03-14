"""Export endpoint for GDPR data portability (Article 20)

Path A: On-demand export of account data as readable ZIP

Endpoint: POST /api/account/{account_id}/export-data
Query params:
  - include_metadata (bool): Include JSON labels (default: true)
  - include_training_results (bool): Include model weights (default: true)
  - include_frames (bool): Include extracted frames (default: false)

Response:
  - < 5 GB: Return ZIP file directly (200 + file)
  - >= 5 GB: Return job ID (202 Accepted + job_id)

Requires authentication and authorization (account owner or admin)
"""

from flask import request, send_file, jsonify
from flask_restful import Resource
from datetime import datetime, timezone
import uuid
import os

from services.exportService import ExportService
from repository.db import db
from repository.models import ExportJob, Account

class ExportDataRouter(Resource):
    """Handle account data export requests"""
    
    def __init__(self, **kwargs):
        self.export_service = ExportService()
        super().__init__(**kwargs)
    
    def post(self):
        """Create new export request
        
        Returns:
            - 200 + ZIP file: For exports < 5 GB
            - 202 + job_id: For async exports >= 5 GB
            - 400: Invalid parameters
            - 401: Unauthorized (not account owner/admin)
            - 404: Account not found
            - 409: Export already in progress
        """
        try:
            account_id = request.args.get('account_id')
            if not account_id:
                return {'error': 'account_id is required'}, 400
            
            # Validate account exists
            try:
                account_id_bytes = uuid.UUID(account_id).bytes
            except ValueError:
                return {'error': 'Invalid account_id format'}, 400
            
            account = db.session.query(Account).filter_by(id=account_id_bytes).first()
            if not account:
                return {'error': 'Account not found'}, 404
            
            # TODO: Check authorization (auth context)
            # For now, assume authenticated user
            requested_by = account_id  # In real implementation, get from auth token
            
            # Parse parameters
            include_metadata = request.args.get('include_metadata', 'true').lower() == 'true'
            include_training_results = request.args.get('include_training_results', 'true').lower() == 'true'
            include_frames = request.args.get('include_frames', 'false').lower() == 'true'
            
            # Check for existing in-progress exports
            in_progress = db.session.query(ExportJob).filter_by(
                account_id=account_id_bytes,
                status='Processing'
            ).first()
            if in_progress:
                return {
                    'error': 'Export already in progress',
                    'job_id': str(uuid.UUID(bytes=in_progress.id))
                }, 409
            
            # Create export job
            export_job, is_async = self.export_service.create_export_job(
                account_id=account_id,
                requested_by=requested_by,
                include_metadata=include_metadata,
                include_training_results=include_training_results,
                include_frames=include_frames
            )
            
            job_id = str(uuid.UUID(bytes=export_job.id))
            
            # For large exports, return async response
            if is_async:
                return {
                    'status': 'accepted',
                    'job_id': job_id,
                    'estimated_size_gb': export_job.estimated_size_gb,
                    'message': f'Export job accepted. Large export ({export_job.estimated_size_gb}GB) will complete in background.'
                }, 202
            
            # For small exports, create synchronously
            try:
                zip_path, download_url = self.export_service.create_export_sync(
                    account_id=account_id,
                    export_job_id=job_id
                )
                
                # Send file
                return send_file(
                    zip_path,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name=f'judge-export-{datetime.now(timezone.utc).strftime("%Y%m%d")}.zip'
                )
            
            except Exception as e:
                return {
                    'error': 'Export creation failed',
                    'details': str(e),
                    'job_id': job_id
                }, 500
        
        except Exception as e:
            return {'error': f'Unexpected error: {str(e)}'}, 500

class ExportDownloadRouter(Resource):
    """Download completed export ZIP"""
    
    def get(self, job_id: str):
        """Download export file
        
        Returns:
            - 200 + ZIP file: If ready
            - 202: If still processing
            - 400: Invalid job_id
            - 404: Job not found
            - 410: Expired (> 7 days)
        """
        try:
            job_id_bytes = uuid.UUID(job_id).bytes
        except ValueError:
            return {'error': 'Invalid job_id format'}, 400
        
        export_job = db.session.query(ExportJob).filter_by(id=job_id_bytes).first()
        if not export_job:
            return {'error': 'Export job not found'}, 404
        
        # Check expiration
        if export_job.expires_at and datetime.now(timezone.utc) > export_job.expires_at:
            return {
                'error': 'Download link expired',
                'expires_at': export_job.expires_at.isoformat()
            }, 410
        
        # Check status
        if export_job.status == 'Processing':
            return {
                'status': 'processing',
                'job_id': job_id,
                'estimated_size_gb': export_job.estimated_size_gb,
                'message': 'Export still processing, check again in a few minutes'
            }, 202
        
        if export_job.status == 'Failed':
            return {
                'error': 'Export failed',
                'reason': export_job.error_message,
                'job_id': job_id
            }, 400
        
        if export_job.status != 'Completed':
            return {
                'error': f'Export not ready (status: {export_job.status})',
                'job_id': job_id
            }, 409
        
        # Return file
        if not export_job.file_path or not os.path.exists(export_job.file_path):
            return {'error': 'Export file not found on disk'}, 404
        
        # Mark as downloaded (for audit)
        if not export_job.downloaded_at:
            export_job.downloaded_at = datetime.now(timezone.utc)
            db.session.commit()
        
        return send_file(
            export_job.file_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'judge-export-{export_job.createdAt.strftime("%Y%m%d")}.zip'
        )

class ExportStatusRouter(Resource):
    """Check export job status"""
    
    def get(self, job_id: str):
        """Get export job status
        
        Returns:
            - 200 + status JSON
            - 404: Job not found
        """
        try:
            job_id_bytes = uuid.UUID(job_id).bytes
        except ValueError:
            return {'error': 'Invalid job_id format'}, 400
        
        export_job = db.session.query(ExportJob).filter_by(id=job_id_bytes).first()
        if not export_job:
            return {'error': 'Export job not found'}, 404
        
        return {
            'job_id': job_id,
            'status': export_job.status,
            'estimated_size_gb': export_job.estimated_size_gb,
            'actual_size_gb': export_job.actual_size_gb,
            'created_at': export_job.createdAt.isoformat() if export_job.createdAt else None,
            'expires_at': export_job.expires_at.isoformat() if export_job.expires_at else None,
            'error_message': export_job.error_message,
            'download_url': export_job.download_url if export_job.status == 'Completed' else None
        }, 200

