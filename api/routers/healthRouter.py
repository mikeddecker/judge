import os
from datetime import datetime, timezone
from flask import current_app, jsonify
from flask_restful import Resource
from repository.db import db
import logging

logger = logging.getLogger(__name__)

class HealthRouter(Resource):
    """liveness probe: Is the API running?"""
    
    def get(self):
        """Basic connectivity check"""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}, 200

class ReadinessRouter(Resource):
    """Readiness probe: Can the API serve traffic?"""
    
    def get(self):
        """Check if all dependencies are reachable"""
        try:
            # Test database connectivity
            db.session.execute("SELECT 1")
            db_status = "connected"
        except Exception as e:
            logger.error(f"Database connectivity check failed: {e}")
            return {"ready": False, "database": "disconnected", "error": str(e)}, 503
        
        # Check external services (if configured)
        external_services = []
        
        # Check if sync hosts are configured (future: ping them)
        regions = os.getenv('ALL_REGIONS', 'belgium,usa').split(',')
        for region in regions:
            region = region.strip()
            host_var = f"REGION_{region.upper()}_HOST"
            host = os.getenv(host_var)
            if host:
                # TODO: Implement actual SSH connectivity check
                external_services.append({
                    "name": f"sync-gateway-{region}",
                    "status": "configured"  # Would be "healthy" after implementation
                })
        
        return {
            "ready": True,
            "database": db_status,
            "external_services": external_services,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, 200

class ReplicationLagRouter(Resource):
    """Database replication lag (production-only)"""
    
    def get(self):
        """Check replication lag from primary MySQL"""
        try:
            # Only allow in production
            if os.getenv('FLASK_ENV') != 'production':
                return {"error": "Only available in production"}, 403
            
            # Query replication status
            result = db.session.execute(
                "SHOW SLAVE STATUS"
            ).fetchone()
            
            if not result:
                # Not a replica
                return {
                    "is_replica": False,
                    "primary": os.getenv('REGION', 'unknown'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, 200
            
            # Extract lag (Seconds_Behind_Master)
            lag_seconds = result.get('Seconds_Behind_Master')
            print("lag_seconds", lag_seconds)
            
            if lag_seconds is None:
                lag_seconds = 0  # Assumed synced if not available
            
            response = {
                "replica_lag_seconds": lag_seconds,
                "primary": os.getenv('REGION', 'unknown'),
                "is_synced": lag_seconds < 5,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Return 503 if lag is too high
            if lag_seconds > 5:
                return {
                    **response,
                    "error": f"Replica lag exceeds 5 seconds ({lag_seconds}s)"
                }, 503
            
            return response, 200
            
        except Exception as e:
            logger.error(f"Replication lag check failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, 503

class MetricsRouter(Resource):
    """Prometheus metrics endpoint for monitoring
    
    Exposes metrics in Prometheus text format.
    Restricted to internal IPs only (configure via METRICS_ALLOWED_IPS).
    """
    
    def get(self):
        """Return Prometheus-format metrics"""
        from flask import request
        
        # Check if request is from allowed IPs (optional security)
        allowed_ips = os.getenv('METRICS_ALLOWED_IPS', '127.0.0.1,::1').split(',')
        allowed_ips = [ip.strip() for ip in allowed_ips]
        
        # Disable IP check if set to '*' (development only)
        if allowed_ips != ['*']:
            client_ip = request.remote_addr
            if client_ip not in allowed_ips:
                logger.warning(f"Metrics access denied from {client_ip}")
                return {"error": "Access denied"}, 403
        
        try:
            metrics_lines = self._collect_metrics()
            # Return Prometheus text format
            return self._format_prometheus_response(metrics_lines), 200, {'Content-Type': 'text/plain; charset=utf-8'}
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {"error": str(e)}, 500
    
    def _collect_metrics(self) -> list:
        """Collect all metrics from the application
        
        Returns list of Prometheus metric lines (type + sample + value)
        """
        metrics = []
        
        # Helper function to add metric
        def add_metric(name: str, help_text: str, metric_type: str, value: float, labels: dict = None):
            labels_str = ''
            if labels:
                label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
                labels_str = '{' + ','.join(label_pairs) + '}'
            
            if not metrics or metrics[-1][0] != name:
                metrics.append((name, help_text, metric_type, []))
            metrics[-1][3].append((labels_str, value))
        
        try:
            # Database metrics
            result = db.session.execute("SELECT COUNT(*) FROM Video").scalar()
            add_metric('videos_total', 'Total videos in database', 'gauge', float(result or 0))
            
            result = db.session.execute("SELECT COUNT(*) FROM FrameLabel").scalar()
            add_metric('frame_labels_total', 'Total labeled frames', 'gauge', float(result or 0))
            
            result = db.session.execute("SELECT COUNT(*) FROM Account").scalar()
            add_metric('accounts_total', 'Total accounts', 'gauge', float(result or 0))
            
            result = db.session.execute("SELECT COUNT(*) FROM Jobs WHERE status='Created'").scalar()
            add_metric('pending_jobs', 'Pending jobs in queue', 'gauge', float(result or 0), {'status': 'Created'})
            
            result = db.session.execute("SELECT COUNT(*) FROM Jobs WHERE job_category='AI'").scalar()
            add_metric('jobs_by_category_total', 'Jobs by category', 'gauge', float(result or 0), {'category': 'AI'})
            
            result = db.session.execute("SELECT COUNT(*) FROM Jobs WHERE job_category='SYNC'").scalar()
            add_metric('jobs_by_category_total', 'Jobs by category', 'gauge', float(result or 0), {'category': 'SYNC'})
            
            result = db.session.execute("SELECT COUNT(*) FROM Jobs WHERE job_category='BACKUP'").scalar()
            add_metric('jobs_by_category_total', 'Jobs by category', 'gauge', float(result or 0), {'category': 'BACKUP'})
            
            result = db.session.execute("SELECT COUNT(*) FROM ConflictLog WHERE auto_resolved=FALSE").scalar()
            add_metric('unresolved_conflicts', 'Unresolved conflicts requiring user action', 'gauge', float(result or 0))
            
            result = db.session.execute("SELECT COUNT(*) FROM DeletedVideos").scalar()
            add_metric('soft_deleted_videos', 'Soft-deleted videos (30-day grace period)', 'gauge', float(result or 0))
            
        except Exception as e:
            logger.warning(f"Error collecting database metrics: {e}")
        
        try:
            # Storage metrics (if accessible)
            storage_dir = os.getenv('STORAGE_DIR_VIDEOS')
            if storage_dir and os.path.exists(storage_dir):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(storage_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except:
                            pass
                
                # Convert bytes to gigabytes
                total_size_gb = total_size / (1024**3)
                add_metric('storage_videos_gb', 'Total video storage usage in GB', 'gauge', total_size_gb)
            
            generated_dir = os.getenv('STORAGE_DIR_GENERATED_DATA')
            if generated_dir and os.path.exists(generated_dir):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(generated_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except:
                            pass
                
                # Convert bytes to gigabytes
                total_size_gb = total_size / (1024**3)
                add_metric('storage_generated_gb', 'Total generated data storage usage in GB', 'gauge', total_size_gb)
        
        except Exception as e:
            logger.warning(f"Error collecting storage metrics: {e}")
        
        # System metrics
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            add_metric('process_cpu_percent', 'CPU usage percentage', 'gauge', float(cpu_percent))
            
            memory = psutil.virtual_memory()
            add_metric('process_memory_mb', 'Memory usage in MB', 'gauge', float(memory.used / (1024**2)))
            add_metric('process_memory_percent', 'Memory usage percentage', 'gauge', float(memory.percent))
        except ImportError:
            logger.debug("psutil not available, skipping system metrics")
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
        
        # Application uptime (approximated from Flask start time)
        try:
            start_time = current_app.config.get('START_TIME')
            if start_time:
                uptime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
                add_metric('uptime_seconds', 'Application uptime in seconds', 'gauge', float(uptime_seconds))
        except Exception as e:
            logger.warning(f"Error calculating uptime: {e}")
        
        return metrics
    
    def _format_prometheus_response(self, metrics: list) -> str:
        """Format metrics in Prometheus text format
        
        Format:
            # HELP metric_name description
            # TYPE metric_name metric_type
            metric_name{labels} value
        """
        lines = []
        
        for metric_name, help_text, metric_type, samples in metrics:
            # HELP line
            lines.append(f"# HELP {metric_name} {help_text}")
            # TYPE line
            lines.append(f"# TYPE {metric_name} {metric_type}")
            # Sample lines
            for labels_str, value in samples:
                if labels_str:
                    lines.append(f"{metric_name}{labels_str} {value}")
                else:
                    lines.append(f"{metric_name} {value}")
        
        lines.append("")  # Final newline
        return '\n'.join(lines)

