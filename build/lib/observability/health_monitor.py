#!/usr/bin/env python3
"""
QuickCapture Health Monitor

Production-grade health monitoring and alerting system for QuickCapture.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path

import psutil

from .metrics_collector import get_metrics_collector, QuickCaptureMetrics


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime
    overall_score: float
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class HealthMonitor:
    """Production-grade health monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.health_history = []
        self.alert_handlers: List[Callable] = []
        self.metrics_collector = get_metrics_collector()
        
        # Health thresholds
        self.thresholds = {
            'error_rate_max': 0.05,  # 5% error rate
            'latency_max_ms': 200,   # 200ms max latency
            'validation_success_min': 0.95,  # 95% validation success
            'semantic_coherence_min': 0.7,   # 0.7 min semantic coherence
            'tag_quality_min': 0.8,          # 0.8 min tag quality
            'storage_latency_max_ms': 50,    # 50ms max storage latency
            'memory_usage_max_mb': 512,      # 512MB max memory
            'cpu_usage_max_percent': 80,     # 80% max CPU
        }
        
        # Update thresholds from config
        if config and 'thresholds' in config:
            self.thresholds.update(config['thresholds'])
        
        # Start monitoring
        self._start_monitoring()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default health monitoring configuration."""
        return {
            'check_interval_seconds': 30,
            'history_size': 100,
            'alert_on_degraded': True,
            'alert_on_unhealthy': True,
            'alert_on_critical': True,
            'thresholds': {
                'error_rate_max': 0.05,
                'latency_max_ms': 200,
                'validation_success_min': 0.95,
                'semantic_coherence_min': 0.7,
                'tag_quality_min': 0.8,
                'storage_latency_max_ms': 50,
                'memory_usage_max_mb': 512,
                'cpu_usage_max_percent': 80,
            }
        }
    
    def _start_monitoring(self):
        """Start background health monitoring."""
        def monitor_health():
            while True:
                try:
                    health = self.check_system_health()
                    self.health_history.append(health)
                    
                    # Keep history size manageable
                    if len(self.health_history) > self.config['history_size']:
                        self.health_history.pop(0)
                    
                    # Trigger alerts if needed
                    self._check_alerts(health)
                    
                    time.sleep(self.config['check_interval_seconds'])
                except Exception as e:
                    print(f"Error in health monitoring: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=monitor_health, daemon=True)
        thread.start()
    
    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive health check."""
        start_time = time.time()
        checks = []
        critical_issues = []
        warnings = []
        
        # Check 1: Database connectivity
        db_check = self._check_database_health()
        checks.append(db_check)
        if db_check.status == HealthStatus.CRITICAL:
            critical_issues.append(f"Database: {db_check.message}")
        elif db_check.status == HealthStatus.DEGRADED:
            warnings.append(f"Database: {db_check.message}")
        
        # Check 2: Storage performance
        storage_check = self._check_storage_health()
        checks.append(storage_check)
        if storage_check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            critical_issues.append(f"Storage: {storage_check.message}")
        elif storage_check.status == HealthStatus.DEGRADED:
            warnings.append(f"Storage: {storage_check.message}")
        
        # Check 3: Validation performance
        validation_check = self._check_validation_health()
        checks.append(validation_check)
        if validation_check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            critical_issues.append(f"Validation: {validation_check.message}")
        elif validation_check.status == HealthStatus.DEGRADED:
            warnings.append(f"Validation: {validation_check.message}")
        
        # Check 4: Semantic quality
        semantic_check = self._check_semantic_health()
        checks.append(semantic_check)
        if semantic_check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            critical_issues.append(f"Semantic: {semantic_check.message}")
        elif semantic_check.status == HealthStatus.DEGRADED:
            warnings.append(f"Semantic: {semantic_check.message}")
        
        # Check 5: Error rates
        error_check = self._check_error_health()
        checks.append(error_check)
        if error_check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            critical_issues.append(f"Errors: {error_check.message}")
        elif error_check.status == HealthStatus.DEGRADED:
            warnings.append(f"Errors: {error_check.message}")
        
        # Check 6: System resources
        resource_check = self._check_resource_health()
        checks.append(resource_check)
        if resource_check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            critical_issues.append(f"Resources: {resource_check.message}")
        elif resource_check.status == HealthStatus.DEGRADED:
            warnings.append(f"Resources: {resource_check.message}")
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        overall_score = self._calculate_health_score(checks)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return SystemHealth(
            status=overall_status,
            checks=checks,
            timestamp=datetime.now(),
            overall_score=overall_score,
            critical_issues=critical_issues,
            warnings=warnings
        )
    
    def _check_database_health(self) -> HealthCheck:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            # Try to connect to database
            db_path = Path("storage/quickcapture.db")
            if not db_path.exists():
                return HealthCheck(
                    name="database_connectivity",
                    status=HealthStatus.CRITICAL,
                    message="Database file not found",
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Test connection and basic operations
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notes'")
            if not cursor.fetchone():
                return HealthCheck(
                    name="database_connectivity",
                    status=HealthStatus.CRITICAL,
                    message="Notes table not found",
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Check table size
            cursor.execute("SELECT COUNT(*) FROM notes")
            note_count = cursor.fetchone()[0]
            
            # Check for recent activity
            cursor.execute("SELECT COUNT(*) FROM notes WHERE timestamp > datetime('now', '-1 hour')")
            recent_notes = cursor.fetchone()[0]
            
            conn.close()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status based on activity
            if recent_notes == 0 and note_count > 0:
                status = HealthStatus.DEGRADED
                message = f"No recent activity (last hour). Total notes: {note_count}"
            elif recent_notes > 0:
                status = HealthStatus.HEALTHY
                message = f"Active. {recent_notes} notes in last hour, {note_count} total"
            else:
                status = HealthStatus.DEGRADED
                message = f"No notes found. Database empty."
            
            return HealthCheck(
                name="database_connectivity",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details={
                    'total_notes': note_count,
                    'recent_notes': recent_notes,
                    'database_size_mb': db_path.stat().st_size / (1024 * 1024)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="database_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Database error: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _check_storage_health(self) -> HealthCheck:
        """Check storage performance."""
        start_time = time.time()
        
        try:
            metrics = self.metrics_collector.get_current_metrics()
            storage_perf = metrics.storage_performance
            
            if not storage_perf:
                return HealthCheck(
                    name="storage_performance",
                    status=HealthStatus.DEGRADED,
                    message="No storage performance data available",
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Check average storage latency
            avg_latency = sum(storage_perf.values()) / len(storage_perf)
            max_latency = max(storage_perf.values())
            
            if max_latency > self.thresholds['storage_latency_max_ms']:
                status = HealthStatus.UNHEALTHY
                message = f"High storage latency: {max_latency:.1f}ms (max: {self.thresholds['storage_latency_max_ms']}ms)"
            elif avg_latency > self.thresholds['storage_latency_max_ms'] * 0.7:
                status = HealthStatus.DEGRADED
                message = f"Elevated storage latency: {avg_latency:.1f}ms avg"
            else:
                status = HealthStatus.HEALTHY
                message = f"Storage performance good: {avg_latency:.1f}ms avg"
            
            return HealthCheck(
                name="storage_performance",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    'avg_latency_ms': avg_latency,
                    'max_latency_ms': max_latency,
                    'operations': list(storage_perf.keys())
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="storage_performance",
                status=HealthStatus.CRITICAL,
                message=f"Storage check error: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _check_validation_health(self) -> HealthCheck:
        """Check validation performance and success rates."""
        start_time = time.time()
        
        try:
            metrics = self.metrics_collector.get_current_metrics()
            
            if metrics.validation_success_rate == 0:
                return HealthCheck(
                    name="validation_performance",
                    status=HealthStatus.DEGRADED,
                    message="No validation data available",
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            success_rate = metrics.validation_success_rate
            
            if success_rate < self.thresholds['validation_success_min']:
                status = HealthStatus.UNHEALTHY
                message = f"Low validation success rate: {success_rate:.1%} (min: {self.thresholds['validation_success_min']:.1%})"
            elif success_rate < self.thresholds['validation_success_min'] + 0.02:
                status = HealthStatus.DEGRADED
                message = f"Validation success rate below optimal: {success_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Validation performance good: {success_rate:.1%} success rate"
            
            return HealthCheck(
                name="validation_performance",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    'success_rate': success_rate,
                    'total_validations': metrics.database_stats.get('total_notes', 0),
                    'avg_processing_latency_ms': metrics.processing_latency_avg * 1000
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="validation_performance",
                status=HealthStatus.CRITICAL,
                message=f"Validation check error: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _check_semantic_health(self) -> HealthCheck:
        """Check semantic quality metrics."""
        start_time = time.time()
        
        try:
            metrics = self.metrics_collector.get_current_metrics()
            
            semantic_coherence = metrics.semantic_coherence_avg
            tag_quality = metrics.tag_quality_avg
            
            issues = []
            if semantic_coherence < self.thresholds['semantic_coherence_min']:
                issues.append(f"Low semantic coherence: {semantic_coherence:.2f}")
            
            if tag_quality < self.thresholds['tag_quality_min']:
                issues.append(f"Low tag quality: {tag_quality:.2f}")
            
            if issues:
                status = HealthStatus.DEGRADED
                message = f"Quality issues: {'; '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Semantic quality good: coherence={semantic_coherence:.2f}, tags={tag_quality:.2f}"
            
            return HealthCheck(
                name="semantic_quality",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    'semantic_coherence': semantic_coherence,
                    'tag_quality': tag_quality,
                    'confidence_score': metrics.confidence_score_avg
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="semantic_quality",
                status=HealthStatus.CRITICAL,
                message=f"Semantic check error: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _check_error_health(self) -> HealthCheck:
        """Check error rates and distribution."""
        start_time = time.time()
        
        try:
            metrics = self.metrics_collector.get_current_metrics()
            error_dist = metrics.error_distribution
            
            if not error_dist:
                return HealthCheck(
                    name="error_rates",
                    status=HealthStatus.HEALTHY,
                    message="No errors detected",
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            total_errors = sum(error_dist.values())
            total_operations = metrics.database_stats.get('total_notes', 1)
            error_rate = total_errors / total_operations if total_operations > 0 else 0
            
            if error_rate > self.thresholds['error_rate_max']:
                status = HealthStatus.UNHEALTHY
                message = f"High error rate: {error_rate:.1%} (max: {self.thresholds['error_rate_max']:.1%})"
            elif error_rate > self.thresholds['error_rate_max'] * 0.7:
                status = HealthStatus.DEGRADED
                message = f"Elevated error rate: {error_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Error rate acceptable: {error_rate:.1%}"
            
            return HealthCheck(
                name="error_rates",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    'error_rate': error_rate,
                    'total_errors': total_errors,
                    'error_distribution': error_dist
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="error_rates",
                status=HealthStatus.CRITICAL,
                message=f"Error check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _check_resource_health(self) -> HealthCheck:
        """Check system resource usage."""
        start_time = time.time()
        
        try:
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            memory_usage_mb = memory.used / (1024 * 1024)
            
            issues = []
            if memory_usage_mb > self.thresholds['memory_usage_max_mb']:
                issues.append(f"High memory usage: {memory_usage_mb:.0f}MB")
            
            if cpu_percent > self.thresholds['cpu_usage_max_percent']:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if issues:
                status = HealthStatus.DEGRADED
                message = f"Resource issues: {'; '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resources good: CPU={cpu_percent:.1f}%, Memory={memory_usage_mb:.0f}MB"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_usage_mb': memory_usage_mb,
                    'memory_percent': memory.percent
                }
            )
            
        except ImportError:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.DEGRADED,
                message="psutil not available - cannot check system resources",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Resource check error: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system status based on individual checks."""
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.UNHEALTHY for check in checks):
            return HealthStatus.UNHEALTHY
        elif any(check.status == HealthStatus.DEGRADED for check in checks):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _calculate_health_score(self, checks: List[HealthCheck]) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        if not checks:
            return 0.0
        
        status_scores = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.7,
            HealthStatus.UNHEALTHY: 0.3,
            HealthStatus.CRITICAL: 0.0
        }
        
        total_score = sum(status_scores[check.status] for check in checks)
        return total_score / len(checks)
    
    def _check_alerts(self, health: SystemHealth):
        """Check if alerts should be triggered."""
        if health.status == HealthStatus.CRITICAL and self.config['alert_on_critical']:
            self._trigger_alert("CRITICAL", health)
        elif health.status == HealthStatus.UNHEALTHY and self.config['alert_on_unhealthy']:
            self._trigger_alert("UNHEALTHY", health)
        elif health.status == HealthStatus.DEGRADED and self.config['alert_on_degraded']:
            self._trigger_alert("DEGRADED", health)
    
    def _trigger_alert(self, level: str, health: SystemHealth):
        """Trigger alert handlers."""
        alert_data = {
            'level': level,
            'timestamp': health.timestamp.isoformat(),
            'score': health.overall_score,
            'critical_issues': health.critical_issues,
            'warnings': health.warnings,
            'checks': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message
                }
                for check in health.checks
            ]
        }
        
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                print(f"Alert handler error: {e}")
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def get_current_health(self) -> SystemHealth:
        """Get current system health."""
        if self.health_history:
            return self.health_history[-1]
        else:
            return self.check_system_health()
    
    def get_health_history(self, minutes: int = 60) -> List[SystemHealth]:
        """Get health history for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [h for h in self.health_history if h.timestamp > cutoff_time]
    
    def export_health_report(self) -> Dict[str, Any]:
        """Export health report as JSON-serializable dictionary."""
        health = self.get_current_health()
        return {
            'status': health.status.value,
            'score': health.overall_score,
            'timestamp': health.timestamp.isoformat(),
            'critical_issues': health.critical_issues,
            'warnings': health.warnings,
            'checks': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'duration_ms': check.duration_ms,
                    'details': check.details
                }
                for check in health.checks
            ]
        }


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor(config: Optional[Dict[str, Any]] = None) -> HealthMonitor:
    """Get or create the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor(config)
    return _health_monitor


def check_system_health() -> SystemHealth:
    """Convenience function to check system health."""
    monitor = get_health_monitor()
    return monitor.check_system_health() 