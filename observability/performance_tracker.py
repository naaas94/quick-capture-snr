
#!/usr/bin/env python3
"""
QuickCapture Performance Tracker

Detailed performance monitoring and bottleneck detection for production systems.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics
import psutil
import os
import sys

from .metrics_collector import get_metrics_collector


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: datetime
    metrics: List[PerformanceMetric]
    system_info: Dict[str, Any]
    bottlenecks: List[str] = field(default_factory=list)


@dataclass
class PerformanceProfile:
    """Performance profile for a specific operation."""
    operation: str
    avg_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    call_count: int
    error_count: int
    success_rate: float


class PerformanceTracker:
    """Production-grade performance tracking system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.metrics_collector = get_metrics_collector()
        
        # Performance data storage
        self.operation_times = defaultdict(list)
        self.operation_errors = defaultdict(int)
        self.operation_calls = defaultdict(int)
        self.system_metrics = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        
        # Performance thresholds
        self.thresholds = {
            'parsing_latency_max_ms': 50,
            'validation_latency_max_ms': 100,
            'storage_latency_max_ms': 50,
            'total_processing_max_ms': 200,
            'memory_usage_max_mb': 512,
            'cpu_usage_max_percent': 80,
            'disk_io_max_mb_per_sec': 100,
        }
        
        # Update thresholds from config
        if config and 'thresholds' in config:
            self.thresholds.update(config['thresholds'])
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Background monitoring flag
        self._monitoring_enabled = self.config.get('enable_system_monitoring', True)
        
        # Start background monitoring only if enabled
        if self._monitoring_enabled:
            self._start_background_monitoring()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default performance tracking configuration."""
        return {
            'monitoring_interval_seconds': 5,
            'history_size': 100,
            'enable_system_monitoring': True,
            'enable_operation_profiling': True,
            'bottleneck_detection': True,
            'thresholds': {
                'parsing_latency_max_ms': 50,
                'validation_latency_max_ms': 100,
                'storage_latency_max_ms': 50,
                'total_processing_max_ms': 200,
                'memory_usage_max_mb': 512,
                'cpu_usage_max_percent': 80,
                'disk_io_max_mb_per_sec': 100,
            }
        }
    
    def _start_background_monitoring(self):
        """Start background performance monitoring."""
        # Only start background monitoring if enabled in config
        if not self.config.get('enable_system_monitoring', True):
            return
            
        def monitor_performance():
            while True:
                try:
                    if self.config['enable_system_monitoring']:
                        self._collect_system_metrics()
                    
                    if self.config.get('bottleneck_detection', True):
                        self._detect_bottlenecks()
                    
                    time.sleep(self.config['monitoring_interval_seconds'])
                except Exception as e:
                    print(f"Error in performance monitoring: {e}")
                    time.sleep(30)  # Wait longer on error
        
        thread = threading.Thread(target=monitor_performance, daemon=True)
        thread.start()
    
    def track_operation(self, operation: str):
        """Context manager for tracking operation performance."""
        return OperationTracker(self, operation)
    
    def record_operation(self, operation: str, duration_ms: float, success: bool = True, 
                        metadata: Optional[Dict[str, Any]] = None):
        """Record operation performance."""
        with self.lock:
            self.operation_times[operation].append(duration_ms)
            self.operation_calls[operation] += 1
            
            if not success:
                self.operation_errors[operation] += 1
            
            # Keep only recent data for performance
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation] = self.operation_times[operation][-1000:]
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        try:
            # CPU usage - use non-blocking call or default value
            try:
                cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
            except:
                cpu_percent = 0.0  # Default if psutil fails
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Process-specific metrics
            process = psutil.Process(os.getpid())
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            try:
                process_cpu_percent = process.cpu_percent(interval=None)  # Non-blocking
            except:
                process_cpu_percent = 0.0  # Default if psutil fails
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_bytes_sent = network_io.bytes_sent
            network_bytes_recv = network_io.bytes_recv
            
            metrics = [
                PerformanceMetric("cpu_percent", cpu_percent, "percent", datetime.now()),
                PerformanceMetric("memory_usage_mb", memory_usage_mb, "MB", datetime.now()),
                PerformanceMetric("process_memory_mb", process_memory_mb, "MB", datetime.now()),
                PerformanceMetric("process_cpu_percent", process_cpu_percent, "percent", datetime.now()),
                PerformanceMetric("disk_read_mb", disk_read_mb, "MB", datetime.now()),
                PerformanceMetric("disk_write_mb", disk_write_mb, "MB", datetime.now()),
                PerformanceMetric("network_bytes_sent", network_bytes_sent, "bytes", datetime.now()),
                PerformanceMetric("network_bytes_recv", network_bytes_recv, "bytes", datetime.now()),
            ]
            
            system_info = {
                'cpu_count': psutil.cpu_count(),
                'memory_total_mb': memory.total / (1024 * 1024),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'platform': sys.platform,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            }
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                metrics=metrics,
                system_info=system_info
            )
            
            self.system_metrics.append(snapshot)
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    def _detect_bottlenecks(self):
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        with self.lock:
            # Check operation latencies
            for operation, times in self.operation_times.items():
                if not times:
                    continue
                
                avg_time = statistics.mean(times)
                threshold_key = f"{operation}_latency_max_ms"
                
                if threshold_key in self.thresholds and avg_time > self.thresholds[threshold_key]:
                    bottlenecks.append(f"{operation}: {avg_time:.1f}ms avg (threshold: {self.thresholds[threshold_key]}ms)")
            
            # Check system resources
            if self.system_metrics:
                latest = self.system_metrics[-1]
                
                for metric in latest.metrics:
                    if metric.name == "cpu_percent" and metric.value > self.thresholds['cpu_usage_max_percent']:
                        bottlenecks.append(f"CPU: {metric.value:.1f}% (threshold: {self.thresholds['cpu_usage_max_percent']}%)")
                    
                    elif metric.name == "memory_usage_mb" and metric.value > self.thresholds['memory_usage_max_mb']:
                        bottlenecks.append(f"Memory: {metric.value:.0f}MB (threshold: {self.thresholds['memory_usage_max_mb']}MB)")
                    
                    elif metric.name == "disk_read_mb" and metric.value > self.thresholds['disk_io_max_mb_per_sec']:
                        bottlenecks.append(f"Disk I/O: {metric.value:.1f}MB/s (threshold: {self.thresholds['disk_io_max_mb_per_sec']}MB/s)")
            
            # Update latest snapshot with bottlenecks
            if self.system_metrics:
                self.system_metrics[-1].bottlenecks = bottlenecks
    
    def get_operation_profile(self, operation: str, lock_held: bool = False) -> Optional[PerformanceProfile]:
        """Get performance profile for a specific operation."""
        if not lock_held:
            with self.lock:
                return self._get_operation_profile_internal(operation)
        else:
            return self._get_operation_profile_internal(operation)
    
    def _get_operation_profile_internal(self, operation: str) -> Optional[PerformanceProfile]:
        """Internal method to get operation profile (assumes lock is held)."""
        if operation not in self.operation_times or not self.operation_times[operation]:
            return None
        
        times = self.operation_times[operation]
        call_count = self.operation_calls[operation]
        error_count = self.operation_errors[operation]
        
        return PerformanceProfile(
            operation=operation,
            avg_duration_ms=statistics.mean(times),
            p95_duration_ms=statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            p99_duration_ms=statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
            min_duration_ms=min(times),
            max_duration_ms=max(times),
            call_count=call_count,
            error_count=error_count,
            success_rate=(call_count - error_count) / call_count if call_count > 0 else 1.0
        )
    
    def get_all_profiles(self) -> Dict[str, PerformanceProfile]:
        """Get performance profiles for all operations."""
        profiles = {}
        with self.lock:
            for operation in self.operation_times.keys():
                profile = self._get_operation_profile_internal(operation)  # Use internal method since lock is held
                if profile:
                    profiles[operation] = profile
        return profiles
    
    def get_system_metrics_history(self, minutes: int = 60) -> List[PerformanceSnapshot]:
        """Get system metrics history for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [snapshot for snapshot in self.system_metrics if snapshot.timestamp > cutoff_time]
    
    def get_current_system_metrics(self) -> Optional[PerformanceSnapshot]:
        """Get current system metrics."""
        if self.system_metrics:
            return self.system_metrics[-1]
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        profiles = self.get_all_profiles()
        current_metrics = self.get_current_system_metrics()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'operations': {
                name: {
                    'avg_duration_ms': profile.avg_duration_ms,
                    'p95_duration_ms': profile.p95_duration_ms,
                    'call_count': profile.call_count,
                    'success_rate': profile.success_rate
                }
                for name, profile in profiles.items()
            },
            'system': {
                'cpu_percent': 0.0,
                'memory_usage_mb': 0.0,
                'bottlenecks': []
            }
        }
        
        if current_metrics:
            summary['system']['bottlenecks'] = current_metrics.bottlenecks
            for metric in current_metrics.metrics:
                if metric.name == "cpu_percent":
                    summary['system']['cpu_percent'] = metric.value
                elif metric.name == "memory_usage_mb":
                    summary['system']['memory_usage_mb'] = metric.value
        
        return summary
    
    def export_performance_data(self) -> Dict[str, Any]:
        """Export performance data for analysis."""
        profiles = self.get_all_profiles()
        system_history = list(self.system_metrics)
        
        return {
            'profiles': {
                name: {
                    'avg_duration_ms': profile.avg_duration_ms,
                    'p95_duration_ms': profile.p95_duration_ms,
                    'p99_duration_ms': profile.p99_duration_ms,
                    'min_duration_ms': profile.min_duration_ms,
                    'max_duration_ms': profile.max_duration_ms,
                    'call_count': profile.call_count,
                    'error_count': profile.error_count,
                    'success_rate': profile.success_rate
                }
                for name, profile in profiles.items()
            },
            'system_history': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'metrics': {
                        metric.name: {
                            'value': metric.value,
                            'unit': metric.unit
                        }
                        for metric in snapshot.metrics
                    },
                    'bottlenecks': snapshot.bottlenecks
                }
                for snapshot in system_history
            ],
            'thresholds': self.thresholds,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def reset_performance_data(self):
        """Reset all performance data (useful for testing)."""
        with self.lock:
            self.operation_times.clear()
            self.operation_errors.clear()
            self.operation_calls.clear()
            self.system_metrics.clear()
            self.performance_history.clear()


class OperationTracker:
    """Context manager for tracking operation performance."""
    
    def __init__(self, tracker: PerformanceTracker, operation: str):
        self.tracker = tracker
        self.operation = operation
        self.start_time = None
        self.metadata = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None
            self.tracker.record_operation(self.operation, duration_ms, success, self.metadata)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the operation."""
        self.metadata[key] = value


# Global performance tracker instance
_performance_tracker: Optional[PerformanceTracker] = None


def reset_global_tracker():
    """Reset the global performance tracker (useful for testing)."""
    global _performance_tracker
    _performance_tracker = None


def get_performance_tracker(config: Optional[Dict[str, Any]] = None) -> PerformanceTracker:
    """Get or create the global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker(config)
    return _performance_tracker


def track_operation(operation: str):
    """Convenience function to track operation performance."""
    tracker = get_performance_tracker()
    return tracker.track_operation(operation)


def record_operation(operation: str, duration_ms: float, success: bool = True, 
                   metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to record operation performance."""
    tracker = get_performance_tracker()
    tracker.record_operation(operation, duration_ms, success, metadata)


def get_performance_summary() -> Dict[str, Any]:
    """Convenience function to get performance summary."""
    tracker = get_performance_tracker()
    return tracker.get_performance_summary() 