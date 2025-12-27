#!/usr/bin/env python3
"""
QuickCapture Metrics Collector

Comprehensive metrics collection for production-grade monitoring and observability.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available. Metrics will be collected locally only.")


@dataclass
class QuickCaptureMetrics:
    """Comprehensive metrics for QuickCapture system."""
    
    # Core metrics
    ingestion_rate: float = 0.0
    validation_success_rate: float = 0.0
    semantic_coherence_avg: float = 0.0
    tag_quality_avg: float = 0.0
    processing_latency_avg: float = 0.0
    
    # Storage metrics
    storage_performance: Dict[str, float] = field(default_factory=dict)
    database_stats: Dict[str, int] = field(default_factory=dict)
    
    # Error metrics
    error_distribution: Dict[str, int] = field(default_factory=dict)
    validation_issues: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    semantic_density_avg: float = 0.0
    confidence_score_avg: float = 0.0
    tag_suggestion_accuracy: float = 0.0
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Production-grade metrics collector with Prometheus integration."""
    
    def __init__(self, prometheus_port: Optional[int] = None):
        self.metrics = QuickCaptureMetrics()
        self.history = deque(maxlen=1000)  # Keep last 1000 data points
        self.lock = threading.Lock()
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and prometheus_port:
            self._init_prometheus_metrics()
            try:
                start_http_server(prometheus_port)
                print(f"Prometheus metrics server started on port {prometheus_port}")
            except Exception as e:
                print(f"Failed to start Prometheus server: {e}")
        
        # Local metrics storage
        self._ingestion_counter = 0
        self._validation_successes = 0
        self._validation_failures = 0
        self._processing_times = deque(maxlen=100)
        self._semantic_scores = deque(maxlen=100)
        self._tag_quality_scores = deque(maxlen=100)
        self._confidence_scores = deque(maxlen=100)
        self._error_counts = defaultdict(int)
        self._validation_issue_counts = defaultdict(int)
        
        # Start background collection
        self._start_background_collection()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.prometheus_metrics = {
            'notes_ingested_total': Counter('quickcapture_notes_ingested_total', 'Total notes ingested'),
            'notes_validated_total': Counter('quickcapture_notes_validated_total', 'Total notes validated'),
            'validation_success_total': Counter('quickcapture_validation_success_total', 'Successful validations'),
            'validation_failure_total': Counter('quickcapture_validation_failure_total', 'Failed validations'),
            'processing_duration_seconds': Histogram('quickcapture_processing_duration_seconds', 'Note processing duration'),
            'semantic_coherence_score': Histogram('quickcapture_semantic_coherence_score', 'Semantic coherence scores'),
            'tag_quality_score': Histogram('quickcapture_tag_quality_score', 'Tag quality scores'),
            'confidence_score': Histogram('quickcapture_confidence_score', 'Confidence scores'),
            'semantic_density': Histogram('quickcapture_semantic_density', 'Semantic density scores'),
            'storage_operation_duration_seconds': Histogram('quickcapture_storage_operation_duration_seconds', 'Storage operation duration'),
            'active_notes_gauge': Gauge('quickcapture_active_notes', 'Number of active notes'),
            'total_tags_gauge': Gauge('quickcapture_total_tags', 'Total number of unique tags'),
            'error_rate': Gauge('quickcapture_error_rate', 'Error rate percentage'),
        }
    
    def _start_background_collection(self):
        """Start background metrics collection."""
        def collect_metrics():
            while True:
                try:
                    self._update_metrics()
                    time.sleep(15)  # Update every 15 seconds
                except Exception as e:
                    print(f"Error in background metrics collection: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()
    
    def record_note_ingestion(self, processing_time: float, semantic_coherence: float, 
                            tag_quality: float, confidence: float, semantic_density: float,
                            validation_success: bool, issues: Optional[List[str]] = None):
        """Record metrics for a single note ingestion."""
        with self.lock:
            # Update counters
            self._ingestion_counter += 1
            if validation_success:
                self._validation_successes += 1
            else:
                self._validation_failures += 1
            
            # Update collections
            self._processing_times.append(processing_time)
            self._semantic_scores.append(semantic_coherence)
            self._tag_quality_scores.append(tag_quality)
            self._confidence_scores.append(confidence)
            
            # Record issues
            if issues:
                for issue in issues:
                    self._validation_issue_counts[issue] += 1
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and hasattr(self, 'prometheus_metrics'):
                self.prometheus_metrics['notes_ingested_total'].inc()
                self.prometheus_metrics['notes_validated_total'].inc()
                
                if validation_success:
                    self.prometheus_metrics['validation_success_total'].inc()
                else:
                    self.prometheus_metrics['validation_failure_total'].inc()
                
                self.prometheus_metrics['processing_duration_seconds'].observe(processing_time)
                self.prometheus_metrics['semantic_coherence_score'].observe(semantic_coherence)
                self.prometheus_metrics['tag_quality_score'].observe(tag_quality)
                self.prometheus_metrics['confidence_score'].observe(confidence)
                self.prometheus_metrics['semantic_density'].observe(semantic_density)
    
    def record_storage_operation(self, operation: str, duration: float, success: bool):
        """Record storage operation metrics."""
        with self.lock:
            if not hasattr(self, '_storage_times'):
                self._storage_times = defaultdict(list)
            
            self._storage_times[operation].append(duration)
            
            if not success:
                self._error_counts[f"storage_{operation}"] += 1
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and hasattr(self, 'prometheus_metrics'):
                self.prometheus_metrics['storage_operation_duration_seconds'].observe(duration)
    
    def record_error(self, error_type: str, error_message: str = ""):
        """Record error metrics."""
        with self.lock:
            self._error_counts[error_type] += 1
            
            # Update Prometheus error rate
            if PROMETHEUS_AVAILABLE and hasattr(self, 'prometheus_metrics'):
                total_errors = sum(self._error_counts.values())
                total_operations = self._ingestion_counter
                if total_operations > 0:
                    error_rate = (total_errors / total_operations) * 100
                    self.prometheus_metrics['error_rate'].set(error_rate)
    
    def update_database_stats(self, total_notes: int, total_tags: int):
        """Update database statistics."""
        with self.lock:
            self.metrics.database_stats = {
                'total_notes': total_notes,
                'total_tags': total_tags
            }
            
            # Update Prometheus gauges
            if PROMETHEUS_AVAILABLE and hasattr(self, 'prometheus_metrics'):
                self.prometheus_metrics['active_notes_gauge'].set(total_notes)
                self.prometheus_metrics['total_tags_gauge'].set(total_tags)
    
    def _update_metrics(self):
        """Update computed metrics."""
        with self.lock:
            # Calculate averages
            if self._processing_times:
                self.metrics.processing_latency_avg = sum(self._processing_times) / len(self._processing_times)
            
            if self._semantic_scores:
                self.metrics.semantic_coherence_avg = sum(self._semantic_scores) / len(self._semantic_scores)
            
            if self._tag_quality_scores:
                self.metrics.tag_quality_avg = sum(self._tag_quality_scores) / len(self._tag_quality_scores)
            
            if self._confidence_scores:
                self.metrics.confidence_score_avg = sum(self._confidence_scores) / len(self._confidence_scores)
            
            # Calculate rates
            total_validations = self._validation_successes + self._validation_failures
            if total_validations > 0:
                self.metrics.validation_success_rate = self._validation_successes / total_validations
            
            # Calculate ingestion rate (notes per minute)
            if self._ingestion_counter > 0:
                # This is a simplified calculation - in production you'd want time-based windows
                self.metrics.ingestion_rate = self._ingestion_counter / 60.0  # Simplified
            
            # Update storage performance
            if hasattr(self, '_storage_times'):
                for operation, times in self._storage_times.items():
                    if times:
                        self.metrics.storage_performance[operation] = sum(times) / len(times)
            
            # Update error distribution
            self.metrics.error_distribution = dict(self._error_counts)
            
            # Update validation issues
            self.metrics.validation_issues = dict(self._validation_issue_counts)
            
            # Update timestamp
            self.metrics.timestamp = datetime.now()
            
            # Store in history
            self.history.append(self.metrics)
    
    def get_current_metrics(self) -> QuickCaptureMetrics:
        """Get current metrics snapshot."""
        with self.lock:
            return self.metrics
    
    def get_metrics_history(self, minutes: int = 60) -> List[QuickCaptureMetrics]:
        """Get metrics history for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.history if m.timestamp > cutoff_time]
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics as JSON-serializable dictionary."""
        metrics = self.get_current_metrics()
        return {
            'ingestion_rate': metrics.ingestion_rate,
            'validation_success_rate': metrics.validation_success_rate,
            'semantic_coherence_avg': metrics.semantic_coherence_avg,
            'tag_quality_avg': metrics.tag_quality_avg,
            'processing_latency_avg': metrics.processing_latency_avg,
            'storage_performance': metrics.storage_performance,
            'database_stats': metrics.database_stats,
            'error_distribution': metrics.error_distribution,
            'validation_issues': metrics.validation_issues,
            'semantic_density_avg': metrics.semantic_density_avg,
            'confidence_score_avg': metrics.confidence_score_avg,
            'tag_suggestion_accuracy': metrics.tag_suggestion_accuracy,
            'memory_usage_mb': metrics.memory_usage_mb,
            'cpu_usage_percent': metrics.cpu_usage_percent,
            'timestamp': metrics.timestamp.isoformat(),
            'total_notes_processed': self._ingestion_counter,
            'total_validation_successes': self._validation_successes,
            'total_validation_failures': self._validation_failures
        }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self.lock:
            self._ingestion_counter = 0
            self._validation_successes = 0
            self._validation_failures = 0
            self._processing_times.clear()
            self._semantic_scores.clear()
            self._tag_quality_scores.clear()
            self._confidence_scores.clear()
            self._error_counts.clear()
            self._validation_issue_counts.clear()
            self.history.clear()
            if hasattr(self, '_storage_times'):
                self._storage_times.clear()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(prometheus_port: Optional[int] = None) -> MetricsCollector:
    """Get or create the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(prometheus_port)
    return _metrics_collector


def record_note_ingestion(processing_time: float, semantic_coherence: float, 
                         tag_quality: float, confidence: float, semantic_density: float,
                         validation_success: bool, issues: Optional[List[str]] = None):
    """Convenience function to record note ingestion metrics."""
    collector = get_metrics_collector()
    collector.record_note_ingestion(processing_time, semantic_coherence, tag_quality, 
                                  confidence, semantic_density, validation_success, issues)


def record_storage_operation(operation: str, duration: float, success: bool):
    """Convenience function to record storage operation metrics."""
    collector = get_metrics_collector()
    collector.record_storage_operation(operation, duration, success)


def record_error(error_type: str, error_message: str = ""):
    """Convenience function to record error metrics."""
    collector = get_metrics_collector()
    collector.record_error(error_type, error_message) 