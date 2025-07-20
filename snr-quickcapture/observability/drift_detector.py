#!/usr/bin/env python3
"""
QuickCapture Drift Detector

Detect semantic, performance, and data quality drift in production systems.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics
import numpy as np
from enum import Enum
import psutil

from .metrics_collector import get_metrics_collector


class DriftType(Enum):
    """Types of drift that can be detected."""
    SEMANTIC = "semantic"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    TAG_USAGE = "tag_usage"
    VALIDATION_PATTERNS = "validation_patterns"


class DriftSeverity(Enum):
    """Drift severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Individual drift alert."""
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    timestamp: datetime
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    timestamp: datetime
    alerts: List[DriftAlert]
    summary: Dict[str, Any]
    drift_score: float  # 0.0 to 1.0, higher means more drift


class DriftDetector:
    """Production-grade drift detection system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.metrics_collector = get_metrics_collector()
        
        # Drift detection data
        self.baseline_data = {}
        self.current_data = defaultdict(list)
        self.drift_history = deque(maxlen=100)
        self.alert_handlers = []
        
        # Drift thresholds
        self.thresholds = {
            'semantic_coherence_drift': 0.1,  # 10% change
            'tag_quality_drift': 0.15,        # 15% change
            'validation_success_drift': 0.05, # 5% change
            'processing_latency_drift': 0.2,  # 20% change
            'error_rate_drift': 0.1,          # 10% change
            'tag_usage_drift': 0.25,          # 25% change
        }
        
        # Update thresholds from config
        if config and 'thresholds' in config:
            self.thresholds.update(config['thresholds'])
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Start drift detection
        self._start_drift_detection()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default drift detection configuration."""
        return {
            'detection_interval_seconds': 60,
            'baseline_window_hours': 24,
            'current_window_hours': 1,
            'min_data_points': 10,
            'enable_semantic_drift': True,
            'enable_performance_drift': True,
            'enable_quality_drift': True,
            'alert_on_drift': True,
            'thresholds': {
                'semantic_coherence_drift': 0.1,
                'tag_quality_drift': 0.15,
                'validation_success_drift': 0.05,
                'processing_latency_drift': 0.2,
                'error_rate_drift': 0.1,
                'tag_usage_drift': 0.25,
            }
        }
    
    def _start_drift_detection(self):
        """Start background drift detection."""
        def detect_drift():
            while True:
                try:
                    # Collect current data
                    self._collect_current_data()
                    
                    # Update baseline if needed
                    self._update_baseline()
                    
                    # Perform drift detection
                    if self.baseline_data:
                        report = self._detect_drift()
                        if report:
                            self.drift_history.append(report)
                            if self.config['alert_on_drift']:
                                self._trigger_drift_alerts(report)
                    
                    time.sleep(self.config['detection_interval_seconds'])
                except Exception as e:
                    print(f"Error in drift detection: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        thread = threading.Thread(target=detect_drift, daemon=True)
        thread.start()
    
    def _collect_current_data(self):
        """Collect current metrics data."""
        try:
            metrics = self.metrics_collector.get_current_metrics()
            
            with self.lock:
                # Core metrics
                self.current_data['semantic_coherence'].append(metrics.semantic_coherence_avg)
                self.current_data['tag_quality'].append(metrics.tag_quality_avg)
                self.current_data['validation_success_rate'].append(metrics.validation_success_rate)
                self.current_data['processing_latency'].append(metrics.processing_latency_avg)
                self.current_data['confidence_score'].append(metrics.confidence_score_avg)
                
                # Error metrics
                total_errors = sum(metrics.error_distribution.values())
                total_operations = metrics.database_stats.get('total_notes', 1)
                error_rate = total_errors / total_operations if total_operations > 0 else 0
                self.current_data['error_rate'].append(error_rate)
                
                # Storage performance
                if metrics.storage_performance:
                    avg_storage_latency = sum(metrics.storage_performance.values()) / len(metrics.storage_performance)
                    self.current_data['storage_latency'].append(avg_storage_latency)
                
                # Keep only recent data
                for key in self.current_data:
                    if len(self.current_data[key]) > 100:
                        self.current_data[key] = self.current_data[key][-100:]
        
        except Exception as e:
            print(f"Error collecting current data: {e}")
    
    def _update_baseline(self):
        """Update baseline data from historical metrics."""
        try:
            # Get metrics history for baseline period
            baseline_hours = self.config['baseline_window_hours']
            history = self.metrics_collector.get_metrics_history(minutes=baseline_hours * 60)
            
            if len(history) < self.config['min_data_points']:
                return  # Not enough data for baseline
            
            # Calculate baseline statistics
            baseline_metrics = {
                'semantic_coherence': [],
                'tag_quality': [],
                'validation_success_rate': [],
                'processing_latency': [],
                'confidence_score': [],
                'error_rate': [],
                'storage_latency': [],
            }
            
            for metrics in history:
                baseline_metrics['semantic_coherence'].append(metrics.semantic_coherence_avg)
                baseline_metrics['tag_quality'].append(metrics.tag_quality_avg)
                baseline_metrics['validation_success_rate'].append(metrics.validation_success_rate)
                baseline_metrics['processing_latency'].append(metrics.processing_latency_avg)
                baseline_metrics['confidence_score'].append(metrics.confidence_score_avg)
                
                # Calculate error rate for each historical point
                total_errors = sum(metrics.error_distribution.values())
                total_operations = metrics.database_stats.get('total_notes', 1)
                error_rate = total_errors / total_operations if total_operations > 0 else 0
                baseline_metrics['error_rate'].append(error_rate)
                
                # Storage latency
                if metrics.storage_performance:
                    avg_storage_latency = sum(metrics.storage_performance.values()) / len(metrics.storage_performance)
                    baseline_metrics['storage_latency'].append(avg_storage_latency)
            
            # Calculate baseline statistics
            with self.lock:
                self.baseline_data = {}
                for metric_name, values in baseline_metrics.items():
                    if values:
                        self.baseline_data[metric_name] = {
                            'mean': statistics.mean(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0,
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }
        
        except Exception as e:
            print(f"Error updating baseline: {e}")
    
    def _detect_drift(self) -> Optional[DriftReport]:
        """Perform comprehensive drift detection."""
        alerts = []
        
        with self.lock:
            for metric_name, current_values in self.current_data.items():
                if not current_values or metric_name not in self.baseline_data:
                    continue
                
                baseline = self.baseline_data[metric_name]
                current_avg = statistics.mean(current_values)
                
                # Calculate drift
                drift_alert = self._check_metric_drift(
                    metric_name, current_avg, baseline
                )
                
                if drift_alert:
                    alerts.append(drift_alert)
        
        if not alerts:
            return None
        
        # Calculate overall drift score
        drift_score = self._calculate_drift_score(alerts)
        
        # Create summary
        summary = {
            'total_alerts': len(alerts),
            'drift_types': list(set(alert.drift_type.value for alert in alerts)),
            'severity_distribution': self._get_severity_distribution(alerts),
            'most_affected_metrics': self._get_most_affected_metrics(alerts)
        }
        
        return DriftReport(
            timestamp=datetime.now(),
            alerts=alerts,
            summary=summary,
            drift_score=drift_score
        )
    
    def _check_metric_drift(self, metric_name: str, current_value: float, 
                           baseline: Dict[str, float]) -> Optional[DriftAlert]:
        """Check if a specific metric has drifted."""
        baseline_mean = baseline['mean']
        baseline_std = baseline['std']
        
        # Calculate change percentage
        if baseline_mean != 0:
            change_percent = abs(current_value - baseline_mean) / abs(baseline_mean)
        else:
            change_percent = 0.0
        
        # Determine drift type and threshold
        drift_type, threshold = self._get_drift_config(metric_name)
        
        if change_percent < threshold:
            return None
        
        # Determine severity
        severity = self._determine_severity(change_percent, threshold)
        
        # Create alert
        description = self._generate_drift_description(
            metric_name, current_value, baseline_mean, change_percent
        )
        
        return DriftAlert(
            drift_type=drift_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_mean,
            change_percent=change_percent,
            timestamp=datetime.now(),
            description=f"Drift detected: {metric_name} changed by {change_percent:.1%}",
            metadata={
                'baseline_std': baseline_std,
                'baseline_min': baseline['min'],
                'baseline_max': baseline['max'],
                'threshold': threshold
            }
        )
    
    def _get_drift_config(self, metric_name: str) -> Tuple[DriftType, float]:
        """Get drift type and threshold for a metric."""
        drift_configs = {
            'semantic_coherence': (DriftType.SEMANTIC, self.thresholds['semantic_coherence_drift']),
            'tag_quality': (DriftType.DATA_QUALITY, self.thresholds['tag_quality_drift']),
            'validation_success_rate': (DriftType.VALIDATION_PATTERNS, self.thresholds['validation_success_drift']),
            'processing_latency': (DriftType.PERFORMANCE, self.thresholds['processing_latency_drift']),
            'error_rate': (DriftType.PERFORMANCE, self.thresholds['error_rate_drift']),
            'storage_latency': (DriftType.PERFORMANCE, self.thresholds['processing_latency_drift']),
            'confidence_score': (DriftType.DATA_QUALITY, self.thresholds['tag_quality_drift']),
        }
        
        return drift_configs.get(metric_name, (DriftType.DATA_QUALITY, 0.1))
    
    def _determine_severity(self, change_percent: float, threshold: float) -> DriftSeverity:
        """Determine drift severity based on change percentage."""
        if change_percent > threshold * 3:
            return DriftSeverity.CRITICAL
        elif change_percent > threshold * 2:
            return DriftSeverity.HIGH
        elif change_percent > threshold * 1.5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _generate_drift_description(self, metric_name: str, current_value: float, 
                                  baseline_value: float, change_percent: float) -> str:
        """Generate human-readable drift description."""
        direction = "increased" if current_value > baseline_value else "decreased"
        
        descriptions = {
            'semantic_coherence': f"Semantic coherence {direction} by {change_percent:.1%}",
            'tag_quality': f"Tag quality {direction} by {change_percent:.1%}",
            'validation_success_rate': f"Validation success rate {direction} by {change_percent:.1%}",
            'processing_latency': f"Processing latency {direction} by {change_percent:.1%}",
            'error_rate': f"Error rate {direction} by {change_percent:.1%}",
            'storage_latency': f"Storage latency {direction} by {change_percent:.1%}",
            'confidence_score': f"Confidence score {direction} by {change_percent:.1%}",
        }
        
        return descriptions.get(metric_name, f"{metric_name} {direction} by {change_percent:.1%}")
    
    def _calculate_drift_score(self, alerts: List[DriftAlert]) -> float:
        """Calculate overall drift score (0.0 to 1.0)."""
        if not alerts:
            return 0.0
        
        severity_weights = {
            DriftSeverity.LOW: 0.1,
            DriftSeverity.MEDIUM: 0.3,
            DriftSeverity.HIGH: 0.6,
            DriftSeverity.CRITICAL: 1.0
        }
        
        total_score = sum(severity_weights[alert.severity] * alert.change_percent for alert in alerts)
        return min(total_score / len(alerts), 1.0)
    
    def _get_severity_distribution(self, alerts: List[DriftAlert]) -> Dict[str, int]:
        """Get distribution of alert severities."""
        distribution = defaultdict(int)
        for alert in alerts:
            distribution[alert.severity.value] += 1
        return dict(distribution)
    
    def _get_most_affected_metrics(self, alerts: List[DriftAlert]) -> List[str]:
        """Get list of most affected metrics."""
        metric_scores = defaultdict(float)
        for alert in alerts:
            metric_scores[alert.metric_name] += alert.change_percent
        
        # Sort by total change and return top 3
        sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        return [metric for metric, _ in sorted_metrics[:3]]
    
    def _trigger_drift_alerts(self, report: DriftReport):
        """Trigger drift alert handlers."""
        alert_data = {
            'timestamp': report.timestamp.isoformat(),
            'drift_score': report.drift_score,
            'summary': report.summary,
            'alerts': [
                {
                    'type': alert.drift_type.value,
                    'severity': alert.severity.value,
                    'metric': alert.metric_name,
                    'current_value': alert.current_value,
                    'baseline_value': alert.baseline_value,
                    'change_percent': alert.change_percent,
                    'description': alert.description
                }
                for alert in report.alerts
            ]
        }
        
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                print(f"Drift alert handler error: {e}")
    
    def add_alert_handler(self, handler):
        """Add a drift alert handler."""
        self.alert_handlers.append(handler)
    
    def get_current_drift_report(self) -> Optional[DriftReport]:
        """Get the most recent drift report."""
        if self.drift_history:
            return self.drift_history[-1]
        return None
    
    def get_drift_history(self, hours: int = 24) -> List[DriftReport]:
        """Get drift history for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [report for report in self.drift_history if report.timestamp > cutoff_time]
    
    def export_drift_data(self) -> Dict[str, Any]:
        """Export drift detection data for analysis."""
        return {
            'baseline_data': self.baseline_data,
            'current_data': dict(self.current_data),
            'drift_history': [
                {
                    'timestamp': report.timestamp.isoformat(),
                    'drift_score': report.drift_score,
                    'summary': report.summary,
                    'alerts': [
                        {
                            'type': alert.drift_type.value,
                            'severity': alert.severity.value,
                            'metric': alert.metric_name,
                            'current_value': alert.current_value,
                            'baseline_value': alert.baseline_value,
                            'change_percent': alert.change_percent,
                            'description': alert.description
                        }
                        for alert in report.alerts
                    ]
                }
                for report in self.drift_history
            ],
            'thresholds': self.thresholds,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def reset_drift_data(self):
        """Reset all drift detection data (useful for testing)."""
        with self.lock:
            self.baseline_data.clear()
            self.current_data.clear()
            self.drift_history.clear()


# Global drift detector instance
_drift_detector: Optional[DriftDetector] = None


def get_drift_detector(config: Optional[Dict[str, Any]] = None) -> DriftDetector:
    """Get or create the global drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector(config)
    return _drift_detector


def get_current_drift_report() -> Optional[DriftReport]:
    """Convenience function to get current drift report."""
    detector = get_drift_detector()
    return detector.get_current_drift_report()


def export_drift_data() -> Dict[str, Any]:
    """Convenience function to export drift data."""
    detector = get_drift_detector()
    return detector.export_drift_data() 