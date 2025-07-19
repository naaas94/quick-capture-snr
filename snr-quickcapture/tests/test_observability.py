#!/usr/bin/env python3
"""
Tests for QuickCapture Observability Modules

Comprehensive testing of metrics collection, health monitoring, performance tracking,
and drift detection systems.
"""

import pytest
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from observability.metrics_collector import (
    MetricsCollector, QuickCaptureMetrics, 
    get_metrics_collector, record_note_ingestion,
    record_storage_operation, record_error
)

from observability.health_monitor import (
    HealthMonitor, HealthStatus, HealthCheck, SystemHealth,
    get_health_monitor, check_system_health
)

from observability.performance_tracker import (
    PerformanceTracker, PerformanceMetric, PerformanceSnapshot, PerformanceProfile,
    get_performance_tracker, track_operation, record_operation, get_performance_summary
)

from observability.drift_detector import (
    DriftDetector, DriftType, DriftSeverity, DriftAlert, DriftReport,
    get_drift_detector, get_current_drift_report, export_drift_data
)


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.collector = MetricsCollector()
        self.collector.reset_metrics()
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        assert self.collector.metrics.ingestion_rate == 0.0
        assert self.collector.metrics.validation_success_rate == 0.0
        assert len(self.collector.history) == 0
    
    def test_record_note_ingestion(self):
        """Test recording note ingestion metrics."""
        self.collector.record_note_ingestion(
            processing_time=0.1,
            semantic_coherence=0.8,
            tag_quality=0.9,
            confidence=0.85,
            semantic_density=0.7,
            validation_success=True,
            issues=["minor_issue"]
        )
        
        assert self.collector._ingestion_counter == 1
        assert self.collector._validation_successes == 1
        assert self.collector._validation_failures == 0
        assert len(self.collector._processing_times) == 1
        assert len(self.collector._semantic_scores) == 1
    
    def test_record_storage_operation(self):
        """Test recording storage operation metrics."""
        self.collector.record_storage_operation("store_note", 0.05, True)
        
        assert len(self.collector._storage_times["store_note"]) == 1
        assert self.collector._error_counts == {}
    
    def test_record_error(self):
        """Test recording error metrics."""
        self.collector.record_error("parsing_error", "Invalid format")
        
        assert self.collector._error_counts["parsing_error"] == 1
    
    def test_update_database_stats(self):
        """Test updating database statistics."""
        self.collector.update_database_stats(100, 25)
        
        assert self.collector.metrics.database_stats["total_notes"] == 100
        assert self.collector.metrics.database_stats["total_tags"] == 25
    
    def test_export_metrics(self):
        """Test metrics export functionality."""
        # Add some test data
        self.collector.record_note_ingestion(0.1, 0.8, 0.9, 0.85, 0.7, True)
        self.collector.update_database_stats(100, 25)
        
        exported = self.collector.export_metrics()
        
        assert "ingestion_rate" in exported
        assert "database_stats" in exported
        assert exported["total_notes_processed"] == 1
        assert exported["database_stats"]["total_notes"] == 100
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test global collector
        global_collector = get_metrics_collector()
        assert global_collector is not None
        
        # Test convenience functions
        record_note_ingestion(0.1, 0.8, 0.9, 0.85, 0.7, True)
        record_storage_operation("test_op", 0.05, True)
        record_error("test_error", "Test error message")


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.monitor = HealthMonitor()
    
    @patch('observability.health_monitor.psutil')
    def test_health_monitor_initialization(self, mock_psutil):
        """Test health monitor initialization."""
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value.used = 1024 * 1024 * 100  # 100MB
        mock_psutil.virtual_memory.return_value.total = 1024 * 1024 * 1024  # 1GB
        mock_psutil.disk_io_counters.return_value.read_bytes = 1024 * 1024
        mock_psutil.disk_io_counters.return_value.write_bytes = 1024 * 1024
        mock_psutil.net_io_counters.return_value.bytes_sent = 1024
        mock_psutil.net_io_counters.return_value.bytes_recv = 1024
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.disk_usage.return_value.total = 1024**3  # 1GB
        
        health = self.monitor.check_system_health()
        
        assert isinstance(health, SystemHealth)
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        assert len(health.checks) > 0
    
    def test_health_check_creation(self):
        """Test health check creation."""
        check = HealthCheck(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="Test check passed",
            timestamp=datetime.now(),
            duration_ms=10.0
        )
        
        assert check.name == "test_check"
        assert check.status == HealthStatus.HEALTHY
        assert check.message == "Test check passed"
    
    def test_health_score_calculation(self):
        """Test health score calculation."""
        checks = [
            HealthCheck("check1", HealthStatus.HEALTHY, "OK", datetime.now(), 10.0),
            HealthCheck("check2", HealthStatus.DEGRADED, "Warning", datetime.now(), 10.0),
            HealthCheck("check3", HealthStatus.HEALTHY, "OK", datetime.now(), 10.0)
        ]
        
        score = self.monitor._calculate_health_score(checks)
        assert 0.0 <= score <= 1.0
    
    def test_export_health_report(self):
        """Test health report export."""
        health = self.monitor.check_system_health()
        report = self.monitor.export_health_report()
        
        assert "status" in report
        assert "score" in report
        assert "checks" in report
        assert isinstance(report["checks"], list)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test global monitor
        global_monitor = get_health_monitor()
        assert global_monitor is not None
        
        # Test health check
        health = check_system_health()
        assert isinstance(health, SystemHealth)


class TestPerformanceTracker:
    """Test performance tracking functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tracker = PerformanceTracker()
        self.tracker.reset_performance_data()
    
    def test_performance_tracker_initialization(self):
        """Test performance tracker initialization."""
        assert len(self.tracker.operation_times) == 0
        assert len(self.tracker.operation_errors) == 0
        assert len(self.tracker.operation_calls) == 0
    
    def test_record_operation(self):
        """Test recording operation performance."""
        self.tracker.record_operation("test_op", 100.0, True)
        
        assert len(self.tracker.operation_times["test_op"]) == 1
        assert self.tracker.operation_calls["test_op"] == 1
        assert self.tracker.operation_errors["test_op"] == 0
    
    def test_record_operation_failure(self):
        """Test recording operation failure."""
        self.tracker.record_operation("test_op", 100.0, False)
        
        assert self.tracker.operation_errors["test_op"] == 1
    
    def test_get_operation_profile(self):
        """Test getting operation profile."""
        # Add some test data
        for i in range(10):
            self.tracker.record_operation("test_op", 100.0 + i, True)
        
        profile = self.tracker.get_operation_profile("test_op")
        
        assert profile is not None
        assert profile.operation == "test_op"
        assert profile.call_count == 10
        assert profile.error_count == 0
        assert profile.success_rate == 1.0
        assert profile.avg_duration_ms > 0
    
    def test_track_operation_context_manager(self):
        """Test operation tracking context manager."""
        with self.tracker.track_operation("context_op") as tracker:
            tracker.add_metadata("test_key", "test_value")
            time.sleep(0.01)  # Simulate some work
        
        profile = self.tracker.get_operation_profile("context_op")
        assert profile is not None
        assert profile.call_count == 1
        assert profile.success_rate == 1.0
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        # Add test data
        self.tracker.record_operation("op1", 100.0, True)
        self.tracker.record_operation("op2", 200.0, True)
        
        summary = self.tracker.get_performance_summary()
        
        assert "timestamp" in summary
        assert "operations" in summary
        assert "system" in summary
        assert "op1" in summary["operations"]
        assert "op2" in summary["operations"]
    
    def test_export_performance_data(self):
        """Test performance data export."""
        # Add test data
        self.tracker.record_operation("test_op", 100.0, True)
        
        exported = self.tracker.export_performance_data()
        
        assert "profiles" in exported
        assert "system_history" in exported
        assert "thresholds" in exported
        assert "test_op" in exported["profiles"]
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test global tracker
        global_tracker = get_performance_tracker()
        assert global_tracker is not None
        
        # Test convenience functions
        record_operation("test_op", 100.0, True)
        summary = get_performance_summary()
        assert "operations" in summary


class TestDriftDetector:
    """Test drift detection functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.detector = DriftDetector()
        self.detector.reset_drift_data()
    
    def test_drift_detector_initialization(self):
        """Test drift detector initialization."""
        assert len(self.detector.baseline_data) == 0
        assert len(self.detector.current_data) == 0
        assert len(self.detector.drift_history) == 0
    
    def test_drift_alert_creation(self):
        """Test drift alert creation."""
        alert = DriftAlert(
            drift_type=DriftType.SEMANTIC,
            severity=DriftSeverity.MEDIUM,
            metric_name="test_metric",
            current_value=0.8,
            baseline_value=0.6,
            change_percent=0.33,
            timestamp=datetime.now(),
            description="Test drift detected"
        )
        
        assert alert.drift_type == DriftType.SEMANTIC
        assert alert.severity == DriftSeverity.MEDIUM
        assert alert.metric_name == "test_metric"
        assert alert.change_percent == 0.33
    
    def test_drift_report_creation(self):
        """Test drift report creation."""
        alerts = [
            DriftAlert(
                DriftType.SEMANTIC, DriftSeverity.MEDIUM, "test_metric",
                0.8, 0.6, 0.33, datetime.now(), "Test drift"
            )
        ]
        
        report = DriftReport(
            timestamp=datetime.now(),
            alerts=alerts,
            summary={"total_alerts": 1},
            drift_score=0.5
        )
        
        assert len(report.alerts) == 1
        assert report.drift_score == 0.5
        assert report.summary["total_alerts"] == 1
    
    def test_drift_severity_determination(self):
        """Test drift severity determination."""
        threshold = 0.1
        
        # Test different change percentages
        assert self.detector._determine_severity(0.05, threshold) == DriftSeverity.LOW
        assert self.detector._determine_severity(0.15, threshold) == DriftSeverity.MEDIUM
        assert self.detector._determine_severity(0.25, threshold) == DriftSeverity.HIGH
        assert self.detector._determine_severity(0.35, threshold) == DriftSeverity.CRITICAL
    
    def test_drift_description_generation(self):
        """Test drift description generation."""
        description = self.detector._generate_drift_description(
            "semantic_coherence", 0.8, 0.6, 0.33
        )
        
        assert "semantic coherence" in description.lower()
        assert "33%" in description
    
    def test_export_drift_data(self):
        """Test drift data export."""
        exported = self.detector.export_drift_data()
        
        assert "baseline_data" in exported
        assert "current_data" in exported
        assert "drift_history" in exported
        assert "thresholds" in exported
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test global detector
        global_detector = get_drift_detector()
        assert global_detector is not None
        
        # Test convenience functions
        report = get_current_drift_report()
        assert report is None  # No drift detected yet
        
        exported = export_drift_data()
        assert "baseline_data" in exported


class TestIntegration:
    """Integration tests for observability modules."""
    
    def test_metrics_and_health_integration(self):
        """Test integration between metrics and health monitoring."""
        # Record some metrics
        record_note_ingestion(0.1, 0.8, 0.9, 0.85, 0.7, True)
        record_storage_operation("test_op", 0.05, True)
        
        # Check health
        health = check_system_health()
        assert isinstance(health, SystemHealth)
    
    def test_performance_and_metrics_integration(self):
        """Test integration between performance tracking and metrics."""
        # Record performance data
        with track_operation("test_integration"):
            time.sleep(0.01)
        
        # Get performance summary
        summary = get_performance_summary()
        assert "operations" in summary
    
    def test_all_modules_initialization(self):
        """Test that all observability modules can be initialized together."""
        # Initialize all modules
        metrics = get_metrics_collector()
        health = get_health_monitor()
        performance = get_performance_tracker()
        drift = get_drift_detector()
        
        assert metrics is not None
        assert health is not None
        assert performance is not None
        assert drift is not None


if __name__ == "__main__":
    pytest.main([__file__]) 