#!/usr/bin/env python3
"""
Debug script for get_performance_summary method
Tests the method step by step to identify where it hangs
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from observability.performance_tracker import PerformanceTracker, reset_global_tracker

def debug_get_performance_summary():
    """Debug the get_performance_summary method step by step."""
    
    print("=== Starting debug of get_performance_summary ===")
    
    # Step 1: Create tracker with test config
    print("\n1. Creating PerformanceTracker with test config...")
    test_config = {
        'monitoring_interval_seconds': 5,
        'history_size': 100,
        'enable_system_monitoring': False,  # Disable for tests
        'enable_operation_profiling': True,
        'bottleneck_detection': False,  # Disable for tests
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
    
    try:
        tracker = PerformanceTracker(test_config)
        print("✓ PerformanceTracker created successfully")
    except Exception as e:
        print(f"✗ Error creating PerformanceTracker: {e}")
        return
    
    # Step 2: Reset performance data
    print("\n2. Resetting performance data...")
    try:
        tracker.reset_performance_data()
        print("✓ Performance data reset successfully")
    except Exception as e:
        print(f"✗ Error resetting performance data: {e}")
        return
    
    # Step 3: Add test data
    print("\n3. Adding test data...")
    try:
        tracker.record_operation("op1", 100.0, True)
        print("✓ Added op1: 100.0ms")
        tracker.record_operation("op2", 200.0, True)
        print("✓ Added op2: 200.0ms")
    except Exception as e:
        print(f"✗ Error adding test data: {e}")
        return
    
    # Step 4: Test get_all_profiles
    print("\n4. Testing get_all_profiles...")
    try:
        profiles = tracker.get_all_profiles()
        print(f"✓ get_all_profiles returned {len(profiles)} profiles")
        for name, profile in profiles.items():
            print(f"  - {name}: {profile.call_count} calls, {profile.avg_duration_ms:.1f}ms avg")
    except Exception as e:
        print(f"✗ Error in get_all_profiles: {e}")
        return
    
    # Step 5: Test get_current_system_metrics
    print("\n5. Testing get_current_system_metrics...")
    try:
        current_metrics = tracker.get_current_system_metrics()
        if current_metrics:
            print(f"✓ get_current_system_metrics returned snapshot with {len(current_metrics.metrics)} metrics")
        else:
            print("✓ get_current_system_metrics returned None (expected when system monitoring is disabled)")
    except Exception as e:
        print(f"✗ Error in get_current_system_metrics: {e}")
        return
    
    # Step 6: Test get_performance_summary
    print("\n6. Testing get_performance_summary...")
    try:
        print("  - Calling get_performance_summary()...")
        start_time = time.time()
        summary = tracker.get_performance_summary()
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        
        print(f"✓ get_performance_summary completed in {duration:.1f}ms")
        print(f"  - Summary keys: {list(summary.keys())}")
        print(f"  - Operations: {list(summary['operations'].keys())}")
        print(f"  - System keys: {list(summary['system'].keys())}")
        
    except Exception as e:
        print(f"✗ Error in get_performance_summary: {e}")
        return
    
    print("\n=== Debug completed successfully! ===")

def debug_global_tracker():
    """Debug the global tracker version."""
    
    print("\n=== Testing Global Tracker Version ===")
    
    # Reset global tracker
    print("\n1. Resetting global tracker...")
    try:
        reset_global_tracker()
        print("✓ Global tracker reset")
    except Exception as e:
        print(f"✗ Error resetting global tracker: {e}")
        return
    
    # Create global tracker with test config
    print("\n2. Creating global tracker with test config...")
    try:
        from observability.performance_tracker import get_performance_tracker
        test_config = {
            'enable_system_monitoring': False,
            'bottleneck_detection': False,
        }
        tracker = get_performance_tracker(test_config)
        print("✓ Global tracker created")
    except Exception as e:
        print(f"✗ Error creating global tracker: {e}")
        return
    
    # Test global convenience function
    print("\n3. Testing global get_performance_summary()...")
    try:
        from observability.performance_tracker import get_performance_summary
        print("  - Calling global get_performance_summary()...")
        start_time = time.time()
        summary = get_performance_summary()
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        
        print(f"✓ Global get_performance_summary completed in {duration:.1f}ms")
        print(f"  - Summary keys: {list(summary.keys())}")
        
    except Exception as e:
        print(f"✗ Error in global get_performance_summary: {e}")
        return
    
    print("\n=== Global tracker debug completed! ===")

if __name__ == "__main__":
    print("Starting performance summary debug...")
    
    # Test instance method
    debug_get_performance_summary()
    
    # Test global tracker
    debug_global_tracker()
    
    print("\nAll debug tests completed!") 