#!/usr/bin/env python3
"""
Debug script for get_operation_profile method
Tests the method step by step to identify where it hangs
"""

import sys
import time
import statistics
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from observability.performance_tracker import PerformanceTracker

def debug_get_operation_profile():
    """Debug the get_operation_profile method step by step."""
    
    print("=== Starting debug of get_operation_profile ===")
    
    # Create tracker
    print("\n1. Creating PerformanceTracker...")
    test_config = {
        'enable_system_monitoring': False,
        'bottleneck_detection': False,
    }
    tracker = PerformanceTracker(test_config)
    tracker.reset_performance_data()
    
    # Add test data
    print("\n2. Adding test data...")
    tracker.record_operation("test_op", 100.0, True)
    tracker.record_operation("test_op", 200.0, True)
    print("✓ Added 2 operations")
    
    # Test each step of get_operation_profile
    print("\n3. Testing get_operation_profile step by step...")
    
    with tracker.lock:
        print("  - Acquired lock")
        
        operation = "test_op"
        print(f"  - Operation: {operation}")
        
        if operation not in tracker.operation_times:
            print("  - Operation not in operation_times")
            return
        else:
            print("  - Operation found in operation_times")
        
        if not tracker.operation_times[operation]:
            print("  - No times for operation")
            return
        else:
            print("  - Times found for operation")
        
        times = tracker.operation_times[operation]
        print(f"  - Times: {times}")
        
        call_count = tracker.operation_calls[operation]
        print(f"  - Call count: {call_count}")
        
        error_count = tracker.operation_errors[operation]
        print(f"  - Error count: {error_count}")
        
        # Test statistics.mean
        print("  - Testing statistics.mean...")
        try:
            avg_duration_ms = statistics.mean(times)
            print(f"    ✓ Mean: {avg_duration_ms}")
        except Exception as e:
            print(f"    ✗ Error in statistics.mean: {e}")
            return
        
        # Test statistics.quantiles for p95
        print("  - Testing statistics.quantiles for p95...")
        try:
            if len(times) >= 20:
                p95_duration_ms = statistics.quantiles(times, n=20)[18]
                print(f"    ✓ P95 (n=20): {p95_duration_ms}")
            else:
                p95_duration_ms = max(times)
                print(f"    ✓ P95 (max): {p95_duration_ms}")
        except Exception as e:
            print(f"    ✗ Error in p95 calculation: {e}")
            return
        
        # Test statistics.quantiles for p99
        print("  - Testing statistics.quantiles for p99...")
        try:
            if len(times) >= 100:
                p99_duration_ms = statistics.quantiles(times, n=100)[98]
                print(f"    ✓ P99 (n=100): {p99_duration_ms}")
            else:
                p99_duration_ms = max(times)
                print(f"    ✓ P99 (max): {p99_duration_ms}")
        except Exception as e:
            print(f"    ✗ Error in p99 calculation: {e}")
            return
        
        # Test min/max
        print("  - Testing min/max...")
        try:
            min_duration_ms = min(times)
            max_duration_ms = max(times)
            print(f"    ✓ Min: {min_duration_ms}, Max: {max_duration_ms}")
        except Exception as e:
            print(f"    ✗ Error in min/max: {e}")
            return
        
        # Test success rate
        print("  - Testing success rate...")
        try:
            success_rate = (call_count - error_count) / call_count if call_count > 0 else 1.0
            print(f"    ✓ Success rate: {success_rate}")
        except Exception as e:
            print(f"    ✗ Error in success rate: {e}")
            return
        
        print("  - All calculations completed successfully!")
    
    print("\n4. Testing full get_operation_profile method...")
    try:
        profile = tracker.get_operation_profile("test_op")
        if profile:
            print(f"✓ get_operation_profile returned profile for {profile.operation}")
            print(f"  - Avg: {profile.avg_duration_ms:.1f}ms")
            print(f"  - P95: {profile.p95_duration_ms:.1f}ms")
            print(f"  - P99: {profile.p99_duration_ms:.1f}ms")
        else:
            print("✗ get_operation_profile returned None")
    except Exception as e:
        print(f"✗ Error in get_operation_profile: {e}")
        return
    
    print("\n=== Debug completed successfully! ===")

if __name__ == "__main__":
    debug_get_operation_profile() 