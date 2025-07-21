#!/usr/bin/env python3
"""
Debug script to test get_operation_profile without locks
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from observability.performance_tracker import PerformanceTracker

def test_without_locks():
    """Test get_operation_profile without using locks."""
    
    print("=== Testing get_operation_profile without locks ===")
    
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
    tracker.record_operation("op1", 100.0, True)
    tracker.record_operation("op2", 200.0, True)
    print("✓ Added test data")
    
    # Test direct access to data
    print("\n3. Testing direct data access...")
    try:
        print(f"  - Operation times: {tracker.operation_times}")
        print(f"  - Operation calls: {tracker.operation_calls}")
        print(f"  - Operation errors: {tracker.operation_errors}")
        print("✓ Direct data access successful")
    except Exception as e:
        print(f"✗ Direct data access failed: {e}")
        return
    
    # Test get_operation_profile with lock_held=True
    print("\n4. Testing get_operation_profile with lock_held=True...")
    try:
        profile = tracker.get_operation_profile("op1", lock_held=True)
        if profile:
            print(f"✓ Profile created: {profile.operation}")
            print(f"  - Avg: {profile.avg_duration_ms:.1f}ms")
            print(f"  - P95: {profile.p95_duration_ms:.1f}ms")
            print(f"  - P99: {profile.p99_duration_ms:.1f}ms")
        else:
            print("✗ No profile returned")
    except Exception as e:
        print(f"✗ Error in get_operation_profile: {e}")
        return
    
    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    test_without_locks() 