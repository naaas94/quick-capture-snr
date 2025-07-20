#!/usr/bin/env python3
"""
Debug script for get_all_profiles method specifically
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from observability.performance_tracker import PerformanceTracker

def debug_get_all_profiles():
    """Debug the get_all_profiles method specifically."""
    
    print("=== Starting debug of get_all_profiles ===")
    
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
    
    # Test get_all_profiles step by step
    print("\n3. Testing get_all_profiles step by step...")
    
    print("  - About to acquire lock...")
    try:
        with tracker.lock:
            print("  ✓ Lock acquired")
            
            print("  - Checking operation_times keys...")
            operations = list(tracker.operation_times.keys())
            print(f"  ✓ Operations found: {operations}")
            
            profiles = {}
            for operation in operations:
                print(f"  - Processing operation: {operation}")
                
                # Test get_operation_profile for this operation
                print(f"    - Calling _get_operation_profile_internal for {operation}...")
                profile = tracker._get_operation_profile_internal(operation)
                
                if profile:
                    print(f"    ✓ Profile created for {operation}")
                    profiles[operation] = profile
                else:
                    print(f"    ✗ No profile for {operation}")
            
            print(f"  ✓ Created {len(profiles)} profiles")
            
        print("  ✓ Lock released")
        
    except Exception as e:
        print(f"  ✗ Error in get_all_profiles: {e}")
        return
    
    print("\n4. Testing full get_all_profiles method...")
    try:
        print("  - Calling get_all_profiles()...")
        start_time = time.time()
        all_profiles = tracker.get_all_profiles()
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        
        print(f"  ✓ get_all_profiles completed in {duration:.1f}ms")
        print(f"  - Returned {len(all_profiles)} profiles")
        for name, profile in all_profiles.items():
            print(f"    - {name}: {profile.call_count} calls, {profile.avg_duration_ms:.1f}ms avg")
        
    except Exception as e:
        print(f"  ✗ Error in get_all_profiles: {e}")
        return
    
    print("\n=== Debug completed successfully! ===")

if __name__ == "__main__":
    debug_get_all_profiles() 