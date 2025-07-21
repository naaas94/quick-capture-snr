#!/usr/bin/env python3
"""
Simple debug script to test basic functionality
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_operations():
    """Test basic operations without threading."""
    
    print("=== Testing Basic Operations ===")
    
    # Test 1: Import
    print("\n1. Testing import...")
    try:
        from observability.performance_tracker import PerformanceTracker
        print("✓ Import successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Test 2: Create tracker
    print("\n2. Testing tracker creation...")
    try:
        tracker = PerformanceTracker({
            'enable_system_monitoring': False,
            'bottleneck_detection': False,
        })
        print("✓ Tracker created")
    except Exception as e:
        print(f"✗ Tracker creation failed: {e}")
        return
    
    # Test 3: Record operation
    print("\n3. Testing operation recording...")
    try:
        tracker.record_operation("test", 100.0, True)
        print("✓ Operation recorded")
    except Exception as e:
        print(f"✗ Operation recording failed: {e}")
        return
    
    # Test 4: Check data
    print("\n4. Testing data access...")
    try:
        print(f"  - Operation times: {tracker.operation_times}")
        print(f"  - Operation calls: {tracker.operation_calls}")
        print(f"  - Operation errors: {tracker.operation_errors}")
        print("✓ Data access successful")
    except Exception as e:
        print(f"✗ Data access failed: {e}")
        return
    
    # Test 5: Simple statistics
    print("\n5. Testing simple statistics...")
    try:
        times = tracker.operation_times["test"]
        avg = sum(times) / len(times)
        print(f"  - Times: {times}")
        print(f"  - Average: {avg}")
        print("✓ Statistics successful")
    except Exception as e:
        print(f"✗ Statistics failed: {e}")
        return
    
    print("\n=== All basic tests passed! ===")

if __name__ == "__main__":
    test_basic_operations() 