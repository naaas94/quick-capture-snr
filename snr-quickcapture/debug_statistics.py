#!/usr/bin/env python3
"""
Debug script to test statistics calculations
"""

import sys
import statistics
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_statistics():
    """Test the statistics calculations that might be causing the hang."""
    
    print("=== Testing Statistics Calculations ===")
    
    # Test data
    times = [100.0, 200.0]
    print(f"Test data: {times}")
    
    # Test 1: statistics.mean
    print("\n1. Testing statistics.mean...")
    try:
        avg = statistics.mean(times)
        print(f"✓ Mean: {avg}")
    except Exception as e:
        print(f"✗ Error in mean: {e}")
        return
    
    # Test 2: statistics.quantiles for p95
    print("\n2. Testing statistics.quantiles for p95...")
    try:
        if len(times) >= 20:
            p95 = statistics.quantiles(times, n=20)[18]
            print(f"✓ P95 (n=20): {p95}")
        else:
            p95 = max(times)
            print(f"✓ P95 (max): {p95}")
    except Exception as e:
        print(f"✗ Error in p95: {e}")
        return
    
    # Test 3: statistics.quantiles for p99
    print("\n3. Testing statistics.quantiles for p99...")
    try:
        if len(times) >= 100:
            p99 = statistics.quantiles(times, n=100)[98]
            print(f"✓ P99 (n=100): {p99}")
        else:
            p99 = max(times)
            print(f"✓ P99 (max): {p99}")
    except Exception as e:
        print(f"✗ Error in p99: {e}")
        return
    
    # Test 4: min/max
    print("\n4. Testing min/max...")
    try:
        min_val = min(times)
        max_val = max(times)
        print(f"✓ Min: {min_val}, Max: {max_val}")
    except Exception as e:
        print(f"✗ Error in min/max: {e}")
        return
    
    print("\n=== All statistics tests passed! ===")

if __name__ == "__main__":
    test_statistics() 