#!/usr/bin/env python3
"""
Simple Environment Test

This script tests basic functionality without causing hanging issues.
Run this to verify your environment is working correctly.
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic imports that should always work."""
    print("Testing basic imports...")
    
    try:
        import json
        print("✓ json imported successfully")
    except ImportError as e:
        print(f"✗ json import failed: {e}")
        return False
    
    try:
        import sqlite3
        print("✓ sqlite3 imported successfully")
    except ImportError as e:
        print(f"✗ sqlite3 import failed: {e}")
        return False
    
    try:
        import uuid
        print("✓ uuid imported successfully")
    except ImportError as e:
        print(f"✗ uuid import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test that project files exist."""
    print("\nTesting project structure...")
    
    required_files = [
        "requirements.txt",
        "pyproject.toml",
        "scripts/quick_add.py",
        "scripts/models.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_database_creation():
    """Test basic database operations."""
    print("\nTesting database operations...")
    
    try:
        import sqlite3
        import tempfile
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        # Test connection
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("INSERT INTO test (name) VALUES (?)", ("test",))
        cursor.execute("SELECT * FROM test")
        result = cursor.fetchone()
        
        conn.close()
        os.unlink(db_path)
        
        if result and result[1] == "test":
            print("✓ Database operations successful")
            return True
        else:
            print("✗ Database test failed")
            return False
            
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_optional_imports():
    """Test optional imports that might not be available."""
    print("\nTesting optional imports...")
    
    optional_modules = [
        ("rich", "Rich console output"),
        ("pyyaml", "YAML configuration"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning")
    ]
    
    available_count = 0
    for module_name, description in optional_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name} ({description}) available")
            available_count += 1
        except ImportError:
            print(f"⚠ {module_name} ({description}) not available")
    
    print(f"\nAvailable optional modules: {available_count}/{len(optional_modules)}")
    return available_count > 0

def main():
    """Run all tests."""
    print("🔍 QuickCapture Environment Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Project Structure", test_project_structure),
        ("Database Operations", test_database_creation),
        ("Optional Imports", test_optional_imports)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Environment is ready.")
        return 0
    elif passed >= total - 1:
        print("⚠️  Most tests passed. Environment is mostly ready.")
        return 0
    else:
        print("❌ Multiple tests failed. Check your setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 