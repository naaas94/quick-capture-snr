#!/usr/bin/env python3
"""
Debug test script for QuickCapture storage.
"""

from scripts.storage_engine import StorageEngine
from scripts.parse_input import parse_note_input
from scripts.validate_note import validate_note
from scripts.models import create_note_from_parsed

def main():
    print("Testing QuickCapture storage...")
    
    # Initialize storage
    storage = StorageEngine(":memory:")
    print("✓ Storage initialized")
    
    # Parse input
    input_text = "python, coding: Test note for debugging"
    parsed = parse_note_input(input_text)
    print("✓ Input parsed")
    
    # Validate
    validation = validate_note(parsed)
    print("✓ Note validated")
    
    # Create note object
    note = create_note_from_parsed(parsed, validation)
    print("✓ Note object created")
    
    # Store note
    success = storage.store_note(note)
    print(f"✓ Storage result: {success}")
    
    # Get stats
    stats = storage.get_database_stats()
    print(f"✓ Database stats: {stats}")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    main() 