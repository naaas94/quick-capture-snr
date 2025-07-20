#!/usr/bin/env python3
"""
Debug script to test storage engine connection behavior
"""

from scripts.storage_engine import StorageEngine
from scripts.models import ParsedNote
from scripts.parse_input import ContentType
from datetime import datetime, timezone
import uuid

def test_memory_storage():
    print("=== Testing :memory: storage behavior ===")
    
    # Create storage engine
    storage = StorageEngine(":memory:")
    print(f"DB Path: {storage.db_path}")
    
    # Create a simple test note
    note = ParsedNote(
        note_id=str(uuid.uuid4()),
        tags=["test", "debug"],
        note="Test note for debugging storage",
        comment="Testing storage behavior",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=1,
        semantic_density=0.8,
        tag_quality_score=0.9,
        content_type=ContentType.GENERAL,
        confidence_score=0.85,
        embedding_vector=None,
        snr_metadata={},
        valid=True,
        issues=[],
        origin="debug",
        raw_text="test, debug: Test note for debugging storage : Testing storage behavior",
        tag_hierarchy={},
        co_occurrence_patterns={}
    )
    
    # Test first storage
    print("\n1. First storage attempt:")
    result1 = storage.store_note(note)
    print(f"Result: {result1}")
    
    # Test second storage (this should reveal the connection issue)
    print("\n2. Second storage attempt (same note):")
    result2 = storage.store_note(note)
    print(f"Result: {result2}")
    
    # Test retrieving notes
    print("\n3. Testing retrieval:")
    try:
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM notes")
            count = cursor.fetchone()[0]
            print(f"Notes in database: {count}")
    except Exception as e:
        print(f"Retrieval failed: {e}")

if __name__ == "__main__":
    test_memory_storage() 