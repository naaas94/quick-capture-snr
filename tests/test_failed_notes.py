#!/usr/bin/env python3
"""
Failed Notes Testing Suite

Test how QuickCapture handles various types of failed/invalid notes
throughout the parsing, validation, and storage pipeline.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from scripts.parse_input import parse_note_input, ParsingError, ContentType
from scripts.validate_note import validate_note
from scripts.models import ParsedNote, create_note_from_parsed
from scripts.storage_engine import StorageEngine


class TestFailedNotes:
    """Test handling of various types of failed/invalid notes."""
    
    def test_parsing_failures(self):
        """Test various parsing failure scenarios."""
        # Empty input
        with pytest.raises(ParsingError):
            parse_note_input("")
        
        # Missing colons
        with pytest.raises(ParsingError):
            parse_note_input("no colons here")
        
        # Only tags, no note body
        with pytest.raises(ParsingError):
            parse_note_input("tag1, tag2:")
        
        # Only colons, no content
        with pytest.raises(ParsingError):
            parse_note_input(":")
        
        # Empty tags
        with pytest.raises(ParsingError):
            parse_note_input(", : some note")
        
        # Whitespace only
        with pytest.raises(ParsingError):
            parse_note_input("   ")
    
    def test_validation_failures(self):
        """Test notes that parse but fail validation."""
        # Very short note
        input_text = "test: Hi"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        
        assert validation['valid'] is False
        assert len(validation['issues']) > 0
        assert validation['overall_confidence'] < 0.5
        
        # Generic tags
        input_text = "general, misc: Some random note : ok"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        
        assert len(validation['issues']) > 0
        assert validation['tag_quality_score'] < 1.0
        
        # Too many tags
        input_text = "tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, tag11: Note with too many tags"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        
        assert len(validation['issues']) > 0
        assert validation['tag_quality_score'] < 1.0
        
        # Very long note
        long_note = "very_long_tag: " + "This is a very long note. " * 50
        parsed = parse_note_input(long_note)
        validation = validate_note(parsed)
        
        assert len(validation['issues']) > 0
    
    def test_storage_of_failed_notes(self):
        """Test that failed notes are still stored properly."""
        storage = StorageEngine(":memory:")
        
        # Test storing a note with validation issues
        input_text = "general, misc: Short note : ok"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        # Should still be stored even with issues
        success = storage.store_note(note)
        assert success is True
        
        # Verify it's in the database
        stats = storage.get_database_stats()
        assert stats['total_notes'] == 1
        assert stats['notes_with_issues'] >= 1
        
        # Retrieve notes with issues
        notes_with_issues = storage.get_notes_with_issues()
        assert len(notes_with_issues) >= 1
        assert notes_with_issues[0].note_id == note.note_id
        assert not notes_with_issues[0].valid or len(notes_with_issues[0].issues) > 0
    
    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        storage = StorageEngine(":memory:")
        
        # Test minimum valid note
        input_text = "test: This is a minimum valid note with exactly ten characters"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        success = storage.store_note(note)
        assert success is True
        
        # Test maximum length note
        max_note = "test: " + "x" * 1000
        parsed = parse_note_input(max_note)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        success = storage.store_note(note)
        assert success is True
        
        # Test single character tag
        input_text = "a: Test with single character tag"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        success = storage.store_note(note)
        assert success is True
        
        # Test very long tag
        input_text = "very_long_tag_name_that_exceeds_normal_length: Test with long tag"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        success = storage.store_note(note)
        assert success is True
    
    def test_special_characters(self):
        """Test notes with special characters and edge cases."""
        storage = StorageEngine(":memory:")
        
        # Test problematic characters in note body
        input_text = "test: Note with & < > \" ' characters"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        success = storage.store_note(note)
        assert success is True
        
        # Test excessive whitespace
        input_text = "test: Note   with   excessive   whitespace"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        success = storage.store_note(note)
        assert success is True
        
        # Test unicode characters
        input_text = "test: Note with unicode: café, naïve, résumé"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        success = storage.store_note(note)
        assert success is True
    
    def test_duplicate_notes(self):
        """Test handling of duplicate notes."""
        storage = StorageEngine(":memory:")
        
        # Store the same note twice
        input_text = "python, coding: Test duplicate note"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note1 = create_note_from_parsed(parsed, validation)
        note2 = create_note_from_parsed(parsed, validation)
        
        # Both should be stored (they have different IDs)
        success1 = storage.store_note(note1)
        success2 = storage.store_note(note2)
        
        assert success1 is True
        assert success2 is True
        
        # Should have 2 notes in database
        stats = storage.get_database_stats()
        assert stats['total_notes'] == 2
    
    def test_mixed_quality_notes(self):
        """Test storing a mix of high and low quality notes."""
        storage = StorageEngine(":memory:")
        
        # High quality note
        good_input = "python, coding: Implemented comprehensive data processing pipeline with error handling and logging : This will significantly improve our ML workflow"
        parsed = parse_note_input(good_input)
        validation = validate_note(parsed)
        good_note = create_note_from_parsed(parsed, validation)
        
        # Low quality note
        bad_input = "general, misc: Short note : ok"
        parsed = parse_note_input(bad_input)
        validation = validate_note(parsed)
        bad_note = create_note_from_parsed(parsed, validation)
        
        # Store both
        success1 = storage.store_note(good_note)
        success2 = storage.store_note(bad_note)
        
        assert success1 is True
        assert success2 is True
        
        # Check statistics
        stats = storage.get_database_stats()
        assert stats['total_notes'] == 2
        assert stats['notes_with_issues'] >= 1
        
        # Check confidence ranges
        good_notes = storage.get_notes_by_confidence_range(0.7, 1.0)
        bad_notes = storage.get_notes_by_confidence_range(0.0, 0.5)
        
        assert len(good_notes) >= 1
        assert len(bad_notes) >= 1
    
    def test_error_recovery(self):
        """Test system recovery after encountering errors."""
        storage = StorageEngine(":memory:")
        
        # Try to store a valid note after encountering errors
        valid_input = "python, coding: Valid note after errors"
        parsed = parse_note_input(valid_input)
        validation = validate_note(parsed)
        valid_note = create_note_from_parsed(parsed, validation)
        
        success = storage.store_note(valid_note)
        assert success is True
        
        # System should still work normally
        stats = storage.get_database_stats()
        assert stats['total_notes'] == 1
        
        # Should be able to retrieve the note
        notes = storage.retrieve_notes_by_tag('python')
        assert len(notes) == 1
        assert notes[0].note_id == valid_note.note_id


if __name__ == "__main__":
    # Run failed notes tests
    print("Running failed notes tests...")
    
    test_failed = TestFailedNotes()
    
    # Test parsing failures
    test_failed.test_parsing_failures()
    print("✓ Parsing failure tests passed")
    
    # Test validation failures
    test_failed.test_validation_failures()
    print("✓ Validation failure tests passed")
    
    # Test storage of failed notes
    test_failed.test_storage_of_failed_notes()
    print("✓ Failed notes storage tests passed")
    
    # Test boundary conditions
    test_failed.test_boundary_conditions()
    print("✓ Boundary condition tests passed")
    
    # Test special characters
    test_failed.test_special_characters()
    print("✓ Special character tests passed")
    
    # Test duplicate notes
    test_failed.test_duplicate_notes()
    print("✓ Duplicate notes tests passed")
    
    # Test mixed quality notes
    test_failed.test_mixed_quality_notes()
    print("✓ Mixed quality notes tests passed")
    
    # Test error recovery
    test_failed.test_error_recovery()
    print("✓ Error recovery tests passed")
    
    print("\n🎉 All failed notes tests passed!") 