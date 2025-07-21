#!/usr/bin/env python3
"""
Basic Functionality Tests

Test the core functionality of the QuickCapture system.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from scripts.parse_input import parse_note_input, ContentType
from scripts.validate_note import validate_note
from scripts.models import ParsedNote, create_note_from_parsed
from scripts.storage_engine import StorageEngine


class TestParsing:
    """Test input parsing functionality."""
    
    def test_basic_parsing(self):
        """Test basic note parsing."""
        input_text = "python, coding: Implemented new feature for data processing : This will help with the ML pipeline"
        
        parsed = parse_note_input(input_text)
        
        assert parsed['tags'] == ['python', 'coding']
        assert parsed['note'] == 'Implemented new feature for data processing'
        assert parsed['comment'] == 'This will help with the ML pipeline'
        assert parsed['semantic_density'] > 0.0
        assert parsed['confidence_score'] > 0.0
        assert parsed['content_type'] == ContentType.TASK  # "implemented" triggers task classification
    
    def test_parsing_without_comment(self):
        """Test parsing without optional comment."""
        input_text = "meeting, project: Discussed Q4 roadmap with team"
        
        parsed = parse_note_input(input_text)
        
        assert parsed['tags'] == ['meeting', 'project']
        assert parsed['note'] == 'Discussed Q4 roadmap with team'
        assert parsed['comment'] is None
        assert parsed['content_type'] == ContentType.MEETING
    
    def test_tag_normalization(self):
        """Test tag normalization."""
        input_text = "PYTHON, Coding, python: Test note"
        
        parsed = parse_note_input(input_text)
        
        assert parsed['tags'] == ['python', 'coding']  # Normalized and deduplicated
        assert parsed['note'] == 'Test note'
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        with pytest.raises(Exception):
            parse_note_input("")  # Empty input
        
        with pytest.raises(Exception):
            parse_note_input("no colons here")  # Missing colons


class TestValidation:
    """Test validation functionality."""
    
    def test_basic_validation(self):
        """Test basic note validation."""
        input_text = "python, coding: Implemented new feature for data processing : This will help with the ML pipeline"
        parsed = parse_note_input(input_text)
        
        validation = validate_note(parsed)
        
        assert validation['valid'] is True
        assert validation['tag_quality_score'] > 0.0
        assert validation['semantic_coherence_score'] > 0.0
        assert validation['overall_confidence'] > 0.0
    
    def test_validation_with_issues(self):
        """Test validation with quality issues."""
        input_text = "general, misc: Short note : ok"
        parsed = parse_note_input(input_text)
        
        validation = validate_note(parsed)
        
        # Should have some issues due to generic tags and short content
        assert len(validation['issues']) > 0
        assert validation['tag_quality_score'] < 1.0
    
    def test_validation_edge_cases(self):
        """Test validation edge cases."""
        # Test with very short note
        input_text = "test: Hi"
        parsed = parse_note_input(input_text)
        
        validation = validate_note(parsed)
        
        assert validation['valid'] is False  # Should be invalid due to short content


class TestModels:
    """Test data models."""
    
    def test_parsed_note_creation(self):
        """Test ParsedNote creation."""
        input_text = "python, coding: Implemented new feature for data processing : This will help with the ML pipeline"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        
        note = create_note_from_parsed(parsed, validation)
        
        assert note.note_id is not None
        assert note.tags == ['python', 'coding']
        assert note.note == 'Implemented new feature for data processing'
        assert note.comment == 'This will help with the ML pipeline'
        assert note.valid is True
        assert note.origin == 'quickcapture'
        assert note.version == 1
    
    def test_parsed_note_serialization(self):
        """Test ParsedNote serialization."""
        note = ParsedNote(
            tags=['python', 'coding'],
            note='Test note',
            semantic_density=0.75,
            tag_quality_score=0.8,
            content_type=ContentType.CODE,
            confidence_score=0.85
        )
        
        # Test to_dict
        note_dict = note.to_dict()
        assert note_dict['tags'] == ['python', 'coding']
        assert note_dict['content_type'] == 'code'
        
        # Test to_json
        json_str = note.to_json()
        assert 'python' in json_str
        assert 'coding' in json_str
        
        # Test from_dict
        note_from_dict = ParsedNote.from_dict(note_dict)
        assert note_from_dict.tags == note.tags
        assert note_from_dict.content_type == note.content_type
    
    def test_parsed_note_methods(self):
        """Test ParsedNote utility methods."""
        note = ParsedNote(
            tags=['python', 'coding'],
            note='Test note with & special <chars>',
            semantic_density=0.75,
            tag_quality_score=0.8,
            content_type=ContentType.CODE,
            confidence_score=0.85
        )
        
        # Test SNR optimization
        optimized_body = note.get_snr_optimized_body()
        assert '&' not in optimized_body
        assert '<' not in optimized_body
        assert '>' not in optimized_body
        
        # Test semantic coherence
        coherence = note.calculate_semantic_coherence()
        assert 0.0 <= coherence <= 1.0
        
        # Test tag hierarchy
        hierarchy = note.get_tag_hierarchy()
        assert 'python' in hierarchy
        assert 'coding' in hierarchy
        
        # Test content hash
        hash1 = note.get_hash()
        hash2 = note.get_hash()
        assert hash1 == hash2  # Should be consistent


class TestStorage:
    """Test storage functionality."""
    
    def test_storage_engine_initialization(self):
        """Test storage engine initialization."""
        # Use temporary database for testing
        storage = StorageEngine(":memory:")
        
        # Test database stats
        stats = storage.get_database_stats()
        assert stats['total_notes'] == 0
        assert stats['total_tags'] == 0
    
    def test_note_storage(self):
        """Test storing and retrieving notes."""
        storage = StorageEngine(":memory:")
        
        # Create a test note
        input_text = "python, coding: Implemented new feature for data processing : This will help with the ML pipeline"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        # Store the note
        success = storage.store_note(note)
        assert success is True
        
        # Check database stats
        stats = storage.get_database_stats()
        assert stats['total_notes'] == 1
        assert stats['total_tags'] == 2  # python, coding
        
        # Retrieve notes by tag
        notes = storage.retrieve_notes_by_tag('python')
        assert len(notes) == 1
        assert notes[0].note_id == note.note_id
        
        # Check tag statistics
        tag_stats = storage.get_tag_statistics()
        assert 'python' in tag_stats
        assert 'coding' in tag_stats
        assert tag_stats['python']['usage_count'] == 1
    
    def test_notes_with_issues(self):
        """Test retrieving notes with validation issues."""
        storage = StorageEngine(":memory:")
        
        # Create a note with issues
        input_text = "general, misc: Short note : ok"
        parsed = parse_note_input(input_text)
        validation = validate_note(parsed)
        note = create_note_from_parsed(parsed, validation)
        
        # Store the note
        storage.store_note(note)
        
        # Get notes with issues
        notes_with_issues = storage.get_notes_with_issues()
        assert len(notes_with_issues) >= 1  # Should include our note with issues


class TestIntegration:
    """Test integration between components."""
    
    def test_full_pipeline(self):
        """Test the full QuickCapture pipeline."""
        storage = StorageEngine(":memory:")
        
        # Test inputs
        test_inputs = [
            "python, coding: Implemented new feature for data processing : This will help with the ML pipeline",
            "meeting, project: Discussed Q4 roadmap with team : Need to follow up on budget approval",
            "idea, ml: Consider using transformer models for text classification : Research BERT vs RoBERTa",
            "task, bug: Fix the authentication issue in login module",
            "reference, paper: Attention is all you need - Vaswani et al. 2017"
        ]
        
        stored_notes = []
        
        for input_text in test_inputs:
            # Parse
            parsed = parse_note_input(input_text)
            
            # Validate
            validation = validate_note(parsed)
            
            # Create note object
            note = create_note_from_parsed(parsed, validation)
            
            # Store
            success = storage.store_note(note)
            assert success is True
            
            stored_notes.append(note)
        
        # Verify storage
        stats = storage.get_database_stats()
        assert stats['total_notes'] == 5
        
        # Test retrieval by content type
        code_notes = storage.retrieve_notes_by_tag('coding')
        assert len(code_notes) >= 1
        
        meeting_notes = storage.retrieve_notes_by_tag('meeting')
        assert len(meeting_notes) >= 1
        
        # Test search functionality
        search_results = storage.search_semantic('data processing')
        assert len(search_results) >= 1


if __name__ == "__main__":
    # Run basic tests
    print("Running basic functionality tests...")
    
    # Test parsing
    test_parsing = TestParsing()
    test_parsing.test_basic_parsing()
    test_parsing.test_parsing_without_comment()
    test_parsing.test_tag_normalization()
    print("✓ Parsing tests passed")
    
    # Test validation
    test_validation = TestValidation()
    test_validation.test_basic_validation()
    test_validation.test_validation_with_issues()
    print("✓ Validation tests passed")
    
    # Test models
    test_models = TestModels()
    test_models.test_parsed_note_creation()
    test_models.test_parsed_note_serialization()
    test_models.test_parsed_note_methods()
    print("✓ Model tests passed")
    
    # Test storage
    test_storage = TestStorage()
    test_storage.test_storage_engine_initialization()
    test_storage.test_note_storage()
    print("✓ Storage tests passed")
    
    # Test integration
    test_integration = TestIntegration()
    test_integration.test_full_pipeline()
    print("✓ Integration tests passed")
    
    print("\n🎉 All basic functionality tests passed!") 