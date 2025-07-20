#!/usr/bin/env python3
"""
Debug validation logic for short notes
"""

from scripts.parse_input import parse_note_input
from scripts.validate_note import validate_note

def test_short_note_validation():
    print("=== Testing Short Note Validation ===")
    
    # Test the exact input from failing test
    input_text = "test: Hi"
    parsed = parse_note_input(input_text)
    validation = validate_note(parsed)
    
    print(f"Input: {input_text}")
    print(f"Parsed note length: {len(parsed.get('note', ''))}")
    print(f"Valid: {validation['valid']}")
    print(f"Issues: {validation['issues']}")
    print(f"Tag quality: {validation['tag_quality_score']}")
    print(f"Overall confidence: {validation['overall_confidence']}")
    
    # Test another short input
    input_text2 = "general, misc: Short note : ok"
    parsed2 = parse_note_input(input_text2)
    validation2 = validate_note(parsed2)
    
    print(f"\nInput: {input_text2}")
    print(f"Parsed note length: {len(parsed2.get('note', ''))}")
    print(f"Valid: {validation2['valid']}")
    print(f"Issues: {validation2['issues']}")

if __name__ == "__main__":
    test_short_note_validation() 