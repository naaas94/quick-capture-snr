#!/usr/bin/env python3
"""
Enhanced Input Parsing Module

Parse user input string into structured components with semantic preprocessing
and intelligent tag extraction.
"""

import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


class ContentType(Enum):
    """Content type classification for notes."""
    TASK = "task"
    IDEA = "idea"
    MEETING = "meeting"
    REFERENCE = "reference"
    CODE = "code"
    GENERAL = "general"


@dataclass
class ParsedInput:
    """Structured output from input parsing."""
    tags: List[str]
    note: str
    comment: Optional[str]
    raw_text: str
    semantic_density: float
    content_type: ContentType
    confidence_score: float


class ParsingError(Exception):
    """Custom exception for parsing failures."""
    pass


def calculate_semantic_density(text: str) -> float:
    """
    Calculate semantic density based on stopword ratio and token diversity.
    
    Higher density = more meaningful content, lower stopword ratio.
    """
    # Common English stopwords
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
        'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
        'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
        'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been',
        'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did',
        'get', 'come', 'made', 'may', 'part'
    }
    
    # Normalize and tokenize
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    
    # Calculate stopword ratio
    stopword_count = sum(1 for word in words if word in stopwords)
    stopword_ratio = stopword_count / len(words)
    
    # Calculate token diversity (unique words ratio)
    unique_words = len(set(words))
    diversity_ratio = unique_words / len(words)
    
    # Calculate average word length (longer words often more meaningful)
    avg_word_length = sum(len(word) for word in words) / len(words)
    length_score = min(avg_word_length / 8.0, 1.0)  # Normalize to 0-1
    
    # Combine metrics for semantic density
    semantic_density = (
        (1.0 - stopword_ratio) * 0.4 +
        diversity_ratio * 0.3 +
        length_score * 0.3
    )
    
    return round(semantic_density, 3)


def classify_content_type(text: str, tags: List[str]) -> ContentType:
    """
    Classify content type based on text patterns and tags.
    """
    text_lower = text.lower()
    tags_lower = [tag.lower() for tag in tags]
    
    # Task indicators
    task_indicators = ['todo', 'task', 'do', 'need', 'must', 'should', 'implement', 'fix']
    if any(indicator in text_lower for indicator in task_indicators) or 'task' in tags_lower:
        return ContentType.TASK
    
    # Meeting indicators
    meeting_indicators = ['meeting', 'discuss', 'discussed', 'call', 'presentation', 'agenda']
    if any(indicator in text_lower for indicator in meeting_indicators) or 'meeting' in tags_lower:
        return ContentType.MEETING
    
    # Code indicators
    code_indicators = ['code', 'function', 'class', 'method', 'bug', 'error', 'debug', 'test']
    if any(indicator in text_lower for indicator in code_indicators) or 'code' in tags_lower:
        return ContentType.CODE
    
    # Idea indicators
    idea_indicators = ['idea', 'think', 'consider', 'maybe', 'could', 'might', 'suggestion']
    if any(indicator in text_lower for indicator in idea_indicators) or 'idea' in tags_lower:
        return ContentType.IDEA
    
    # Reference indicators
    reference_indicators = ['reference', 'link', 'url', 'article', 'paper', 'book', 'document']
    if any(indicator in text_lower for indicator in reference_indicators) or 'reference' in tags_lower:
        return ContentType.REFERENCE
    
    return ContentType.GENERAL


def calculate_confidence_score(parsed: Dict) -> float:
    """
    Calculate confidence score based on parsing quality indicators.
    """
    score = 1.0
    
    # Penalize for missing tags
    if not parsed['tags']:
        score -= 0.3
    
    # Penalize for very short notes
    if len(parsed['note']) < 10:
        score -= 0.2
    
    # Penalize for very low semantic density
    if parsed['semantic_density'] < 0.2:
        score -= 0.2
    
    # Bonus for good semantic density
    if parsed['semantic_density'] > 0.7:
        score += 0.1
    
    # Bonus for having a comment
    if parsed['comment']:
        score += 0.1
    
    return max(0.0, min(1.0, score))


def normalize_tags(tags: List[str]) -> List[str]:
    """
    Normalize tags: lowercase, remove duplicates, strip whitespace.
    """
    normalized = []
    seen = set()
    
    for tag in tags:
        # Clean and normalize
        clean_tag = tag.strip().lower()
        clean_tag = re.sub(r'[^\w\-_]', '', clean_tag)  # Remove special chars except - and _
        
        # Skip empty tags and duplicates
        if clean_tag and clean_tag not in seen:
            normalized.append(clean_tag)
            seen.add(clean_tag)
    
    return normalized


def parse_note_input(text: str) -> Dict:
    """
    Parse the user input string into structured components.
    
    Expected grammar: tag1, tag2: note body : optional comment
    
    Returns:
        Dict with parsed components and semantic analysis
    """
    if not text or not text.strip():
        raise ParsingError("Input text cannot be empty")
    
    raw_text = text.strip()
    
    # Split on first two colons only
    parts = raw_text.split(':', 2)
    
    if len(parts) < 2:
        raise ParsingError("Invalid format: must contain at least one colon separator")
    
    # Extract tags and note body
    tag_part = parts[0].strip()
    note_body = parts[1].strip()
    comment = parts[2].strip() if len(parts) > 2 else None
    
    # Parse tags
    if not tag_part:
        raise ParsingError("Tags section cannot be empty")
    
    tags = [tag.strip() for tag in tag_part.split(',')]
    tags = normalize_tags(tags)
    
    if not tags:
        raise ParsingError("At least one valid tag is required")
    
    if not note_body:
        raise ParsingError("Note body cannot be empty")
    
    # Calculate semantic density
    full_text = f"{note_body} {comment or ''}"
    semantic_density = calculate_semantic_density(full_text)
    
    # Classify content type
    content_type = classify_content_type(note_body, tags)
    
    # Create parsed structure
    parsed = {
        "tags": tags,
        "note": note_body,
        "comment": comment,
        "raw_text": raw_text,
        "semantic_density": semantic_density,
        "content_type": content_type,
        "confidence_score": 0.0  # Will be calculated next
    }
    
    # Calculate confidence score
    parsed["confidence_score"] = calculate_confidence_score(parsed)
    
    return parsed


def parse_note_input_with_validation(text: str) -> Tuple[Dict, List[str]]:
    """
    Parse input with additional validation and return issues.
    
    Returns:
        Tuple of (parsed_dict, list_of_issues)
    """
    issues = []
    
    try:
        parsed = parse_note_input(text)
        
        # Additional validation checks
        if parsed['semantic_density'] < 0.2:
            issues.append("Low semantic density - consider adding more meaningful content")
        
        if len(parsed['note']) < 20:
            issues.append("Note body is quite short - consider expanding")
        
        if len(parsed['tags']) > 10:
            issues.append("Many tags detected - consider consolidating")
        
        if parsed['confidence_score'] < 0.5:
            issues.append("Low confidence score - review input format and content")
        
        return parsed, issues
        
    except ParsingError as e:
        issues.append(f"Parsing error: {str(e)}")
        return {}, issues
    except Exception as e:
        issues.append(f"Unexpected error: {str(e)}")
        return {}, issues


if __name__ == "__main__":
    # Test the parser
    test_inputs = [
        "python, coding: Implemented new feature for data processing : This will help with the ML pipeline",
        "meeting, project: Discussed Q4 roadmap with team : Need to follow up on budget approval",
        "idea, ml: Consider using transformer models for text classification : Research BERT vs RoBERTa",
        "task, bug: Fix the authentication issue in login module",
        "reference, paper: Attention is all you need - Vaswani et al. 2017"
    ]
    
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        try:
            parsed, issues = parse_note_input_with_validation(test_input)
            if parsed:
                print(f"Tags: {parsed['tags']}")
                print(f"Note: {parsed['note']}")
                print(f"Comment: {parsed['comment']}")
                print(f"Semantic Density: {parsed['semantic_density']}")
                print(f"Content Type: {parsed['content_type'].value}")
                print(f"Confidence: {parsed['confidence_score']}")
            if issues:
                print(f"Issues: {issues}")
        except Exception as e:
            print(f"Error: {e}") 