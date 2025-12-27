#!/usr/bin/env python3
"""
Semantic Validation Engine

Validate parsed input against both structural and semantic constraints with
intelligent quality scoring.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .parse_input import ContentType


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Structured validation issue."""
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.valid: bool = True
        self.issues: List[ValidationIssue] = []
        self.semantic_coherence_score: float = 0.0
        self.tag_quality_score: float = 0.0
        self.overall_confidence: float = 0.0
        self.validation_details: Dict[str, Any] = {}
    
    def add_issue(self, level: ValidationLevel, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add a validation issue."""
        issue = ValidationIssue(level=level, message=message, field=field, suggestion=suggestion)
        self.issues.append(issue)
        
        # Mark as invalid for critical errors
        if level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
            self.valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "valid": self.valid,
            "issues": [{"level": i.level.value, "message": i.message, "field": i.field, "suggestion": i.suggestion} 
                      for i in self.issues],
            "semantic_coherence_score": self.semantic_coherence_score,
            "tag_quality_score": self.tag_quality_score,
            "overall_confidence": self.overall_confidence,
            "validation_details": self.validation_details
        }


def validate_tags(tags: List[str], result: ValidationResult) -> float:
    """
    Validate tag quality and return quality score.
    """
    if not tags:
        result.add_issue(ValidationLevel.ERROR, "At least one tag is required", "tags")
        return 0.0
    
    score = 1.0
    issues = []
    
    # Check tag length
    for i, tag in enumerate(tags):
        if len(tag) < 2:
            issues.append(f"Tag '{tag}' is too short (minimum 2 characters)")
            score -= 0.1
        elif len(tag) > 20:
            issues.append(f"Tag '{tag}' is too long (maximum 20 characters)")
            score -= 0.1
    
    # Check for duplicate tags (case-insensitive)
    tag_lower = [tag.lower() for tag in tags]
    if len(tag_lower) != len(set(tag_lower)):
        result.add_issue(ValidationLevel.WARNING, "Duplicate tags detected", "tags", 
                        "Remove duplicate tags")
        score -= 0.2
    
    # Check tag format (alphanumeric, hyphens, underscores only)
    for tag in tags:
        if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
            result.add_issue(ValidationLevel.WARNING, f"Tag '{tag}' contains invalid characters", "tags",
                           "Use only letters, numbers, hyphens, and underscores")
            score -= 0.1
    
    # Check for too many tags
    if len(tags) > 10:
        result.add_issue(ValidationLevel.WARNING, f"Too many tags ({len(tags)})", "tags",
                        "Consider consolidating to 10 or fewer tags")
        score -= 0.3  # Increased penalty
    
    # Check for very generic tags
    generic_tags = {'general', 'misc', 'other', 'todo', 'note', 'info'}
    generic_count = sum(1 for tag in tags if tag.lower() in generic_tags)
    if generic_count > 0:
        result.add_issue(ValidationLevel.INFO, f"{generic_count} generic tag(s) detected", "tags",
                        "Consider using more specific tags")
        score -= 0.2 * generic_count  # Increased penalty
    
    # Add individual tag issues
    for issue in issues:
        result.add_issue(ValidationLevel.WARNING, issue, "tags")
    
    return max(0.0, score)


def validate_note_body(note: str, result: ValidationResult) -> float:
    """
    Validate note body quality and return coherence score.
    """
    if not note or not note.strip():
        result.add_issue(ValidationLevel.ERROR, "Note body cannot be empty", "note")
        return 0.0
    
    score = 1.0
    
    # Check minimum length
    if len(note) < 10:
        result.add_issue(ValidationLevel.ERROR, "Note body is very short", "note",
                        "Consider adding more detail")
        score -= 0.5  # Increased penalty
    elif len(note) < 20:
        result.add_issue(ValidationLevel.INFO, "Note body is quite short", "note")
        score -= 0.2  # Increased penalty
    
    # Check maximum length
    if len(note) > 1000:
        result.add_issue(ValidationLevel.WARNING, "Note body is very long", "note",
                        "Consider breaking into multiple notes")
        score -= 0.2
    
    # Check for excessive whitespace
    if re.search(r'\s{3,}', note):
        result.add_issue(ValidationLevel.INFO, "Excessive whitespace detected", "note",
                        "Clean up extra spaces")
        score -= 0.1
    
    # Check for proper sentence structure
    sentences = re.split(r'[.!?]+', note)
    if len(sentences) > 1:
        # Multiple sentences - good
        score += 0.1
    else:
        # Single sentence - check if it's complete
        if not note.strip().endswith(('.', '!', '?')):
            result.add_issue(ValidationLevel.INFO, "Note appears to be incomplete", "note",
                           "Consider adding proper punctuation")
            score -= 0.1
    
    # Check for common issues
    if note.lower().startswith('the '):
        result.add_issue(ValidationLevel.INFO, "Note starts with 'the'", "note",
                        "Consider more direct phrasing")
        score -= 0.05
    
    return max(0.0, min(1.0, score))


def validate_semantic_coherence(parsed: Dict, result: ValidationResult) -> float:
    """
    Validate semantic coherence and return score.
    """
    semantic_density = parsed.get('semantic_density', 0.0)
    content_type = parsed.get('content_type')
    note = parsed.get('note', '')
    
    score = semantic_density  # Start with semantic density
    
    # Content type specific validation
    if content_type == ContentType.TASK:
        # Tasks should have action verbs
        action_verbs = ['implement', 'fix', 'create', 'update', 'review', 'test', 'deploy', 'configure']
        if not any(verb in note.lower() for verb in action_verbs):
            result.add_issue(ValidationLevel.WARNING, "Task note lacks clear action verb", "note",
                           "Use action verbs like 'implement', 'fix', 'create'")
            score -= 0.2
    
    elif content_type == ContentType.MEETING:
        # Meeting notes should mention participants or outcomes
        meeting_indicators = ['team', 'discussed', 'agreed', 'decided', 'participants', 'attendees']
        if not any(indicator in note.lower() for indicator in meeting_indicators):
            result.add_issue(ValidationLevel.INFO, "Meeting note could mention participants or outcomes", "note")
            score -= 0.1
    
    elif content_type == ContentType.CODE:
        # Code notes should mention specific technical details
        code_indicators = ['function', 'class', 'method', 'bug', 'error', 'test', 'api', 'database']
        if not any(indicator in note.lower() for indicator in code_indicators):
            result.add_issue(ValidationLevel.WARNING, "Code note lacks technical specificity", "note",
                           "Mention specific functions, classes, or technical details")
            score -= 0.2
    
    # Check for semantic density thresholds
    if semantic_density < 0.2:
        result.add_issue(ValidationLevel.WARNING, "Very low semantic density", "semantic_density",
                        "Consider adding more meaningful content")
        score -= 0.3
    elif semantic_density < 0.4:
        result.add_issue(ValidationLevel.INFO, "Low semantic density", "semantic_density")
        score -= 0.1
    
    return max(0.0, min(1.0, score))


def validate_comment(comment: Optional[str], result: ValidationResult) -> float:
    """
    Validate comment quality if present.
    """
    if not comment:
        return 1.0  # No comment is fine
    
    score = 1.0
    
    # Check comment length
    if len(comment) < 5:
        result.add_issue(ValidationLevel.INFO, "Comment is very short", "comment")
        score -= 0.2  # Increased penalty
    elif len(comment) > 500:
        result.add_issue(ValidationLevel.WARNING, "Comment is very long", "comment",
                        "Consider moving detailed information to the main note")
        score -= 0.2
    
    # Check for meaningful content
    if comment.lower() in ['ok', 'good', 'nice', 'cool', 'yes', 'no']:
        result.add_issue(ValidationLevel.INFO, "Comment is very generic", "comment",
                        "Consider adding more specific information")
        score -= 0.3  # Increased penalty
    
    return max(0.0, score)


def calculate_overall_confidence(parsed: Dict, validation_scores: Dict[str, float]) -> float:
    """
    Calculate overall confidence based on parsing and validation scores.
    """
    # Weights for different components
    weights = {
        'parsing_confidence': 0.3,
        'tag_quality': 0.2,
        'note_quality': 0.2,
        'semantic_coherence': 0.2,
        'comment_quality': 0.1
    }
    
    # Get individual scores
    parsing_confidence = parsed.get('confidence_score', 0.0)
    tag_quality = validation_scores.get('tag_quality', 0.0)
    note_quality = validation_scores.get('note_quality', 0.0)
    semantic_coherence = validation_scores.get('semantic_coherence', 0.0)
    comment_quality = validation_scores.get('comment_quality', 0.0)
    
    # Calculate weighted average
    overall_confidence = (
        parsing_confidence * weights['parsing_confidence'] +
        tag_quality * weights['tag_quality'] +
        note_quality * weights['note_quality'] +
        semantic_coherence * weights['semantic_coherence'] +
        comment_quality * weights['comment_quality']
    )
    
    return round(overall_confidence, 3)


def validate_note(parsed: Dict) -> Dict:
    """
    Validate the parsed input against structural and semantic constraints.
    
    Args:
        parsed: Dictionary from parse_note_input()
    
    Returns:
        Dictionary with validation results and quality scores
    """
    result = ValidationResult()
    
    # Extract components
    tags = parsed.get('tags', [])
    note = parsed.get('note', '')
    comment = parsed.get('comment')
    semantic_density = parsed.get('semantic_density', 0.0)
    
    # Perform validations
    tag_quality = validate_tags(tags, result)
    note_quality = validate_note_body(note, result)
    semantic_coherence = validate_semantic_coherence(parsed, result)
    comment_quality = validate_comment(comment, result)
    
    # Store validation scores
    validation_scores = {
        'tag_quality': tag_quality,
        'note_quality': note_quality,
        'semantic_coherence': semantic_coherence,
        'comment_quality': comment_quality
    }
    
    # Calculate overall confidence
    overall_confidence = calculate_overall_confidence(parsed, validation_scores)
    
    # Update result
    result.tag_quality_score = tag_quality
    result.semantic_coherence_score = semantic_coherence
    result.overall_confidence = overall_confidence
    result.validation_details = {
        'validation_scores': validation_scores,
        'semantic_density': semantic_density,
        'content_type': parsed.get('content_type', ContentType.GENERAL).value
    }
    
    return result.to_dict()


def validate_note_with_suggestions(parsed: Dict) -> Dict:
    """
    Enhanced validation with intelligent suggestions for improvement.
    """
    validation = validate_note(parsed)
    
    # Add intelligent suggestions based on validation results
    suggestions = []
    
    if validation['tag_quality_score'] < 0.7:
        suggestions.append("Consider improving tag quality by using more specific, descriptive tags")
    
    if validation['semantic_coherence_score'] < 0.6:
        suggestions.append("Improve semantic coherence by adding more meaningful content and context")
    
    if validation['overall_confidence'] < 0.5:
        suggestions.append("Overall quality is low - consider rewriting with more detail and structure")
    
    # Add suggestions to validation details
    validation['validation_details']['suggestions'] = suggestions
    
    return validation


if __name__ == "__main__":
    # Test the validator
    from parse_input import parse_note_input
    
    test_inputs = [
        "python, coding: Implemented new feature for data processing : This will help with the ML pipeline",
        "task, bug: Fix the authentication issue in login module",
        "meeting, project: Discussed Q4 roadmap with team : Need to follow up on budget approval",
        "idea, ml: Consider using transformer models for text classification : Research BERT vs RoBERTa",
        "general, misc: Some random note : ok",
        "misc: Ok",  # Very short and generic note
        "general, misc:    This   note   has   too   much   whitespace   and   invalid   characters   like   @#$%^&*",  # Excessive whitespace and invalid characters
        "tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, tag11: Short"  # Too many tags and short content
    ]
    
    for test_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"Testing: {test_input}")
        print('='*60)
        
        try:
            parsed = parse_note_input(test_input)
            validation = validate_note_with_suggestions(parsed)
            
            print(f"Valid: {validation['valid']}")
            print(f"Tag Quality: {validation['tag_quality_score']}")
            print(f"Semantic Coherence: {validation['semantic_coherence_score']}")
            print(f"Overall Confidence: {validation['overall_confidence']}")
            
            if validation['issues']:
                print("\nIssues:")
                for issue in validation['issues']:
                    print(f"  [{issue['level'].upper()}] {issue['message']}")
                    if issue['suggestion']:
                        print(f"    Suggestion: {issue['suggestion']}")
            
            if validation['validation_details'].get('suggestions'):
                print("\nSuggestions:")
                for suggestion in validation['validation_details']['suggestions']:
                    print(f"  - {suggestion}")
                    
        except Exception as e:
            print(f"Error: {e}") 