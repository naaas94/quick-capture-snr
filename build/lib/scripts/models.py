#!/usr/bin/env python3
"""
Enhanced Metadata Models

Define the core data structures for QuickCapture with rich metadata
and semantic intelligence.
"""

import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import hashlib

from .parse_input import ContentType


@dataclass
class TagSuggestion:
    """Tag suggestion with confidence score."""
    tag: str
    confidence: float
    reason: str
    source: str  # 'semantic', 'co_occurrence', 'hierarchy'


@dataclass
class TagDriftReport:
    """Report on tag usage patterns and drift."""
    emerging_tags: List[str]
    dying_tags: List[str]
    tag_usage_frequency: Dict[str, int]
    drift_score: float
    recommendations: List[str]


@dataclass
class TagConsolidation:
    """Tag consolidation suggestion."""
    source_tags: List[str]
    suggested_tag: str
    confidence: float
    reason: str


@dataclass
class ParsedNote:
    """
    Enhanced ParsedNote with comprehensive metadata and semantic intelligence.
    
    This is the core data structure that flows through the QuickCapture system
    and provides downstream compatibility with semantic systems like SNR.
    """
    
    # Core fields
    note_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: List[str] = field(default_factory=list)
    note: str = ""
    comment: Optional[str] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    valid: bool = True
    issues: List[str] = field(default_factory=list)
    origin: str = "quickcapture"
    version: int = 1
    
    # Raw data
    raw_text: str = ""
    
    # Semantic analysis
    semantic_density: float = 0.0
    tag_quality_score: float = 0.0
    content_type: ContentType = ContentType.GENERAL
    confidence_score: float = 0.0
    
    # Enhanced semantic features
    tag_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    co_occurrence_patterns: Dict[str, float] = field(default_factory=dict)
    embedding_vector: Optional[List[float]] = None
    
    # SNR compatibility
    snr_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.note_id:
            self.note_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def add_snr_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata for SNR compatibility."""
        self.snr_metadata.update(metadata)
    
    def get_snr_optimized_body(self) -> str:
        """
        Get note body optimized for SNR processing.
        
        Strips problematic characters and normalizes for embedding generation.
        """
        # Remove problematic characters for embeddings
        problematic_chars = ['&', '<', '>', '"', "'"]
        optimized = self.note
        
        for char in problematic_chars:
            optimized = optimized.replace(char, ' ')
        
        # Normalize whitespace
        optimized = ' '.join(optimized.split())
        
        return optimized
    
    def calculate_semantic_coherence(self) -> float:
        """
        Calculate semantic coherence score based on content analysis.
        """
        # Base score from semantic density
        score = self.semantic_density
        
        # Boost for good tag quality
        score += self.tag_quality_score * 0.2
        
        # Boost for content type specificity
        if self.content_type != ContentType.GENERAL:
            score += 0.1
        
        # Penalty for issues
        if self.issues:
            score -= len(self.issues) * 0.05
        
        return max(0.0, min(1.0, score))
    
    def get_tag_hierarchy(self) -> Dict[str, List[str]]:
        """
        Generate tag hierarchy based on tag relationships.
        """
        hierarchy = {}
        
        for tag in self.tags:
            # Simple hierarchy: main tag -> subcategories
            if '.' in tag:
                main_tag, sub_tag = tag.split('.', 1)
                if main_tag not in hierarchy:
                    hierarchy[main_tag] = []
                hierarchy[main_tag].append(sub_tag)
            else:
                if tag not in hierarchy:
                    hierarchy[tag] = []
        
        return hierarchy
    
    def update_confidence_score(self) -> None:
        """
        Update confidence score based on current state.
        """
        # Base confidence from parsing
        base_confidence = self.confidence_score
        
        # Adjust based on validation
        if not self.valid:
            base_confidence *= 0.7
        
        # Adjust based on issues
        if self.issues:
            base_confidence *= (0.9 ** len(self.issues))
        
        # Boost for good semantic coherence
        coherence = self.calculate_semantic_coherence()
        if coherence > 0.7:
            base_confidence *= 1.1
        
        self.confidence_score = max(0.0, min(1.0, base_confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enum to string
        data['content_type'] = self.content_type.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParsedNote':
        """Create ParsedNote from dictionary."""
        # Convert content_type string back to enum
        if 'content_type' in data and isinstance(data['content_type'], str):
            data['content_type'] = ContentType(data['content_type'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ParsedNote':
        """Create ParsedNote from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_hash(self) -> str:
        """Get content hash for deduplication."""
        content = f"{self.note}{self.comment or ''}{''.join(sorted(self.tags))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_duplicate_of(self, other: 'ParsedNote') -> bool:
        """Check if this note is a duplicate of another."""
        return self.get_hash() == other.get_hash()
    
    def merge_with(self, other: 'ParsedNote') -> 'ParsedNote':
        """
        Merge with another note, combining metadata and incrementing version.
        """
        merged = ParsedNote(
            note_id=self.note_id,
            tags=list(set(self.tags + other.tags)),
            note=self.note,
            comment=self.comment or other.comment,
            timestamp=self.timestamp,
            valid=self.valid and other.valid,
            issues=list(set(self.issues + other.issues)),
            origin=self.origin,
            version=self.version + 1,
            raw_text=self.raw_text,
            semantic_density=max(self.semantic_density, other.semantic_density),
            tag_quality_score=max(self.tag_quality_score, other.tag_quality_score),
            content_type=self.content_type,
            confidence_score=max(self.confidence_score, other.confidence_score),
            snr_metadata={**self.snr_metadata, **other.snr_metadata}
        )
        
        return merged
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the note for quick inspection."""
        return {
            'note_id': self.note_id,
            'tags': self.tags,
            'note_preview': self.note[:100] + '...' if len(self.note) > 100 else self.note,
            'content_type': self.content_type.value,
            'confidence_score': self.confidence_score,
            'valid': self.valid,
            'issue_count': len(self.issues),
            'timestamp': self.timestamp
        }


@dataclass
class ProcessingMetrics:
    """Metrics for processing performance and quality."""
    processing_time_ms: float
    parsing_confidence: float
    validation_score: float
    storage_success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BatchProcessingResult:
    """Result of batch processing multiple notes."""
    total_notes: int
    successful_notes: int
    failed_notes: int
    processing_time_ms: float
    average_confidence: float
    notes: List[ParsedNote] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# Utility functions for working with ParsedNote objects

def create_note_from_parsed(parsed: Dict, validation: Dict) -> ParsedNote:
    """
    Create a ParsedNote from parsed input and validation results.
    """
    return ParsedNote(
        tags=parsed.get('tags', []),
        note=parsed.get('note', ''),
        comment=parsed.get('comment'),
        raw_text=parsed.get('raw_text', ''),
        semantic_density=parsed.get('semantic_density', 0.0),
        tag_quality_score=validation.get('tag_quality_score', 0.0),
        content_type=parsed.get('content_type', ContentType.GENERAL),
        confidence_score=validation.get('overall_confidence', 0.0),
        valid=validation.get('valid', True),
        issues=[issue['message'] for issue in validation.get('issues', [])],
        tag_hierarchy=validation.get('validation_details', {}).get('tag_hierarchy', {}),
        snr_metadata={
            'semantic_coherence_score': validation.get('semantic_coherence_score', 0.0),
            'validation_details': validation.get('validation_details', {}),
            'processing_timestamp': datetime.now(timezone.utc).isoformat()
        }
    )


def optimize_note_for_snr(note: ParsedNote) -> ParsedNote:
    """
    Optimize a note for downstream SNR processing.
    """
    # Create optimized version
    optimized = ParsedNote(
        note_id=note.note_id,
        tags=note.tags,
        note=note.get_snr_optimized_body(),
        comment=note.comment,
        timestamp=note.timestamp,
        valid=note.valid,
        issues=note.issues,
        origin=note.origin,
        version=note.version,
        raw_text=note.raw_text,
        semantic_density=note.semantic_density,
        tag_quality_score=note.tag_quality_score,
        content_type=note.content_type,
        confidence_score=note.confidence_score,
        tag_hierarchy=note.get_tag_hierarchy(),
        co_occurrence_patterns=note.co_occurrence_patterns,
        embedding_vector=note.embedding_vector,
        snr_metadata={
            **note.snr_metadata,
            'optimized_for_embeddings': True,
            'optimization_timestamp': datetime.now(timezone.utc).isoformat(),
            'semantic_coherence': note.calculate_semantic_coherence()
        }
    )
    
    return optimized


if __name__ == "__main__":
    # Test the models
    test_note = ParsedNote(
        tags=['python', 'coding'],
        note='Implemented new feature for data processing',
        comment='This will help with the ML pipeline',
        semantic_density=0.75,
        tag_quality_score=0.8,
        content_type=ContentType.CODE,
        confidence_score=0.85
    )
    
    print("Test Note:")
    print(test_note.to_json())
    
    print(f"\nSNR Optimized Body: {test_note.get_snr_optimized_body()}")
    print(f"Semantic Coherence: {test_note.calculate_semantic_coherence()}")
    print(f"Tag Hierarchy: {test_note.get_tag_hierarchy()}")
    print(f"Content Hash: {test_note.get_hash()}")
    
    # Test optimization
    optimized = optimize_note_for_snr(test_note)
    print(f"\nOptimized SNR Metadata: {optimized.snr_metadata}") 