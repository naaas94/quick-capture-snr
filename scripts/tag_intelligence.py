#!/usr/bin/env python3
"""
Tag Intelligence System

Intelligent tag management with suggestion, drift detection, and quality scoring.
"""

import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .models import TagSuggestion, TagDriftReport, TagConsolidation, ParsedNote
from .storage_engine import StorageEngine

logger = logging.getLogger(__name__)


class TagIntelligence:
    """
    Intelligent tag management system with semantic analysis and pattern recognition.
    """
    
    def __init__(self, storage_engine: StorageEngine):
        """
        Initialize tag intelligence system.
        
        Args:
            storage_engine: Storage engine for accessing notes and tag data
        """
        self.storage = storage_engine
        self.tag_cache = {}  # Cache for tag analysis results
        self.similarity_cache = {}  # Cache for tag similarity calculations
    
    def suggest_tags(self, note_body: str, existing_tags: Optional[List[str]] = None) -> List[TagSuggestion]:
        """
        Suggest tags based on note content and existing tag patterns.
        
        Args:
            note_body: The note content to analyze
            existing_tags: Already assigned tags (optional)
            
        Returns:
            List of tag suggestions with confidence scores
        """
        suggestions = []
        
        # Get all existing tags and their usage patterns
        tag_stats = self.storage.get_tag_statistics()
        
        # Extract potential tags from note body
        potential_tags = self._extract_potential_tags(note_body)
        
        # Score potential tags based on various factors
        for tag in potential_tags:
            if existing_tags and tag in existing_tags:
                continue  # Skip already assigned tags
            
            confidence = self._calculate_tag_confidence(tag, note_body, tag_stats)
            
            if confidence > 0.3:  # Minimum confidence threshold
                suggestion = TagSuggestion(
                    tag=tag,
                    confidence=confidence,
                    reason=self._get_tag_suggestion_reason(tag, note_body, tag_stats),
                    source='semantic'
                )
                suggestions.append(suggestion)
        
        # Add co-occurrence based suggestions
        co_occurrence_suggestions = self._suggest_by_co_occurrence(existing_tags or [], tag_stats)
        suggestions.extend(co_occurrence_suggestions)
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:10]  # Return top 10 suggestions
    
    def detect_tag_drift(self, time_window_days: int = 30) -> TagDriftReport:
        """
        Detect tag usage patterns and drift over time.
        
        Args:
            time_window_days: Number of days to analyze
            
        Returns:
            TagDriftReport with drift analysis
        """
        # Get recent notes
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        # Get all notes and filter by date
        all_notes = self.storage.retrieve_notes_by_tag("")  # Get all notes
        recent_notes = [
            note for note in all_notes 
            if datetime.fromisoformat(note.timestamp.replace('Z', '+00:00')) > cutoff_date
        ]
        
        # Calculate tag usage frequency
        tag_usage = Counter()
        for note in recent_notes:
            for tag in note.tags:
                tag_usage[tag] += 1
        
        # Get historical tag usage for comparison
        historical_usage = self.storage.get_tag_statistics()
        
        # Identify emerging and dying tags
        emerging_tags = []
        dying_tags = []
        
        for tag, recent_count in tag_usage.items():
            historical_count = historical_usage.get(tag, {}).get('usage_count', 0)
            
            # Calculate growth rate
            if historical_count > 0:
                growth_rate = (recent_count - historical_count) / historical_count
                
                if growth_rate > 0.5:  # 50% growth threshold
                    emerging_tags.append(tag)
                elif growth_rate < -0.3:  # 30% decline threshold
                    dying_tags.append(tag)
        
        # Calculate drift score
        total_tags = len(tag_usage)
        drift_score = len(emerging_tags) / max(total_tags, 1) + len(dying_tags) / max(total_tags, 1)
        
        # Generate recommendations
        recommendations = []
        if emerging_tags:
            recommendations.append(f"Monitor emerging tags: {', '.join(emerging_tags[:5])}")
        if dying_tags:
            recommendations.append(f"Consider consolidating dying tags: {', '.join(dying_tags[:5])}")
        if drift_score > 0.3:
            recommendations.append("High tag drift detected - consider tag standardization")
        
        return TagDriftReport(
            emerging_tags=emerging_tags,
            dying_tags=dying_tags,
            tag_usage_frequency=dict(tag_usage),
            drift_score=drift_score,
            recommendations=recommendations
        )
    
    def calculate_tag_quality(self, tag: str) -> float:
        """
        Calculate quality score for a tag based on various metrics.
        
        Args:
            tag: Tag to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        tag_stats = self.storage.get_tag_statistics()
        tag_data = tag_stats.get(tag, {})
        
        if not tag_data:
            return 0.0
        
        score = 0.0
        
        # Usage frequency (moderate usage is good)
        usage_count = tag_data.get('usage_count', 0)
        if 2 <= usage_count <= 50:
            score += 0.3
        elif usage_count > 50:
            score += 0.2  # Slightly penalize overused tags
        
        # Average confidence of notes using this tag
        avg_confidence = tag_data.get('avg_confidence', 0.0)
        score += avg_confidence * 0.3
        
        # Average semantic density of notes using this tag
        avg_density = tag_data.get('avg_semantic_density', 0.0)
        score += avg_density * 0.2
        
        # Tag specificity (longer, more specific tags are better)
        if len(tag) >= 3 and len(tag) <= 15:
            score += 0.1
        
        # Tag format quality
        if re.match(r'^[a-z][a-z0-9_-]*$', tag):
            score += 0.1  # Good format
        
        return min(1.0, score)
    
    def get_tag_hierarchy(self) -> Dict[str, List[str]]:
        """
        Generate tag hierarchy based on usage patterns and relationships.
        
        Returns:
            Dictionary mapping parent tags to child tags
        """
        tag_stats = self.storage.get_tag_statistics()
        hierarchy = defaultdict(list)
        
        # Group tags by common prefixes
        tags_by_prefix = defaultdict(list)
        for tag in tag_stats.keys():
            if '.' in tag:
                prefix, suffix = tag.split('.', 1)
                tags_by_prefix[prefix].append(suffix)
            else:
                # Check if this tag is a prefix for other tags
                for other_tag in tag_stats.keys():
                    if other_tag.startswith(tag + '.'):
                        hierarchy[tag].append(other_tag)
        
        # Add explicit hierarchy from prefix groups
        for prefix, suffixes in tags_by_prefix.items():
            hierarchy[prefix].extend([f"{prefix}.{suffix}" for suffix in suffixes])
        
        return dict(hierarchy)
    
    def suggest_tag_consolidation(self) -> List[TagConsolidation]:
        """
        Suggest tag consolidations based on similarity and usage patterns.
        
        Returns:
            List of tag consolidation suggestions
        """
        tag_stats = self.storage.get_tag_statistics()
        consolidations = []
        
        # Find similar tags
        tags = list(tag_stats.keys())
        
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                similarity = self._calculate_tag_similarity(tag1, tag2)
                
                if similarity > 0.7:  # High similarity threshold
                    # Determine which tag to keep
                    score1 = self.calculate_tag_quality(tag1)
                    score2 = self.calculate_tag_quality(tag2)
                    
                    if score1 > score2:
                        suggested_tag = tag1
                        source_tags = [tag2]
                    else:
                        suggested_tag = tag2
                        source_tags = [tag1]
                    
                    consolidation = TagConsolidation(
                        source_tags=source_tags,
                        suggested_tag=suggested_tag,
                        confidence=similarity,
                        reason=f"High similarity ({similarity:.2f}) between tags"
                    )
                    consolidations.append(consolidation)
        
        # Sort by confidence and return top suggestions
        consolidations.sort(key=lambda x: x.confidence, reverse=True)
        return consolidations[:10]
    
    def _extract_potential_tags(self, text: str) -> List[str]:
        """
        Extract potential tags from text content.
        """
        # Convert to lowercase and tokenize
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())
        
        # Filter out common words and short words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        potential_tags = []
        for word in words:
            if (len(word) >= 3 and 
                word not in stopwords and 
                not word.isdigit()):
                potential_tags.append(word)
        
        return list(set(potential_tags))
    
    def _calculate_tag_confidence(self, tag: str, note_body: str, tag_stats: Dict) -> float:
        """
        Calculate confidence score for a tag suggestion.
        """
        confidence = 0.0
        
        # Frequency in note body
        tag_lower = tag.lower()
        note_lower = note_body.lower()
        occurrences = note_lower.count(tag_lower)
        if occurrences > 0:
            confidence += min(0.4, occurrences * 0.1)
        
        # Existing tag usage (popular tags get higher confidence)
        if tag in tag_stats:
            usage_count = tag_stats[tag].get('usage_count', 0)
            if usage_count > 0:
                confidence += min(0.3, usage_count * 0.01)
        
        # Tag length and format
        if 3 <= len(tag) <= 15:
            confidence += 0.1
        
        if re.match(r'^[a-z][a-z0-9_-]*$', tag):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _get_tag_suggestion_reason(self, tag: str, note_body: str, tag_stats: Dict) -> str:
        """
        Generate human-readable reason for tag suggestion.
        """
        reasons = []
        
        # Check frequency in note
        tag_lower = tag.lower()
        note_lower = note_body.lower()
        occurrences = note_lower.count(tag_lower)
        if occurrences > 0:
            reasons.append(f"appears {occurrences} time(s) in note")
        
        # Check existing usage
        if tag in tag_stats:
            usage_count = tag_stats[tag].get('usage_count', 0)
            if usage_count > 0:
                reasons.append(f"used {usage_count} time(s) previously")
        
        # Check similarity to existing tags
        similar_tags = self._find_similar_tags(tag, tag_stats.keys())
        if similar_tags:
            reasons.append(f"similar to existing tags: {', '.join(similar_tags[:3])}")
        
        return "; ".join(reasons) if reasons else "semantic analysis"
    
    def _suggest_by_co_occurrence(self, existing_tags: List[str], tag_stats: Dict) -> List[TagSuggestion]:
        """
        Suggest tags based on co-occurrence patterns with existing tags.
        """
        suggestions = []
        
        if not existing_tags:
            return suggestions
        
        # Get notes with existing tags to find co-occurrence patterns
        co_occurrence_patterns = defaultdict(int)
        
        for tag in existing_tags:
            notes = self.storage.retrieve_notes_by_tag(tag)
            for note in notes:
                for note_tag in note.tags:
                    if note_tag not in existing_tags:
                        co_occurrence_patterns[note_tag] += 1
        
        # Create suggestions based on co-occurrence
        for tag, count in co_occurrence_patterns.items():
            if count >= 2:  # Minimum co-occurrence threshold
                confidence = min(0.8, count * 0.1)
                suggestion = TagSuggestion(
                    tag=tag,
                    confidence=confidence,
                    reason=f"co-occurs {count} times with existing tags",
                    source='co_occurrence'
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_tag_similarity(self, tag1: str, tag2: str) -> float:
        """
        Calculate similarity between two tags.
        """
        # Check cache first
        cache_key = tuple(sorted([tag1, tag2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        similarity = 0.0
        
        # Exact match
        if tag1 == tag2:
            similarity = 1.0
        # Prefix match
        elif tag1.startswith(tag2) or tag2.startswith(tag1):
            similarity = 0.8
        # Contains match
        elif tag1 in tag2 or tag2 in tag1:
            similarity = 0.6
        # Levenshtein distance (simplified)
        else:
            # Simple character overlap
            chars1 = set(tag1)
            chars2 = set(tag2)
            overlap = len(chars1.intersection(chars2))
            total = len(chars1.union(chars2))
            if total > 0:
                similarity = overlap / total * 0.4
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _find_similar_tags(self, tag: str, all_tags: Set[str]) -> List[str]:
        """
        Find tags similar to the given tag.
        """
        similar = []
        for other_tag in all_tags:
            if other_tag != tag:
                similarity = self._calculate_tag_similarity(tag, other_tag)
                if similarity > 0.5:
                    similar.append(other_tag)
        
        return sorted(similar, key=lambda x: self._calculate_tag_similarity(tag, x), reverse=True)


if __name__ == "__main__":
    # Test the tag intelligence system
    storage = StorageEngine()
    tag_intelligence = TagIntelligence(storage)
    
    # Test tag suggestions
    test_note = "Implemented new machine learning feature for data processing pipeline"
    suggestions = tag_intelligence.suggest_tags(test_note)
    
    print("Tag Suggestions:")
    for suggestion in suggestions:
        print(f"  {suggestion.tag} (confidence: {suggestion.confidence:.2f}) - {suggestion.reason}")
    
    # Test drift detection
    drift_report = tag_intelligence.detect_tag_drift()
    print(f"\nTag Drift Report:")
    print(f"  Emerging tags: {drift_report.emerging_tags}")
    print(f"  Dying tags: {drift_report.dying_tags}")
    print(f"  Drift score: {drift_report.drift_score:.2f}")
    
    # Test tag quality
    tag_stats = storage.get_tag_statistics()
    if tag_stats:
        sample_tag = list(tag_stats.keys())[0]
        quality = tag_intelligence.calculate_tag_quality(sample_tag)
        print(f"\nTag Quality for '{sample_tag}': {quality:.2f}")
    
    # Test consolidation suggestions
    consolidations = tag_intelligence.suggest_tag_consolidation()
    print(f"\nConsolidation Suggestions:")
    for consolidation in consolidations:
        print(f"  {consolidation.source_tags} -> {consolidation.suggested_tag} (confidence: {consolidation.confidence:.2f})") 