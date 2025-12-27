#!/usr/bin/env python3
"""
QuickCapture SNR Preprocessing

Enhanced text preprocessing and batch processing for downstream semantic alignment
with the Semantic Note Router (SNR) system.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from .models import ParsedNote, ContentType
from .parse_input import parse_note_input
from .validate_note import validate_note
from .storage_engine import StorageEngine
from .tag_intelligence import TagIntelligence
from observability.metrics_collector import record_note_ingestion
from observability.performance_tracker import track_operation

logger = logging.getLogger(__name__)


@dataclass
class SNRPreprocessingConfig:
    """Configuration for SNR preprocessing."""
    
    # Text preprocessing settings
    strip_problematic_chars: bool = True
    normalize_whitespace: bool = True
    preserve_semantic_structure: bool = True
    content_specific_optimization: bool = True
    
    # Quality thresholds
    min_semantic_density: float = 0.3
    min_confidence_score: float = 0.7
    min_tag_quality: float = 0.8
    
    # Batch processing settings
    batch_size: int = 100
    parallel_processing: bool = True
    max_workers: int = 4
    
    # SNR metadata settings
    include_quality_metrics: bool = True
    include_processing_metadata: bool = True
    include_tag_hierarchy: bool = True


class SNRPreprocessor:
    """Enhanced text preprocessing for SNR compatibility."""
    
    def __init__(self, config: Optional[SNRPreprocessingConfig] = None):
        self.config = config or SNRPreprocessingConfig()
        self.storage = StorageEngine()
        self.tag_intelligence = TagIntelligence(self.storage)
        
        # Compile regex patterns for efficiency
        self.problematic_chars_pattern = re.compile(r'[&<>"\']')
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s\-.,!?;:]')
        
        # Content-specific optimization patterns
        self.code_pattern = re.compile(r'`[^`]+`|```[\s\S]*?```')
        self.url_pattern = re.compile(r'https?://\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    def optimize_text_for_embeddings(self, text: str, content_type: Optional[ContentType] = None) -> str:
        """
        Enhanced text optimization for downstream vectorization.
        
        Args:
            text: Raw text to optimize
            content_type: Type of content for specific optimization
            
        Returns:
            Optimized text ready for embedding
        """
        if not text:
            return ""
        
        optimized = text
        
        # Strip problematic characters that can interfere with embeddings
        if self.config.strip_problematic_chars:
            optimized = self.problematic_chars_pattern.sub(' ', optimized)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            optimized = self.whitespace_pattern.sub(' ', optimized)
            optimized = optimized.strip()
        
        # Content-specific optimizations
        if self.config.content_specific_optimization and content_type:
            optimized = self._apply_content_specific_optimization(optimized, content_type)
        
        # Preserve semantic structure
        if self.config.preserve_semantic_structure:
            optimized = self._preserve_semantic_structure(optimized)
        
        return optimized
    
    def _apply_content_specific_optimization(self, text: str, content_type: ContentType) -> str:
        """Apply content-type specific optimizations."""
        if content_type == ContentType.CODE:
            # Preserve code structure but clean up formatting
            text = self.code_pattern.sub(lambda m: m.group().replace('\n', ' '), text)
            text = re.sub(r'\s+', ' ', text)
        
        elif content_type == ContentType.REFERENCE:
            # Extract meaningful parts from URLs
            text = self.url_pattern.sub(lambda m: self._extract_url_meaning(m.group()), text)
        
        elif content_type == ContentType.CODE:
            # Preserve technical terms and acronyms
            text = self._preserve_technical_terms(text)
        
        elif content_type == ContentType.IDEA:
            # Enhance conceptual clarity
            text = self._enhance_conceptual_clarity(text)
        
        return text
    
    def _preserve_semantic_structure(self, text: str) -> str:
        """Preserve important semantic structure while cleaning."""
        # Preserve sentence boundaries
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Preserve list structures
        text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
        
        # Preserve emphasis markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        
        return text
    
    def _extract_url_meaning(self, url: str) -> str:
        """Extract meaningful parts from URLs."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path.strip('/')
            
            if path:
                return f"{domain} - {path.replace('-', ' ').replace('_', ' ')}"
            else:
                return domain
        except:
            return url
    
    def _extract_email_meaning(self, email: str) -> str:
        """Extract meaningful parts from email addresses."""
        username = email.split('@')[0]
        domain = email.split('@')[1].split('.')[0]
        return f"{username} at {domain}"
    
    def _preserve_technical_terms(self, text: str) -> str:
        """Preserve technical terms and acronyms."""
        # Common technical acronyms
        acronyms = ['API', 'HTTP', 'HTTPS', 'JSON', 'XML', 'SQL', 'NoSQL', 'REST', 'GraphQL', 'OAuth', 'JWT']
        
        for acronym in acronyms:
            text = re.sub(rf'\b{acronym}\b', f'<acronym>{acronym}</acronym>', text, flags=re.IGNORECASE)
        
        return text
    
    def _enhance_conceptual_clarity(self, text: str) -> str:
        """Enhance conceptual clarity for better semantic understanding."""
        # Add context for abstract concepts
        concept_mappings = {
            'epistemology': 'epistemology (theory of knowledge)',
            'ontology': 'ontology (study of being)',
            'methodology': 'methodology (study of methods)',
            'paradigm': 'paradigm (framework of thought)',
        }
        
        for concept, enhanced in concept_mappings.items():
            text = re.sub(rf'\b{concept}\b', enhanced, text, flags=re.IGNORECASE)
        
        return text
    
    def process_note_for_snr(self, note: ParsedNote) -> Dict[str, Any]:
        """
        Process a single note for SNR compatibility.
        
        Args:
            note: ParsedNote object to process
            
        Returns:
            Dictionary with SNR-optimized data
        """
        start_time = time.time()
        
        try:
            # Optimize text for embeddings
            optimized_body = self.optimize_text_for_embeddings(note.note, note.content_type)
            
            # Calculate additional quality metrics
            semantic_coherence = self._calculate_semantic_coherence(optimized_body)
            embedding_quality_estimate = self._estimate_embedding_quality(optimized_body)
            
            # Generate tag hierarchy alignment
            tag_hierarchy = self._generate_tag_hierarchy_alignment(note.tags)
            
            # Create SNR metadata
            snr_metadata = {
                'original_note_id': note.note_id,
                'processing_timestamp': time.time(),
                'optimization_applied': True,
                'content_type': note.content_type.value,
                'semantic_coherence_score': semantic_coherence,
                'embedding_quality_estimate': embedding_quality_estimate,
                'tag_hierarchy': tag_hierarchy,
                'quality_metrics': {
                    'semantic_density': note.semantic_density,
                    'tag_quality_score': note.tag_quality_score,
                    'confidence_score': note.confidence_score,
                    'validation_success': note.valid,
                },
                'processing_metadata': {
                    'optimization_version': '1.0',
                    'processing_duration_ms': (time.time() - start_time) * 1000,
                    'text_length_original': len(note.note),
                    'text_length_optimized': len(optimized_body),
                }
            }
            
            # Create SNR-compatible note structure
            snr_note = {
                'id': f"snr_{note.note_id}",
                'body': optimized_body,
                'tags': note.tags,
                'metadata': snr_metadata,
                'timestamp': note.timestamp,
                'origin': 'quickcapture',
                'version': note.version,
                'quality_score': self._calculate_overall_quality_score(note, semantic_coherence),
                'snr_compatible': True
            }
            
            return snr_note
            
        except Exception as e:
            logger.error(f"Error processing note {note.note_id} for SNR: {e}")
            return {
                'id': f"snr_{note.note_id}",
                'body': note.note,  # Fallback to original
                'tags': note.tags,
                'metadata': {'error': str(e), 'snr_compatible': False},
                'timestamp': note.timestamp,
                'origin': 'quickcapture',
                'version': note.version,
                'quality_score': 0.0,
                'snr_compatible': False
            }
    
    def process_batch_for_snr(self, notes: List[ParsedNote]) -> List[Dict[str, Any]]:
        """
        Process a batch of notes for SNR compatibility.
        
        Args:
            notes: List of ParsedNote objects to process
            
        Returns:
            List of SNR-compatible note dictionaries
        """
        with track_operation("snr_batch_processing"):
            processed_notes = []
            
            for note in notes:
                try:
                    processed_note = self.process_note_for_snr(note)
                    processed_notes.append(processed_note)
                except Exception as e:
                    logger.error(f"Error processing note {note.note_id} in batch: {e}")
                    # Add error note to maintain batch integrity
                    processed_notes.append({
                        'id': f"snr_{note.note_id}",
                        'body': note.note,
                        'tags': note.tags,
                        'metadata': {'error': str(e), 'snr_compatible': False},
                        'timestamp': note.timestamp,
                        'origin': 'quickcapture',
                        'version': note.version,
                        'quality_score': 0.0,
                        'snr_compatible': False
                    })
            
            return processed_notes
    
    def process_all_notes_for_snr(self, quality_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Process all notes in storage for SNR compatibility.
        
        Args:
            quality_threshold: Minimum quality score for inclusion
            
        Returns:
            List of SNR-compatible note dictionaries
        """
        with track_operation("snr_full_processing"):
            # Get all notes from storage (using confidence range to get all)
            all_notes = self.storage.get_notes_by_confidence_range(0.0, 1.0, 10000)
            
            # Filter by quality threshold
            quality_notes = [
                note for note in all_notes 
                if self._calculate_overall_quality_score(note) >= quality_threshold
            ]
            
            logger.info(f"Processing {len(quality_notes)} notes for SNR (quality threshold: {quality_threshold})")
            
            # Process in batches
            processed_notes = []
            for i in range(0, len(quality_notes), self.config.batch_size):
                batch = quality_notes[i:i + self.config.batch_size]
                processed_batch = self.process_batch_for_snr(batch)
                processed_notes.extend(processed_batch)
                
                logger.info(f"Processed batch {i//self.config.batch_size + 1}/{(len(quality_notes) + self.config.batch_size - 1)//self.config.batch_size}")
            
            return processed_notes
    
    def _calculate_semantic_coherence(self, text: str) -> float:
        """Calculate semantic coherence score for optimized text."""
        if not text:
            return 0.0
        
        # Simple heuristic based on text characteristics
        words = text.split()
        if len(words) < 3:
            return 0.5
        
        # Calculate word diversity
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words)
        
        # Calculate sentence complexity
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Normalize scores
        diversity_score = min(diversity_ratio * 2, 1.0)  # Higher diversity is better
        complexity_score = min(avg_sentence_length / 20, 1.0)  # Moderate complexity is better
        
        return (diversity_score + complexity_score) / 2
    
    def _estimate_embedding_quality(self, text: str) -> float:
        """Estimate embedding quality for the optimized text."""
        if not text:
            return 0.0
        
        # Factors that affect embedding quality
        factors = []
        
        # Text length (moderate length is optimal)
        length_score = min(len(text) / 500, 1.0)  # Normalize to 500 chars
        factors.append(length_score)
        
        # Word count
        word_count = len(text.split())
        word_score = min(word_count / 100, 1.0)  # Normalize to 100 words
        factors.append(word_score)
        
        # Vocabulary richness
        unique_words = len(set(text.lower().split()))
        vocab_score = min(unique_words / 50, 1.0)  # Normalize to 50 unique words
        factors.append(vocab_score)
        
        # Semantic density (avoid repetitive content)
        words = text.split()
        if words:
            most_common_word = max(set(words), key=words.count)
            repetition_ratio = words.count(most_common_word) / len(words)
            density_score = 1.0 - repetition_ratio
            factors.append(density_score)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _generate_tag_hierarchy_alignment(self, tags: List[str]) -> Dict[str, Any]:
        """Generate tag hierarchy alignment for SNR."""
        try:
            hierarchy = self.tag_intelligence.get_tag_hierarchy()
            
            # Map tags to hierarchy levels
            tag_mapping = {}
            for tag in tags:
                for category, category_tags in hierarchy.items():
                    if tag in category_tags:
                        tag_mapping[tag] = {
                            'category': category,
                            'level': 1,
                            'confidence': 0.9
                        }
                        break
                else:
                    # Tag not in predefined hierarchy
                    tag_mapping[tag] = {
                        'category': 'uncategorized',
                        'level': 0,
                        'confidence': 0.5
                    }
            
            return {
                'tag_mapping': tag_mapping,
                'hierarchy_version': '1.0',
                'coverage_percentage': len([t for t in tag_mapping.values() if t['category'] != 'uncategorized']) / len(tags) if tags else 0
            }
            
        except Exception as e:
            logger.warning(f"Error generating tag hierarchy: {e}")
            return {
                'tag_mapping': {tag: {'category': 'unknown', 'level': 0, 'confidence': 0.0} for tag in tags},
                'hierarchy_version': 'error',
                'coverage_percentage': 0.0
            }
    
    def _calculate_overall_quality_score(self, note: ParsedNote, semantic_coherence: Optional[float] = None) -> float:
        """Calculate overall quality score for SNR processing."""
        if semantic_coherence is None:
            semantic_coherence = self._calculate_semantic_coherence(note.note)
        
        # Weighted combination of quality metrics
        weights = {
            'semantic_density': 0.25,
            'tag_quality': 0.20,
            'confidence': 0.20,
            'semantic_coherence': 0.20,
            'validation_success': 0.15
        }
        
        scores = {
            'semantic_density': note.semantic_density,
            'tag_quality': note.tag_quality_score,
            'confidence': note.confidence_score,
            'semantic_coherence': semantic_coherence,
            'validation_success': 1.0 if note.valid else 0.0
        }
        
        overall_score = sum(weights[metric] * scores[metric] for metric in weights)
        return min(overall_score, 1.0)
    
    def export_snr_batch(self, output_path: str, quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Export all notes as SNR-compatible batch file.
        
        Args:
            output_path: Path to save the batch file
            quality_threshold: Minimum quality score for inclusion
            
        Returns:
            Export summary
        """
        with track_operation("snr_batch_export"):
            # Process all notes
            processed_notes = self.process_all_notes_for_snr(quality_threshold)
            
            # Create export metadata
            export_metadata = {
                'export_timestamp': time.time(),
                'total_notes': len(processed_notes),
                'quality_threshold': quality_threshold,
                'snr_compatible_count': len([n for n in processed_notes if n.get('snr_compatible', False)]),
                'average_quality_score': sum(n.get('quality_score', 0) for n in processed_notes) / len(processed_notes) if processed_notes else 0,
                'processing_config': {
                    'strip_problematic_chars': self.config.strip_problematic_chars,
                    'normalize_whitespace': self.config.normalize_whitespace,
                    'preserve_semantic_structure': self.config.preserve_semantic_structure,
                    'content_specific_optimization': self.config.content_specific_optimization,
                }
            }
            
            # Create export structure
            export_data = {
                'metadata': export_metadata,
                'notes': processed_notes
            }
            
            # Save to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(processed_notes)} notes to {output_path}")
            
            return export_metadata


# Convenience functions
def optimize_text_for_embeddings(text: str, content_type: Optional[ContentType] = None) -> str:
    """Convenience function to optimize text for embeddings."""
    preprocessor = SNRPreprocessor()
    return preprocessor.optimize_text_for_embeddings(text, content_type)


def process_batch_for_snr(notes: List[ParsedNote]) -> List[Dict[str, Any]]:
    """Convenience function to process batch for SNR."""
    preprocessor = SNRPreprocessor()
    return preprocessor.process_batch_for_snr(notes)


def export_snr_batch(output_path: str, quality_threshold: float = 0.7) -> Dict[str, Any]:
    """Convenience function to export SNR batch."""
    preprocessor = SNRPreprocessor()
    return preprocessor.export_snr_batch(output_path, quality_threshold) 