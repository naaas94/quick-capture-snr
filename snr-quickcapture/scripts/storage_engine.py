#!/usr/bin/env python3
"""
Intelligent Storage Engine

Hybrid storage with SQLite primary storage, vector store for semantic search,
and JSONL backup with atomic operations.
"""

import json
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import logging
from contextlib import contextmanager

from .models import ParsedNote, BatchProcessingResult
from .parse_input import ContentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageEngine:
    """
    Hybrid storage engine with SQLite primary storage and vector store for semantic search.
    """
    
    def __init__(self, db_path: str = "storage/quickcapture.db", vector_store_path: str = "storage/vector_store"):
        """
        Initialize storage engine.
        
        Args:
            db_path: Path to SQLite database
            vector_store_path: Path to vector store directory
        """
        self.db_path = Path(db_path)
        self.vector_store_path = Path(vector_store_path)
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Vector store placeholder (will be implemented with sentence-transformers)
        self.vector_store = None
        self._init_vector_store()
    
    def _init_database(self):
        """Initialize SQLite database with proper schema and indexing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create notes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    tags TEXT,  -- JSON array
                    note_body TEXT,
                    comment TEXT,
                    timestamp TEXT,
                    version INTEGER,
                    semantic_density REAL,
                    tag_quality_score REAL,
                    content_type TEXT,
                    confidence_score REAL,
                    embedding_vector BLOB,
                    snr_metadata TEXT,  -- JSON
                    valid BOOLEAN,
                    issues TEXT,  -- JSON array
                    origin TEXT,
                    raw_text TEXT,
                    tag_hierarchy TEXT,  -- JSON
                    co_occurrence_patterns TEXT,  -- JSON
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON notes(tags)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON notes(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_density ON notes(semantic_density)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON notes(content_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence_score ON notes(confidence_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_valid ON notes(valid)")
            
            # Create tags table for tag statistics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tag_statistics (
                    tag TEXT PRIMARY KEY,
                    usage_count INTEGER DEFAULT 0,
                    first_used TEXT,
                    last_used TEXT,
                    avg_confidence REAL,
                    avg_semantic_density REAL
                )
            """)
            
            # Create processing metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    processing_time_ms REAL,
                    parsing_confidence REAL,
                    validation_score REAL,
                    storage_success BOOLEAN,
                    errors TEXT,  -- JSON array
                    warnings TEXT  -- JSON array
                )
            """)
            
            conn.commit()
    
    def _init_vector_store(self):
        """Initialize vector store for semantic search."""
        # Placeholder for vector store initialization
        # In a full implementation, this would use sentence-transformers
        # and a vector database like FAISS or Chroma
        logger.info("Vector store initialization placeholder - semantic search not yet implemented")
    
    def _get_connection(self):
        """Get or create a persistent database connection."""
        if not hasattr(self, '_persistent_conn') or self._persistent_conn is None:
            self._persistent_conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._persistent_conn.row_factory = sqlite3.Row  # Enable dict-like access
        return self._persistent_conn
    
    def store_note(self, note: ParsedNote) -> bool:
        """
        Store a note with atomic operations.
        
        Args:
            note: ParsedNote object to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if note already exists
                cursor.execute("SELECT note_id FROM notes WHERE note_id = ?", (note.note_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing note
                    cursor.execute("""
                        UPDATE notes SET
                            tags = ?, note_body = ?, comment = ?, timestamp = ?, version = ?,
                            semantic_density = ?, tag_quality_score = ?, content_type = ?,
                            confidence_score = ?, embedding_vector = ?, snr_metadata = ?,
                            valid = ?, issues = ?, origin = ?, raw_text = ?,
                            tag_hierarchy = ?, co_occurrence_patterns = ?, updated_at = ?
                        WHERE note_id = ?
                    """, (
                        json.dumps(note.tags),
                        note.note,
                        note.comment,
                        note.timestamp,
                        note.version,
                        note.semantic_density,
                        note.tag_quality_score,
                        note.content_type.value,
                        note.confidence_score,
                        pickle.dumps(note.embedding_vector) if note.embedding_vector else None,
                        json.dumps(note.snr_metadata),
                        note.valid,
                        json.dumps(note.issues),
                        note.origin,
                        note.raw_text,
                        json.dumps(note.tag_hierarchy),
                        json.dumps(note.co_occurrence_patterns),
                        datetime.now(timezone.utc).isoformat(),
                        note.note_id
                    ))
                else:
                    # Insert new note
                    cursor.execute("""
                        INSERT INTO notes (
                            note_id, tags, note_body, comment, timestamp, version,
                            semantic_density, tag_quality_score, content_type,
                            confidence_score, embedding_vector, snr_metadata,
                            valid, issues, origin, raw_text, tag_hierarchy,
                            co_occurrence_patterns, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        note.note_id,
                        json.dumps(note.tags),
                        note.note,
                        note.comment,
                        note.timestamp,
                        note.version,
                        note.semantic_density,
                        note.tag_quality_score,
                        note.content_type.value,
                        note.confidence_score,
                        pickle.dumps(note.embedding_vector) if note.embedding_vector else None,
                        json.dumps(note.snr_metadata),
                        note.valid,
                        json.dumps(note.issues),
                        note.origin,
                        note.raw_text,
                        json.dumps(note.tag_hierarchy),
                        json.dumps(note.co_occurrence_patterns),
                        datetime.now(timezone.utc).isoformat(),
                        datetime.now(timezone.utc).isoformat()
                    ))
                
                # Update tag statistics
                self._update_tag_statistics(note, cursor)
                
                conn.commit()
                logger.info(f"Successfully stored note {note.note_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store note {note.note_id}: {e}")
            return False
    
    def _update_tag_statistics(self, note: ParsedNote, cursor: sqlite3.Cursor):
        """Update tag usage statistics."""
        now = datetime.now(timezone.utc).isoformat()
        
        for tag in note.tags:
            cursor.execute("""
                INSERT OR REPLACE INTO tag_statistics (
                    tag, usage_count, first_used, last_used, avg_confidence, avg_semantic_density
                ) VALUES (
                    ?,
                    COALESCE((SELECT usage_count FROM tag_statistics WHERE tag = ?), 0) + 1,
                    COALESCE((SELECT first_used FROM tag_statistics WHERE tag = ?), ?),
                    ?,
                    (COALESCE((SELECT avg_confidence FROM tag_statistics WHERE tag = ?), 0) + ?) / 2,
                    (COALESCE((SELECT avg_semantic_density FROM tag_statistics WHERE tag = ?), 0) + ?) / 2
                )
            """, (tag, tag, tag, now, now, tag, note.confidence_score, tag, note.semantic_density))
    
    def retrieve_notes_by_tag(self, tag: str, limit: int = 100) -> List[ParsedNote]:
        """
        Retrieve notes by tag with optional limit.
        
        Args:
            tag: Tag to search for
            limit: Maximum number of notes to return
            
        Returns:
            List of ParsedNote objects
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM notes 
                    WHERE tags LIKE ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (f'%"{tag}"%', limit))
                
                rows = cursor.fetchall()
                notes = []
                
                for row in rows:
                    note = self._row_to_parsed_note(row)
                    notes.append(note)
                
                return notes
                
        except Exception as e:
            logger.error(f"Failed to retrieve notes by tag {tag}: {e}")
            return []
    
    def search_semantic(self, query: str, limit: int = 10) -> List[ParsedNote]:
        """
        Semantic search using vector store.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of ParsedNote objects ordered by relevance
        """
        # Placeholder for semantic search
        # In a full implementation, this would use sentence-transformers
        # to generate embeddings and perform similarity search
        
        logger.info(f"Semantic search for '{query}' (placeholder implementation)")
        
        # Fallback to simple text search
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM notes 
                    WHERE note_body LIKE ? OR comment LIKE ?
                    ORDER BY confidence_score DESC, timestamp DESC 
                    LIMIT ?
                """, (f'%{query}%', f'%{query}%', limit))
                
                rows = cursor.fetchall()
                notes = []
                
                for row in rows:
                    note = self._row_to_parsed_note(row)
                    notes.append(note)
                
                return notes
                
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    def get_tag_statistics(self) -> Dict[str, int]:
        """
        Get tag usage statistics.
        
        Returns:
            Dictionary mapping tags to usage counts
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT tag, usage_count, avg_confidence, avg_semantic_density
                    FROM tag_statistics 
                    ORDER BY usage_count DESC
                """)
                
                stats = {}
                for row in cursor.fetchall():
                    stats[row['tag']] = {
                        'usage_count': row['usage_count'],
                        'avg_confidence': row['avg_confidence'],
                        'avg_semantic_density': row['avg_semantic_density']
                    }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get tag statistics: {e}")
            return {}
    
    def get_notes_by_confidence_range(self, min_confidence: float = 0.0, max_confidence: float = 1.0, limit: int = 100) -> List[ParsedNote]:
        """
        Get notes within a confidence score range.
        
        Args:
            min_confidence: Minimum confidence score
            max_confidence: Maximum confidence score
            limit: Maximum number of notes to return
            
        Returns:
            List of ParsedNote objects
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM notes 
                    WHERE confidence_score BETWEEN ? AND ?
                    ORDER BY confidence_score DESC, timestamp DESC 
                    LIMIT ?
                """, (min_confidence, max_confidence, limit))
                
                rows = cursor.fetchall()
                notes = []
                
                for row in rows:
                    note = self._row_to_parsed_note(row)
                    notes.append(note)
                
                return notes
                
        except Exception as e:
            logger.error(f"Failed to get notes by confidence range: {e}")
            return []
    
    def get_notes_with_issues(self, limit: int = 100) -> List[ParsedNote]:
        """
        Get notes that have validation issues.
        
        Args:
            limit: Maximum number of notes to return
            
        Returns:
            List of ParsedNote objects with issues
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM notes 
                    WHERE valid = 0 OR issues != '[]'
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                notes = []
                
                for row in rows:
                    note = self._row_to_parsed_note(row)
                    notes.append(note)
                
                return notes
                
        except Exception as e:
            logger.error(f"Failed to get notes with issues: {e}")
            return []
    
    def backup_to_jsonl(self, backup_path: str = "storage/backup/") -> bool:
        """
        Create JSONL backup of all notes.
        
        Args:
            backup_path: Directory to store backup files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"quickcapture_backup_{timestamp}.jsonl"
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM notes ORDER BY timestamp")
                
                with open(backup_file, 'w', encoding='utf-8') as f:
                    for row in cursor.fetchall():
                        note = self._row_to_parsed_note(row)
                        f.write(note.to_json() + '\n')
            
            logger.info(f"Backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def _row_to_parsed_note(self, row: sqlite3.Row) -> ParsedNote:
        """Convert database row to ParsedNote object."""
        return ParsedNote(
            note_id=row['note_id'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            note=row['note_body'],
            comment=row['comment'],
            timestamp=row['timestamp'],
            version=row['version'],
            semantic_density=row['semantic_density'],
            tag_quality_score=row['tag_quality_score'],
            content_type=ContentType(row['content_type']),
            confidence_score=row['confidence_score'],
            embedding_vector=pickle.loads(row['embedding_vector']) if row['embedding_vector'] else None,
            snr_metadata=json.loads(row['snr_metadata']) if row['snr_metadata'] else {},
            valid=bool(row['valid']),
            issues=json.loads(row['issues']) if row['issues'] else [],
            origin=row['origin'],
            raw_text=row['raw_text'],
            tag_hierarchy=json.loads(row['tag_hierarchy']) if row['tag_hierarchy'] else {},
            co_occurrence_patterns=json.loads(row['co_occurrence_patterns']) if row['co_occurrence_patterns'] else {}
        )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notes'")
                if not cursor.fetchone():
                    # Tables don't exist yet, return empty stats
                    return {
                        'total_notes': 0,
                        'valid_notes': 0,
                        'notes_with_issues': 0,
                        'avg_confidence': 0.0,
                        'avg_semantic_density': 0.0,
                        'total_tags': 0,
                        'database_size_mb': 0.0
                    }
                
                # Total notes
                cursor.execute("SELECT COUNT(*) as count FROM notes")
                total_notes = cursor.fetchone()['count']
                
                # Valid notes
                cursor.execute("SELECT COUNT(*) as count FROM notes WHERE valid = 1")
                valid_notes = cursor.fetchone()['count']
                
                # Notes with issues
                cursor.execute("SELECT COUNT(*) as count FROM notes WHERE valid = 0 OR issues != '[]'")
                notes_with_issues = cursor.fetchone()['count']
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence_score) as avg_confidence FROM notes")
                avg_confidence = cursor.fetchone()['avg_confidence'] or 0.0
                
                # Average semantic density
                cursor.execute("SELECT AVG(semantic_density) as avg_density FROM notes")
                avg_density = cursor.fetchone()['avg_density'] or 0.0
                
                # Total tags
                cursor.execute("SELECT COUNT(*) as count FROM tag_statistics")
                total_tags = cursor.fetchone()['count']
                
                return {
                    'total_notes': total_notes,
                    'valid_notes': valid_notes,
                    'notes_with_issues': notes_with_issues,
                    'avg_confidence': round(avg_confidence, 3),
                    'avg_semantic_density': round(avg_density, 3),
                    'total_tags': total_tags,
                    'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                'total_notes': 0,
                'valid_notes': 0,
                'notes_with_issues': 0,
                'avg_confidence': 0.0,
                'avg_semantic_density': 0.0,
                'total_tags': 0,
                'database_size_mb': 0.0
            }


if __name__ == "__main__":
    # Test the storage engine
    storage = StorageEngine()
    
    # Test note
    from models import ParsedNote
    test_note = ParsedNote(
        tags=['python', 'coding'],
        note='Implemented new feature for data processing',
        comment='This will help with the ML pipeline',
        semantic_density=0.75,
        tag_quality_score=0.8,
        content_type=ContentType.CODE,
        confidence_score=0.85
    )
    
    # Store note
    success = storage.store_note(test_note)
    print(f"Storage success: {success}")
    
    # Get statistics
    stats = storage.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Get tag statistics
    tag_stats = storage.get_tag_statistics()
    print(f"Tag statistics: {tag_stats}")
    
    # Retrieve notes by tag
    notes = storage.retrieve_notes_by_tag('python')
    print(f"Found {len(notes)} notes with tag 'python'")
    
    # Search
    search_results = storage.search_semantic('data processing')
    print(f"Search results: {len(search_results)} notes") 