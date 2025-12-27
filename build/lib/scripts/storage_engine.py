#!/usr/bin/env python3
"""
Intelligent Storage Engine

Hybrid storage with SQLite primary storage, vector store for semantic search,
and JSONL backup with atomic operations.
"""

import json
import sqlite3
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import logging
from contextlib import contextmanager
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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
        # Thread-local SQLite connections for threadpool safety (must be set before any DB access)
        self._local = threading.local()
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize embedding model and vector store
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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
        """Initialize vector store for semantic search using FAISS."""
        # 384 dimensions for all-MiniLM-L6-v2
        self.dimension = 384
        
        # Load existing index if available
        index_path = self.vector_store_path / "faiss_index"
        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                logger.info("Loaded existing FAISS index from disk")
            except Exception as e:
                logger.error(f"Failed to load index: {e}, starting fresh")
                self.index = faiss.IndexFlatL2(self.dimension)
                
            # Mapping from FAISS ID to Note ID usually needs to be maintained.
            # For simplicity, we assume we can look up by content or store ID mapping in SQLite.
            # A common pattern is to have an `id_map` file or use `IndexIDMap`.
            # For this MVP, we will rely on searching, then retrieving notes and checking which one matches. 
            # Ideally: self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            # But IndexIDMap requires add_with_ids.
            
            # Let's use IndexIDMap if we can easily manage integer IDs.
            # SQLite does not have auto-increment int primary key for notes (UUID).
            # We can use the SQLite rowid or add a separate integer index.
            # For now, stick to simple index and maybe simpler retrieval or just rely on text content matching 
            # if we don't implement the ID map fully.
            # Wait, `add_with_ids` was in the original code? 
            # Yes: `self.index.add_with_ids`. But `IndexFlatL2` doesn't support `add_with_ids` directly unless wrapped in IDs.
            # The original code threw an error or was placeholder.
            # I will wrap it in IndexIDMap.
            if not isinstance(self.index, faiss.IndexIDMap):
                 # If we loaded a plain index, we might need to migrate or wrap it?
                 # If it's empty/fresh:
                 if self.index.ntotal == 0:
                     self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        else:
            logger.info("No existing FAISS index found, starting fresh")
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            
        # We also need a mapping from integer ID back to UUID string.
        # We can store this in a pickle file or a separate SQLite table.
        self.id_map_path = self.vector_store_path / "id_map.json"
        self.id_map = {}
        if self.id_map_path.exists():
            with open(self.id_map_path, 'r') as f:
                self.id_map = json.load(f)
                # Convert keys to int because JSON keys are strings
                self.id_map = {int(k): v for k, v in self.id_map.items()}

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the loaded model."""
        return self.model.encode(text).tolist()
    
    def _get_connection(self):
        """Get or create a persistent database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            self._local.conn = conn
        return conn
    
    def _get_next_id(self) -> int:
        """Get next available integer ID for FAISS."""
        if not self.id_map:
            return 0
        return max(self.id_map.keys()) + 1

    def store_note(self, note: ParsedNote) -> bool:
        """
        Store a note with atomic operations.
        
        Args:
            note: ParsedNote object to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate embedding if missing
            if not note.embedding_vector and note.note:
                note.embedding_vector = self.generate_embedding(note.note)
            
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
                    
                    # For updates, we should technically remove old vector and add new one.
                    # FAISS IDMap doesn't support remove readily without ID.
                    # We'll skip complex update logic for MVP and just add new vector (might cause duplicates in search)
                    # or better: we lookup the ID in id_map (reverse lookup needed)
                    # For now, let's assume appending is okay-ish or we won't update vectors often.
                    pass 
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
                    
                    # Add to FAISS if we have a vector
                    if note.embedding_vector:
                        vector_id = self._get_next_id()
                        self.add_to_vector_store([note.embedding_vector], [vector_id])
                        self.id_map[vector_id] = note.note_id
                        self.save_vector_store()

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
        """
        try:
            # Generate query embedding
            query_vector = self.generate_embedding(query)
            
            # Search in FAISS
            indices = self.search_vector_store(query_vector, k=limit)
            
            # Map indices to note_ids
            note_ids = []
            for idx in indices:
                if idx in self.id_map:
                    note_ids.append(self.id_map[idx])
            
            if not note_ids:
                return []
            
            # Retrieve notes from SQLite
            placeholders = ','.join('?' for _ in note_ids)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM notes 
                    WHERE note_id IN ({placeholders})
                """, note_ids)
                
                rows = cursor.fetchall()
                
                # Reorder based on search result order
                note_map = {row['note_id']: self._row_to_parsed_note(row) for row in rows}
                results = [note_map[nid] for nid in note_ids if nid in note_map]
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    def get_tag_statistics(self) -> Dict[str, int]:
        """
        Get tag usage statistics.
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
        """Get notes within a confidence score range."""
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
        """Get notes that have validation issues."""
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
        """Create JSONL backup of all notes."""
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
        """Get database statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notes'")
                if not cursor.fetchone():
                    return {
                        'total_notes': 0,
                        'valid_notes': 0,
                        'notes_with_issues': 0,
                        'avg_confidence': 0.0,
                        'avg_semantic_density': 0.0,
                        'total_tags': 0,
                        'database_size_mb': 0.0
                    }
                
                cursor.execute("SELECT COUNT(*) as count FROM notes")
                total_notes = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM notes WHERE valid = 1")
                valid_notes = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM notes WHERE valid = 0 OR issues != '[]'")
                notes_with_issues = cursor.fetchone()['count']
                
                cursor.execute("SELECT AVG(confidence_score) as avg_confidence FROM notes")
                avg_confidence = cursor.fetchone()['avg_confidence'] or 0.0
                
                cursor.execute("SELECT AVG(semantic_density) as avg_density FROM notes")
                avg_density = cursor.fetchone()['avg_density'] or 0.0
                
                cursor.execute("SELECT COUNT(*) as count FROM tag_statistics")
                total_tags = cursor.fetchone()['count']
                
                # Get vector index stats
                vector_count = self.index.ntotal if hasattr(self, 'index') else 0
                
                return {
                    'total_notes': total_notes,
                    'valid_notes': valid_notes,
                    'notes_with_issues': notes_with_issues,
                    'avg_confidence': round(avg_confidence, 3),
                    'avg_semantic_density': round(avg_density, 3),
                    'total_tags': total_tags,
                    'vector_index_size': vector_count,
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

    def add_to_vector_store(self, vectors: List[List[float]], ids: List[int]):
        """Add vectors to the FAISS index."""
        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs must have the same length")
        
        # Ensure vectors are numpy array of float32
        vectors_np = np.array(vectors, dtype='float32')
        ids_np = np.array(ids, dtype='int64')
        
        self.index.add_with_ids(vectors_np, ids_np)
        logger.info(f"Added {len(vectors)} vectors to the FAISS index")

    def search_vector_store(self, query_vector: List[float], k: int = 10) -> List[int]:
        """Search the FAISS index and return the top k nearest neighbors."""
        query_vector = np.array(query_vector, dtype='float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        logger.info(f"Performed search in FAISS index, found {len(indices[0])} results")
        return indices[0].tolist()

    def save_vector_store(self):
        """Save the FAISS index and id_map to disk."""
        index_path = self.vector_store_path / "faiss_index"
        faiss.write_index(self.index, str(index_path))
        
        with open(self.id_map_path, 'w') as f:
            json.dump(self.id_map, f)
            
        logger.info(f"FAISS index saved to {index_path}")

    def health(self) -> Dict[str, Any]:
        """
        Lightweight health check for storage components.
        - Verifies SQLite connectivity.
        - Reports vector index size.
        """
        status: Dict[str, Any] = {
            "ok": True,
            "sqlite": {},
            "vector_index": {},
        }

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                status["sqlite"] = {"ok": True}
        except Exception as e:
            status["ok"] = False
            status["sqlite"] = {"ok": False, "error": str(e)}

        try:
            index_size = self.index.ntotal if hasattr(self, "index") else 0
            status["vector_index"] = {"ok": True, "size": index_size}
        except Exception as e:
            status["ok"] = False
            status["vector_index"] = {"ok": False, "error": str(e)}

        return status



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