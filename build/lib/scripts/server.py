#!/usr/bin/env python3
"""
QuickCapture — Background Brain Service

FastAPI server that keeps ML models loaded and handles note ingestion and semantics.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.storage_engine import StorageEngine
from scripts.parse_input import NeuralParser, ContentType
from scripts.models import ParsedNote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuickCaptureServer")

app = FastAPI(title="QuickCapture Brain")

# Global instances
storage: Optional[StorageEngine] = None
parser: Optional[NeuralParser] = None

class NoteRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

@app.on_event("startup")
async def startup_event():
    """Initialize models and storage on startup."""
    global storage, parser
    logger.info("Initializing Storage Engine (Loading Vector Models)...")
    storage = StorageEngine()
    
    logger.info("Initializing Neural Parser (Connecting to Ollama)...")
    parser = NeuralParser()
    logger.info("QuickCapture Brain Ready!")

@app.post("/capture")
def capture_note(req: NoteRequest, background_tasks: BackgroundTasks):
    """
    Capture a note using Neural Parser.
    Returns immediately, processing happens in background (or await if fast enough).
    We use synchronous def to let FastAPI run this in a threadpool, preventing event loop blocking.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    
    logger.info(f"Received note: {req.text[:50]}...")
    
    # 1. Parse with LLM
    try:
        cognitive_data = parser.parse(req.text)
        if not cognitive_data:
             raise ValueError("Neural parser returned None")
    except Exception as e:
        logger.error(f"Neural parsing failed: {e}")
        raise HTTPException(status_code=500, detail="Parsing failed")
    
    # 2. Convert to ParsedNote
    # Map intent to ContentType
    intent_map = {
        "task": ContentType.TASK,
        "idea": ContentType.IDEA,
        "meeting": ContentType.MEETING,
        "reference": ContentType.REFERENCE,
        "code": ContentType.CODE,
        "journal": ContentType.JOURNAL,
        "log": ContentType.GENERAL
    }
    
    content_type = intent_map.get(cognitive_data.get('intent', 'log'), ContentType.GENERAL)
    
    parsed_note = ParsedNote(
        note=req.text,
        tags=cognitive_data.get('tags', []),
        content_type=content_type,
        semantic_density=0.0, # TODO: calculate if needed
        confidence_score=1.0, # LLM assumption
        snr_metadata={
            "summary": cognitive_data.get('summary'),
            "sentiment": cognitive_data.get('sentiment'),
            "entities": cognitive_data.get('entities'),
            "action_items": cognitive_data.get('action_items'),
            "people": cognitive_data.get('people'),
            "intent": cognitive_data.get('intent')
        }
    )
    
    # 3. Store
    success = storage.store_note(parsed_note)
    
    if success:
        return {"status": "captured", "note_id": parsed_note.note_id, "summary": cognitive_data.get('summary')}
    else:
        raise HTTPException(status_code=500, detail="Storage failed")

@app.get("/search")
async def search_notes(query: str, limit: int = 10):
    """Semantic search for notes."""
    results = storage.search_semantic(query, limit=limit)
    
    # Convert to simpler response
    response_items = []
    for note in results:
        response_items.append({
            "note_id": note.note_id,
            "text": note.note,
            "tags": note.tags,
            "score": 0.0, # FAISS distance not easily propagated yet in search_semantic return
            "date": note.timestamp,
            "summary": note.snr_metadata.get('summary', '')
        })
    
    return response_items

@app.get("/health")
async def health_check():
    """Basic health check for parser and storage."""
    parser_ok = parser is not None
    storage_status = {"ok": False, "error": "storage not initialized"}

    try:
        storage_status = storage.health() if storage else storage_status
    except Exception as e:
        logger.error(f"Storage health failed: {e}")
        storage_status = {"ok": False, "error": str(e)}

    overall_ok = parser_ok and storage_status.get("ok", False)

    return {
        "ok": overall_ok,
        "parser": {"ok": parser_ok},
        "storage": storage_status,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
