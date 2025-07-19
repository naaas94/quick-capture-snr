# Usage Examples

## Overview

This document provides practical examples of how to use QuickCapture for various note-taking and knowledge management scenarios. Each example includes the input, expected output, and explanation of the processing steps.

## Basic Usage Examples

### 1. Simple Note Addition

#### Input
```bash
python quick_add.py "Meeting notes from today's team standup. Discussed project timeline and identified blockers."
```

#### Expected Output
```
Note created successfully: a37e743064bf11f0970d05fa391d7ad1
Classification: work
Tags: [meeting, team, project]
Processing time: 1.2s
```

#### Processing Steps
1. **Input Parsing**: CLI text input is parsed and validated
2. **Preprocessing**: Text is cleaned and normalized
3. **Embedding**: Content is converted to semantic vector
4. **Classification**: Content is classified as "work" category
5. **Tagging**: Automatic tags are generated based on content
6. **Storage**: Note is stored in work storage with vector embedding

### 2. File-Based Note Processing

#### Input File (`meeting_notes.txt`)
```
Project Alpha - Weekly Review
Date: 2024-01-15

Agenda:
- Review sprint progress
- Discuss technical challenges
- Plan next sprint

Action Items:
1. Fix authentication bug
2. Update documentation
3. Schedule code review

Notes:
The team made good progress on the authentication system.
The main challenge is integrating with the legacy system.
We need to allocate more time for testing in the next sprint.
```

#### Command
```bash
python quick_add.py -f meeting_notes.txt -t "Project Alpha Review" -g project -g meeting
```

#### Expected Output
```
Note created successfully: b2c8073064bf11f0970d05fa391d7ad1
Classification: work
Tags: [project, meeting, sprint, authentication, documentation]
Processing time: 2.1s
```

### 3. Research Note with Metadata

#### Input
```json
{
  "content": "Machine learning models for natural language processing have evolved significantly. Transformer architectures like BERT and GPT have revolutionized the field by enabling better understanding of context and semantics.",
  "title": "NLP Model Evolution",
  "tags": ["machine-learning", "nlp", "research"],
  "metadata": {
    "source": "academic_paper",
    "author": "Research Team",
    "publication_date": "2024-01-10",
    "keywords": ["transformers", "BERT", "GPT", "semantics"]
  }
}
```

#### Command
```bash
python quick_add.py --json input.json
```

#### Expected Output
```
Note created successfully: c229dd2064bf11f0970d05fa391d7ad1
Classification: research
Tags: [machine-learning, nlp, research, transformers, semantics]
Processing time: 1.8s
```

## Advanced Usage Examples

### 4. Batch Processing

#### Input Directory Structure
```
batch_notes/
├── research/
│   ├── paper1.txt
│   ├── paper2.txt
│   └── paper3.txt
├── personal/
│   ├── diary_entry1.txt
│   └── diary_entry2.txt
└── work/
    ├── meeting1.txt
    └── meeting2.txt
```

#### Batch Processing Script
```python
import os
from pathlib import Path
from quick_add import QuickCaptureOrchestrator

def process_batch_directory(directory_path: str):
    orchestrator = QuickCaptureOrchestrator()
    results = []
    
    for file_path in Path(directory_path).rglob("*.txt"):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            result = orchestrator.process_note({
                'content': content,
                'title': file_path.stem,
                'source': str(file_path),
                'category': file_path.parent.name
            })
            
            results.append({
                'file': str(file_path),
                'success': result.success,
                'note_id': result.note_id if result.success else None,
                'errors': result.errors if not result.success else []
            })
            
        except Exception as e:
            results.append({
                'file': str(file_path),
                'success': False,
                'errors': [str(e)]
            })
    
    return results

# Usage
results = process_batch_directory("batch_notes/")
for result in results:
    if result['success']:
        print(f"✓ {result['file']} -> {result['note_id']}")
    else:
        print(f"✗ {result['file']} -> {', '.join(result['errors'])}")
```

#### Expected Output
```
✓ batch_notes/research/paper1.txt -> d63e3ea064bf11f0970d05fa391d7ad1
✓ batch_notes/research/paper2.txt -> e8f978c064bf11f0970d05fa391d7ad1
✓ batch_notes/personal/diary_entry1.txt -> febe978064bf11f0970d05fa391d7ad1
✓ batch_notes/work/meeting1.txt -> 1fb4178064c011f0970d05fa391d7ad1
```

### 5. API Integration

#### REST API Usage
```python
import requests
import json

def add_note_via_api(content, title=None, tags=None):
    url = "http://localhost:8000/notes"
    
    payload = {
        "content": content,
        "title": title,
        "tags": tags or []
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# Usage examples
try:
    # Simple note
    result = add_note_via_api(
        content="Quick reminder to check email",
        title="Email Reminder"
    )
    print(f"Note created: {result['note_id']}")
    
    # Research note
    result = add_note_via_api(
        content="Study on neural network architectures for image recognition",
        title="Neural Networks Research",
        tags=["research", "neural-networks", "image-recognition"]
    )
    print(f"Research note created: {result['note_id']}")
    
except Exception as e:
    print(f"Error: {e}")
```

### 6. Custom Classification Rules

#### Custom Classification Configuration
```yaml
# custom_classification.yaml
classification_rules:
  technical_documentation:
    keywords: ["api", "documentation", "code", "function", "class"]
    category: "technical"
    confidence_threshold: 0.6
  
  project_management:
    keywords: ["project", "timeline", "milestone", "deadline", "sprint"]
    category: "project"
    confidence_threshold: 0.7
  
  personal_reflection:
    keywords: ["thought", "feeling", "experience", "reflection", "personal"]
    category: "personal"
    confidence_threshold: 0.8
```

#### Usage with Custom Rules
```python
from quick_add import QuickCaptureOrchestrator
from config import ConfigurationManager

# Load custom configuration
config_manager = ConfigurationManager()
config_manager.update_config("tag_intelligence", {
    "custom_rules": "custom_classification.yaml"
})

orchestrator = QuickCaptureOrchestrator(config_manager.get_config())

# Test custom classification
content = "API documentation for the user authentication endpoint"
result = orchestrator.process_note({
    'content': content,
    'title': 'Auth API Docs'
})

print(f"Category: {result.classification.primary_category}")
print(f"Confidence: {result.classification.confidence_score}")
```

## Integration Examples

### 7. Web Application Integration

#### Flask Integration
```python
from flask import Flask, request, jsonify
from quick_add import QuickCaptureOrchestrator

app = Flask(__name__)
orchestrator = QuickCaptureOrchestrator()

@app.route('/api/notes', methods=['POST'])
def create_note():
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({'error': 'Content is required'}), 400
        
        result = orchestrator.process_note(data)
        
        if result.success:
            return jsonify({
                'success': True,
                'note_id': result.note_id,
                'classification': result.classification.to_dict(),
                'processing_time': result.processing_time
            }), 201
        else:
            return jsonify({
                'success': False,
                'errors': result.errors
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notes/<note_id>', methods=['GET'])
def get_note(note_id):
    try:
        from storage_engine import StorageEngine
        storage = StorageEngine()
        note = storage.get_note(note_id)
        
        if note:
            return jsonify(note.to_dict()), 200
        else:
            return jsonify({'error': 'Note not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 8. Database Integration

#### SQLite Integration Example
```python
import sqlite3
from quick_add import QuickCaptureOrchestrator

class DatabaseIntegration:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.orchestrator = QuickCaptureOrchestrator()
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    title TEXT,
                    category TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def add_note_from_database(self, content: str, title: str = None):
        """Add note and store in database"""
        # Process with QuickCapture
        result = self.orchestrator.process_note({
            'content': content,
            'title': title
        })
        
        if result.success:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO notes (id, content, title, category, tags)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    result.note_id,
                    content,
                    title,
                    result.classification.primary_category,
                    ','.join(result.classification.tags)
                ))
            
            return result.note_id
        else:
            raise Exception(f"Processing failed: {result.errors}")
    
    def get_notes_by_category(self, category: str):
        """Retrieve notes by category"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM notes WHERE category = ?
                ORDER BY created_at DESC
            """, (category,))
            return cursor.fetchall()

# Usage
db_integration = DatabaseIntegration("notes.db")

# Add notes
note_id = db_integration.add_note_from_database(
    "Research findings on machine learning algorithms",
    "ML Research"
)

# Retrieve notes
research_notes = db_integration.get_notes_by_category("research")
for note in research_notes:
    print(f"ID: {note[0]}, Title: {note[2]}")
```

## Error Handling Examples

### 9. Error Handling and Recovery

#### Robust Error Handling
```python
from quick_add import QuickCaptureOrchestrator
from exceptions import ProcessingError, ValidationError

def robust_note_processing(content: str, max_retries: int = 3):
    """Process note with error handling and retries"""
    orchestrator = QuickCaptureOrchestrator()
    
    for attempt in range(max_retries):
        try:
            result = orchestrator.process_note({
                'content': content,
                'retry_attempt': attempt + 1
            })
            
            if result.success:
                return result
            else:
                print(f"Processing failed (attempt {attempt + 1}): {result.errors}")
                
        except ValidationError as e:
            print(f"Validation error (attempt {attempt + 1}): {e}")
            # Try to fix validation issues
            content = fix_validation_issues(content)
            
        except ProcessingError as e:
            print(f"Processing error (attempt {attempt + 1}): {e}")
            # Wait before retry
            time.sleep(2 ** attempt)
            
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}): {e}")
            break
    
    raise Exception(f"Failed to process note after {max_retries} attempts")

def fix_validation_issues(content: str) -> str:
    """Attempt to fix common validation issues"""
    # Remove HTML tags
    import re
    content = re.sub(r'<[^>]+>', '', content)
    
    # Truncate if too long
    if len(content) > 10000:
        content = content[:10000] + "..."
    
    # Ensure minimum length
    if len(content) < 10:
        content = content + " [Content too short]"
    
    return content

# Usage
try:
    result = robust_note_processing("Your note content here")
    print(f"Success: {result.note_id}")
except Exception as e:
    print(f"Failed: {e}")
```

### 10. Monitoring and Logging

#### Comprehensive Logging
```python
import logging
from quick_add import QuickCaptureOrchestrator
from observability import MetricsCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quickcapture.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
metrics = MetricsCollector()

def monitored_note_processing(content: str):
    """Process note with comprehensive monitoring"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting note processing: {len(content)} characters")
        
        orchestrator = QuickCaptureOrchestrator()
        result = orchestrator.process_note({'content': content})
        
        processing_time = time.time() - start_time
        
        if result.success:
            logger.info(f"Note processed successfully: {result.note_id}")
            metrics.record_success(processing_time, result.classification.primary_category)
        else:
            logger.error(f"Processing failed: {result.errors}")
            metrics.record_failure(processing_time, result.errors)
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception(f"Unexpected error during processing: {e}")
        metrics.record_error(processing_time, str(e))
        raise

# Usage
try:
    result = monitored_note_processing("Your note content here")
    print(f"Processing completed: {result.note_id}")
except Exception as e:
    print(f"Processing failed: {e}")
```

## Performance Examples

### 11. Batch Processing with Progress Tracking

#### Efficient Batch Processing
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from quick_add import QuickCaptureOrchestrator

def batch_process_with_progress(contents: List[str], max_workers: int = 4):
    """Process multiple notes with progress tracking"""
    orchestrator = QuickCaptureOrchestrator()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_content = {
            executor.submit(orchestrator.process_note, {'content': content}): content
            for content in contents
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(contents), desc="Processing notes") as pbar:
            for future in as_completed(future_to_content):
                content = future_to_content[future]
                
                try:
                    result = future.result()
                    results.append({
                        'content': content[:50] + "...",
                        'success': result.success,
                        'note_id': result.note_id if result.success else None,
                        'errors': result.errors if not result.success else []
                    })
                except Exception as e:
                    results.append({
                        'content': content[:50] + "...",
                        'success': False,
                        'errors': [str(e)]
                    })
                
                pbar.update(1)
    
    return results

# Usage
contents = [
    "First note content",
    "Second note content",
    "Third note content",
    # ... more contents
]

results = batch_process_with_progress(contents)

# Summary
successful = sum(1 for r in results if r['success'])
print(f"Processed {len(results)} notes: {successful} successful, {len(results) - successful} failed")
```

These examples demonstrate the flexibility and power of QuickCapture for various use cases, from simple note-taking to complex batch processing and system integration scenarios. 
noteId: "87ad5ef064c011f0970d05fa391d7ad1"
tags: []

---

 