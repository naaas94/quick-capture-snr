# Core Data Structures

## Overview

QuickCapture uses a well-defined set of data structures to represent notes, metadata, and system state. These structures ensure consistency, type safety, and efficient data processing throughout the system.

## Primary Data Models

### Note Structure

The core data structure representing a note in the system:

```python
class Note:
    note_id: str              # Unique identifier
    content: str              # Main note content
    title: Optional[str]      # Note title (optional)
    tags: List[str]           # Associated tags
    metadata: Dict[str, Any]  # Additional metadata
    created_at: datetime      # Creation timestamp
    updated_at: datetime      # Last update timestamp
    embedding: Optional[List[float]]  # Semantic embedding vector
    category: Optional[str]   # Assigned category
    confidence_score: float   # Classification confidence
    source: Optional[str]     # Source of the note
    quality_score: float      # Content quality assessment
```

### Metadata Structure

Extended information about notes:

```python
class NoteMetadata:
    author: Optional[str]     # Note author
    source_type: str          # Type of source (file, api, manual)
    file_path: Optional[str]  # Original file location
    processing_time: float    # Processing duration
    validation_status: str    # Validation result
    error_messages: List[str] # Any processing errors
    word_count: int           # Content length
    language: Optional[str]   # Detected language
    sentiment: Optional[str]  # Sentiment analysis result
    entities: List[str]       # Named entities found
```

## Configuration Data Structures

### Grammar Rules Configuration

```yaml
grammar_rules:
  validation_rules:
    - rule_name: "content_length"
      min_length: 10
      max_length: 10000
    - rule_name: "required_fields"
      fields: ["content", "note_id"]
  
  content_rules:
    - rule_name: "no_empty_content"
      pattern: "^\\s*$"
      action: "reject"
    - rule_name: "html_stripping"
      pattern: "<[^>]*>"
      action: "clean"
```

### Semantic Validation Configuration

```yaml
semantic_validation:
  embedding_config:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    dimension: 384
    similarity_threshold: 0.8
  
  classification_config:
    categories: ["research", "personal", "work", "learning"]
    confidence_threshold: 0.7
    fallback_category: "general"
```

### Storage Configuration

```yaml
storage_config:
  vector_store:
    type: "chroma"
    path: "./storage/vector_store"
    collection_name: "notes"
  
  file_storage:
    base_path: "./storage/notes"
    backup_enabled: true
    compression: true
  
  metadata_store:
    type: "sqlite"
    path: "./storage/metadata.db"
```

## Processing Data Structures

### Processing Pipeline State

```python
class ProcessingState:
    note_id: str
    stage: str                # Current processing stage
    status: str               # Success, error, pending
    start_time: datetime
    end_time: Optional[datetime]
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
```

### Validation Results

```python
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    score: float              # Overall validation score
    details: Dict[str, Any]   # Detailed validation info

class ValidationError:
    field: str
    message: str
    severity: str             # error, warning, info
    code: str                 # Error code for programmatic handling
```

### Classification Results

```python
class ClassificationResult:
    primary_category: str
    confidence_score: float
    alternative_categories: List[Tuple[str, float]]
    tags: List[str]
    reasoning: str            # Explanation for classification
    model_version: str        # Model used for classification
```

## Storage Data Structures

### Vector Store Entry

```python
class VectorEntry:
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    source_note_id: str
```

### File Storage Entry

```python
class FileEntry:
    file_path: str
    file_name: str
    file_size: int
    content_hash: str
    created_at: datetime
    modified_at: datetime
    metadata: Dict[str, Any]
```

## Observability Data Structures

### Metrics Data

```python
class MetricsData:
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]
    source: str
```

### Health Status

```python
class HealthStatus:
    component: str
    status: str               # healthy, degraded, unhealthy
    last_check: datetime
    response_time: float
    error_count: int
    details: Dict[str, Any]
```

### Performance Metrics

```python
class PerformanceMetrics:
    processing_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    error_rate: float
    timestamp: datetime
```

## API Data Structures

### Request/Response Models

```python
class NoteRequest:
    content: str
    title: Optional[str]
    tags: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]

class NoteResponse:
    note_id: str
    status: str
    message: str
    note: Optional[Note]
    processing_time: float
    errors: List[str]
```

### Batch Processing

```python
class BatchRequest:
    notes: List[NoteRequest]
    batch_id: str
    priority: str             # high, normal, low
    callback_url: Optional[str]

class BatchResponse:
    batch_id: str
    total_notes: int
    successful: int
    failed: int
    results: List[NoteResponse]
    processing_time: float
```

## Data Validation Rules

### Content Validation

- Minimum content length: 10 characters
- Maximum content length: 10,000 characters
- Required fields: content, note_id
- Prohibited content: HTML tags, script tags
- Language detection: Automatic language identification

### Metadata Validation

- Timestamp format: ISO 8601
- UUID format for note_id
- Tag format: alphanumeric with hyphens
- Category validation: Must be from predefined list
- Confidence score: 0.0 to 1.0 range

### Storage Validation

- File size limits: Configurable per storage type
- Path validation: Secure path construction
- Hash verification: Content integrity checks
- Backup validation: Backup integrity verification

## Data Serialization

### JSON Format

```json
{
  "note_id": "uuid-string",
  "content": "Note content here",
  "title": "Optional title",
  "tags": ["tag1", "tag2"],
  "metadata": {
    "author": "user",
    "source": "manual",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "embedding": [0.1, 0.2, 0.3, ...],
  "category": "research",
  "confidence_score": 0.85
}
```

### YAML Format

```yaml
note_id: uuid-string
content: Note content here
title: Optional title
tags:
  - tag1
  - tag2
metadata:
  author: user
  source: manual
  created_at: 2024-01-01T00:00:00Z
embedding: [0.1, 0.2, 0.3]
category: research
confidence_score: 0.85
```

## Data Migration and Versioning

### Version Control

- Schema version tracking
- Backward compatibility support
- Migration scripts for data updates
- Version validation on startup

### Data Migration

- Automatic migration detection
- Incremental migration support
- Rollback capabilities
- Migration logging and monitoring 
noteId: "c229dd2064bf11f0970d05fa391d7ad1"
tags: []

---

 