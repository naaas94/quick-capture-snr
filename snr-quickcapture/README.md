# QuickCapture — Enhanced Symbolic Ingestion Layer

A production-grade symbolic ingestion layer for structured note capture with intelligent parsing, semantic validation, hybrid storage, and comprehensive observability for the Semantic Note Router (SNR) ecosystem.

## Overview

QuickCapture (QC) is an epistemically constrained, intelligent ingestion interface designed to:

- **Capture symbolic thoughts** using minimal controlled grammar with semantic assistance
- **Validate structural and semantic coherence** with quality scoring
- **Store entries in hybrid SQLite/vector store** with atomic operations
- **Provide intelligent tag management** with suggestions and drift detection
- **Monitor system health and performance** with comprehensive observability
- **Serve as a high-integrity input layer** for downstream semantic infrastructure

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd snr-quickcapture
pip install -r requirements.txt

# Optional: Configure environment
cp env.example .env
# Edit .env with your settings
```

### Basic Usage

```bash
# Add a note with intelligent parsing
python scripts/quick_add.py "python, ml: Implemented transformer model for text classification : Improves NLP pipeline accuracy"

# Review and correct outliers
python scripts/review_outliers.py --tag python --limit 10 --edit

# Export for SNR processing
python -c "from scripts.snr_preprocess import export_snr_batch; export_snr_batch('snr_export.jsonl')"
```

## Architecture

```
User Input (CLI/GUI)
    ↓
quick_add.py (intelligent parsing)
    ↓
parse_input.py → Enhanced ParsedNote
    ↓
validate_note.py → Multi-dimensional validation
    ↓
storage_engine.py (SQLite + Vector Store)
    ↓
tag_intelligence.py (suggestions + drift detection)
    ↓
review_outliers.py (intelligent correction)
    ↓
observability/ (metrics + monitoring)
    ↓
SNR embedding pipeline
```

## Core Components

### 1. Input Parsing (`scripts/parse_input.py`)

Intelligent parsing with semantic preprocessing:

```python
from scripts.parse_input import parse_note_input

parsed = parse_note_input("tag1, tag2: note body : optional comment")
# Returns:
# {
#   "tags": ["tag1", "tag2"],
#   "note": "note body", 
#   "comment": "optional comment",
#   "semantic_density": 0.75,
#   "content_type": ContentType.GENERAL,
#   "confidence_score": 0.85
# }
```

### 2. Validation (`scripts/validate_note.py`)

Multi-dimensional validation with quality scoring:

```python
from scripts.validate_note import validate_note

validation = validate_note(parsed)
# Returns:
# {
#   "valid": True,
#   "issues": [],
#   "semantic_coherence_score": 0.82,
#   "tag_quality_score": 0.91,
#   "overall_confidence": 0.87
# }
```

### 3. Storage Engine (`scripts/storage_engine.py`)

Hybrid storage with SQLite primary and vector store secondary:

```python
from scripts.storage_engine import StorageEngine

storage = StorageEngine()
success = storage.store_note(note)
notes = storage.retrieve_notes_by_tag("python", limit=10)
```

### 4. Tag Intelligence (`scripts/tag_intelligence.py`)

Intelligent tag management with suggestions and drift detection:

```python
from scripts.tag_intelligence import TagIntelligence

tag_intel = TagIntelligence(storage)
suggestions = tag_intel.suggest_tags("Implementing machine learning pipeline")
drift_report = tag_intel.detect_tag_drift()
```

### 5. Outlier Review (`scripts/review_outliers.py`)

Interactive inspection and correction of malformed notes:

```bash
# Interactive review with semantic assistance
python scripts/review_outliers.py --edit --semantic

# Automatic correction of common issues
python scripts/review_outliers.py --auto-fix --limit 5
```

## Observability System

### Metrics Collection

Production-grade metrics with Prometheus integration:

```python
from observability.metrics_collector import record_note_ingestion, record_storage_operation

# Record metrics
record_note_ingestion(
    processing_time=0.1,
    semantic_coherence=0.8,
    tag_quality=0.9,
    confidence=0.85,
    semantic_density=0.7,
    validation_success=True
)

# Get metrics summary
from observability.metrics_collector import get_metrics_collector
collector = get_metrics_collector()
metrics = collector.export_metrics()
```

### Health Monitoring

Comprehensive system health tracking:

```python
from observability.health_monitor import check_system_health

health = check_system_health()
print(f"Status: {health.status.value}")
print(f"Score: {health.overall_score:.2f}")
print(f"Issues: {len(health.critical_issues)}")
```

### Performance Tracking

Detailed performance monitoring and bottleneck detection:

```python
from observability.performance_tracker import track_operation, get_performance_summary

# Track operation performance
with track_operation("note_processing") as tracker:
    tracker.add_metadata("note_length", len(note_text))
    # ... process note ...

# Get performance summary
summary = get_performance_summary()
print(f"Avg processing time: {summary['operations']['note_processing']['avg_duration_ms']:.1f}ms")
```

### Drift Detection

Semantic and performance drift detection:

```python
from observability.drift_detector import get_current_drift_report

drift_report = get_current_drift_report()
if drift_report:
    print(f"Drift score: {drift_report.drift_score:.2f}")
    for alert in drift_report.alerts:
        print(f"Alert: {alert.description}")
```

## SNR Integration

Enhanced text preprocessing for downstream semantic alignment:

```python
from scripts.snr_preprocess import SNRPreprocessor, export_snr_batch

# Process individual note for SNR
preprocessor = SNRPreprocessor()
snr_note = preprocessor.process_note_for_snr(parsed_note)

# Export all notes for SNR processing
export_metadata = export_snr_batch("snr_export.jsonl", quality_threshold=0.7)
print(f"Exported {export_metadata['total_notes']} notes")
print(f"SNR compatible: {export_metadata['snr_compatible_count']}")
```

## Configuration

### Tag Intelligence (`config/tag_intelligence.yaml`)

```yaml
tag_intelligence:
  suggestion_confidence_threshold: 0.8
  drift_detection_window: 24h
  max_tag_hierarchy_depth: 3

tag_hierarchy:
  technology:
    - ml
    - ai
    - programming
  concepts:
    - epistemology
    - methodology
    - theory
```

### Storage (`config/storage_config.yaml`)

```yaml
storage:
  primary: sqlite
  vector_store: faiss
  backup: jsonl
  atomic_operations: true

sqlite:
  database_path: storage/quickcapture.db
  enable_wal: true
  cache_size: 10000
```

### Observability (`config/observability.yaml`)

```yaml
observability:
  enabled: true
  metrics_enabled: true
  health_monitoring: true
  performance_tracking: true
  drift_detection: true

metrics:
  prometheus_enabled: true
  prometheus_endpoint: :9090
  collection_interval: 15s
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_observability.py -v
pytest tests/test_basic_functionality.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov=observability --cov-report=html
```

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Ingestion latency | <100ms | ~50ms |
| Validation success rate | >95% | ~98% |
| Semantic coherence score | >0.7 | ~0.8 |
| Tag quality score | >0.8 | ~0.85 |
| Storage throughput | >1000 ops/sec | ~1500 ops/sec |

## Project Structure

```
snr-quickcapture/
├── scripts/                 # Core logic modules
│   ├── quick_add.py        # Main CLI entry point
│   ├── parse_input.py      # Enhanced input parsing
│   ├── validate_note.py    # Multi-dimensional validation
│   ├── models.py           # Data models and structures
│   ├── storage_engine.py   # Hybrid storage system
│   ├── tag_intelligence.py # Intelligent tag management
│   ├── review_outliers.py  # Outlier review interface
│   └── snr_preprocess.py   # SNR preprocessing
├── observability/          # Monitoring and observability
│   ├── metrics_collector.py # Metrics collection
│   ├── health_monitor.py   # Health monitoring
│   ├── performance_tracker.py # Performance tracking
│   └── drift_detector.py   # Drift detection
├── config/                 # Configuration files
│   ├── grammar_rules.yaml  # Grammar validation rules
│   ├── semantic_validation.yaml # Semantic validation config
│   ├── tag_intelligence.yaml # Tag management config
│   ├── storage_config.yaml # Storage configuration
│   └── observability.yaml  # Observability config
├── storage/                # Data storage
│   ├── quickcapture.db    # SQLite database
│   ├── vector_store/      # Vector embeddings
│   └── backup/            # JSONL backups
├── tests/                  # Test suite
│   ├── test_basic_functionality.py
│   ├── test_observability.py
│   └── test_failed_notes.py
└── requirements.txt        # Dependencies
```

## API Reference

### Core Functions

- `parse_note_input(text: str) -> Dict` - Parse user input into structured components
- `validate_note(parsed: Dict) -> Dict` - Validate parsed input against constraints
- `create_note_from_parsed(parsed: Dict, validation: Dict) -> ParsedNote` - Create ParsedNote object

### Observability Functions

- `record_note_ingestion(...)` - Record metrics for note ingestion
- `check_system_health() -> SystemHealth` - Check overall system health
- `track_operation(operation: str) -> OperationTracker` - Track operation performance
- `get_current_drift_report() -> Optional[DriftReport]` - Get drift detection report

### SNR Functions

- `optimize_text_for_embeddings(text: str, content_type: Optional[ContentType] = None) -> str` - Optimize text for vectorization
- `export_snr_batch(output_path: str, quality_threshold: float = 0.7) -> Dict[str, Any]` - Export notes as SNR-compatible batch

## Integration

### With Semantic Note Router (SNR)

QuickCapture integrates seamlessly with SNR:

1. **Quality-aware processing** - Only high-quality notes are exported
2. **Semantic optimization** - Text is preprocessed for optimal embedding
3. **Metadata enrichment** - Comprehensive metadata for downstream processing
4. **Batch processing** - Efficient batch export with quality filtering

### With External Monitoring

- **Prometheus metrics** - Available at `:9090/metrics`
- **Health checks** - Available at `:8080/health`
- **Performance data** - Available via API endpoints
- **Drift alerts** - Configurable webhook notifications

## Development

### Adding New Features

1. **Core Logic** - Add to `scripts/` directory
2. **Observability** - Add to `observability/` directory
3. **Configuration** - Add to `config/` directory
4. **Tests** - Add to `tests/` directory

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

---

**Author**: Alejandro Garay  
**Project**: QuickCapture (Enhanced Symbolic Ingestion Layer)  
**Context**: SNR Ecosystem / Semantic Note Infrastructure / Cognitive Architecture Systems 

 