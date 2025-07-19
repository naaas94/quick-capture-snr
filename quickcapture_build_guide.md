# QUICKCAPTURE — BUILD GUIDE FOR AI AGENT (STAGE-BY-STAGE)

## OVERVIEW

This document is a stage-by-stage technical execution plan for an AI development agent (e.g., via Cursor or code assistant) to implement the **QuickCapture** system — a CLI-based symbolic ingestion layer designed for structured note capture with downstream compatibility for semantic systems such as the Semantic Note Router (SNR).

The goal is to build a production-ready system with modular architecture, semantic validation, intelligent tagging, distributed logging, and clean interface for downstream ML use.

---

## STAGE 0 — PROJECT SCAFFOLDING

### Objective

Create a clean, modular project structure with well-defined directories, config placeholders, and testable components.

### Actions

1. Create root directory: `snr-quickcapture/`
2. Subdirectories:
   - `scripts/`: all logic scripts
   - `storage/`: SQLite database and vector store
   - `tests/`: unit and integration test cases
   - `config/`: grammar rules, tag files, and metadata
   - `observability/`: metrics, logging, and monitoring
3. Add project files:
   - `README.md`
   - `requirements.txt`
   - `.env.example`
   - `pyproject.toml` (for modern Python packaging)
4. Create main entrypoint script: `scripts/quick_add.py`

---

## STAGE 1 — ENHANCED INPUT PARSING MODULE

### Objective

Parse the user input string into structured components with semantic preprocessing and intelligent tag extraction.

### Actions

1. Create `scripts/parse_input.py`
2. Define function `parse_note_input(text: str) -> dict` with enhanced output:
   ```python
   {
     "tags": List[str],
     "note": str,
     "comment": Optional[str],
     "raw_text": str,
     "semantic_density": float,
     "content_type": ContentType,
     "confidence_score": float
   }
   ```
3. Expected input grammar: `tag1, tag2: note body : optional comment`
4. Enhanced edge case handling:
   - Multiple colons (only split on the first two)
   - Strip whitespace and normalize
   - Normalize lowercase for tags
   - Remove trailing commas or empty tags
   - Semantic density calculation
   - Content type classification
5. Add semantic preprocessing:
   - Stopword ratio analysis
   - Token diversity calculation
   - Semantic coherence scoring
6. Raise structured exceptions for critical parsing failures

---

## STAGE 2 — SEMANTIC VALIDATION ENGINE

### Objective

Validate the parsed input against both structural and semantic constraints with intelligent quality scoring.

### Actions

1. Create `scripts/validate_note.py`
2. Define function `validate_note(parsed: dict) -> dict`
   ```python
   {
     "valid": bool,
     "issues": List[str],
     "semantic_coherence_score": float,
     "tag_quality_score": float,
     "overall_confidence": float,
     "validation_details": Dict[str, Any]
   }
   ```
3. Enhanced validation rules:
   - At least one tag required
   - Note body must be non-empty
   - Semantic density threshold (configurable)
   - Tag quality scoring
   - Content type validation
   - Co-occurrence pattern analysis
4. Add semantic validation:
   - Minimum semantic coherence score
   - Maximum stopword ratio
   - Content length validation
   - Pattern recognition for common issues
5. Validation should not raise — only flag with detailed scoring
6. Output must be mergeable with parsed dictionary

---

## STAGE 3 — ENHANCED METADATA ENRICHMENT

### Objective

Wrap parsed and validated input into a comprehensive `ParsedNote` object with rich metadata and semantic intelligence.

### Actions

1. Create `scripts/models.py`
2. Define enhanced `ParsedNote` as a `@dataclass` with fields:
   ```python
   note_id: str  # UUID4
   tags: List[str]
   note: str
   comment: Optional[str]
   timestamp: str  # ISO format
   valid: bool
   issues: List[str]
   origin: str  # Always "quickcapture"
   version: int  # Initial version = 1
   raw_text: str
   semantic_density: float
   tag_quality_score: float
   content_type: ContentType
   confidence_score: float
   tag_hierarchy: Dict[str, List[str]]
   co_occurrence_patterns: Dict[str, float]
   embedding_vector: Optional[List[float]]
   snr_metadata: Dict[str, Any] = field(default_factory=dict)
   ```
3. Implement enhanced methods:
   - `add_snr_metadata()`
   - `get_snr_optimized_body()`
   - `calculate_semantic_coherence()`
   - `get_tag_hierarchy()`
   - `update_confidence_score()`
4. Generate UUID and timestamp upon instantiation
5. Prepare for JSON serializability and vector storage

---

## STAGE 4 — INTELLIGENT STORAGE ENGINE

### Objective

Implement hybrid storage with SQLite primary storage, vector store for semantic search, and JSONL backup.

### Actions

1. Create `scripts/storage_engine.py`
2. Define `StorageEngine` class with methods:
   ```python
   class StorageEngine:
       def store_note(self, note: ParsedNote) -> bool
       def retrieve_notes_by_tag(self, tag: str) -> List[ParsedNote]
       def search_semantic(self, query: str, limit: int) -> List[ParsedNote]
       def get_tag_statistics(self) -> Dict[str, int]
       def backup_to_jsonl(self) -> bool
   ```
3. SQLite schema with proper indexing:
   ```sql
   CREATE TABLE notes (
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
       snr_metadata TEXT  -- JSON
   );
   
   CREATE INDEX idx_tags ON notes(tags);
   CREATE INDEX idx_timestamp ON notes(timestamp);
   CREATE INDEX idx_semantic_density ON notes(semantic_density);
   ```
4. Vector store integration for semantic search
5. Atomic write operations with proper error handling
6. Create `scripts/quick_add.py` to orchestrate full flow

---

## STAGE 5 — TAG INTELLIGENCE SYSTEM

### Objective

Implement intelligent tag management with suggestion, drift detection, and quality scoring.

### Actions

1. Create `scripts/tag_intelligence.py`
2. Define `TagIntelligence` class:
   ```python
   class TagIntelligence:
       def suggest_tags(self, note_body: str) -> List[TagSuggestion]
       def detect_tag_drift(self) -> TagDriftReport
       def calculate_tag_quality(self, tag: str) -> float
       def get_tag_hierarchy(self) -> Dict[str, List[str]]
       def suggest_tag_consolidation(self) -> List[TagConsolidation]
   ```
3. Tag suggestion using embeddings:
   - Semantic similarity to existing tags
   - Co-occurrence pattern analysis
   - Tag hierarchy alignment
4. Drift detection:
   - Monitor tag usage patterns over time
   - Detect emerging vs dying tags
   - Identify tag proliferation issues
5. Quality scoring:
   - Tag relevance to note content
   - Tag specificity vs generality
   - Tag consistency across similar notes

---

## STAGE 6 — ENHANCED OUTLIER REVIEW INTERFACE

### Objective

Enable intelligent inspection and correction of malformed notes with semantic assistance.

### Actions

1. Create `scripts/review_outliers.py`
2. CLI Arguments:
   - `--tag <tag>`: target file
   - `--limit N`: number of notes to display
   - `--edit`: optional interactive fix mode
   - `--semantic`: enable semantic suggestions
   - `--auto-fix`: attempt automatic corrections
3. Enhanced functionality:
   - Load from SQLite with semantic filtering
   - Filter by validation issues and confidence scores
   - Display semantic coherence scores and tag quality
   - Provide intelligent suggestions for improvements
   - On edit: allow manual correction with semantic assistance
   - Preserve note_id and log corrected note with version bump
   - Track correction patterns for learning

---

## STAGE 7 — COMPREHENSIVE OBSERVABILITY

### Objective

Implement production-grade monitoring, metrics, and observability for system health and performance.

### Actions

1. Create `observability/` directory with modules:
   - `metrics_collector.py`
   - `health_monitor.py`
   - `performance_tracker.py`
   - `drift_detector.py`
2. Define `QuickCaptureMetrics` class:
   ```python
   class QuickCaptureMetrics:
       ingestion_rate: float
       validation_success_rate: float
       semantic_coherence_avg: float
       tag_quality_avg: float
       processing_latency_avg: float
       storage_performance: Dict[str, float]
       error_distribution: Dict[str, int]
   ```
3. Real-time monitoring:
   - Ingestion rate tracking
   - Validation pattern analysis
   - Performance bottleneck detection
   - Semantic drift monitoring
4. Health reporting:
   - System health dashboard
   - Data quality metrics
   - Performance indicators
   - Alert system for anomalies

---

## STAGE 8 — SNR ALIGNMENT MODULES

### Objective

Prepare QuickCapture outputs for downstream processing with enhanced semantic optimization.

### Actions

1. Create `scripts/snr_preprocess.py`
2. Enhanced function `optimize_text_for_embeddings(text: str) -> str`:
   - Strip problematic characters (`&`, `<`, `>`)
   - Normalize whitespace and punctuation
   - Preserve semantic structure
   - Apply content-specific optimizations
3. Implement enhanced batch processor `process_batch_for_snr(notes: List[str]) -> List[ParsedNote]`:
   - Apply all preprocessing and metadata enrichment
   - Add semantic density scoring
   - Calculate embedding quality estimates
   - Generate tag hierarchy alignments
   - Return structured list for downstream vector storage
4. Ensure all notes include comprehensive `snr_metadata`:
   - Semantic coherence scores
   - Tag quality metrics
   - Content type classification
   - Processing confidence levels

---

## STAGE 9 — TESTING & QA

### Objective

Validate system functionality with comprehensive unit, integration, and performance tests.

### Actions

1. Create test files in `tests/`:
   - `test_parser.py`
   - `test_validator.py`
   - `test_storage.py`
   - `test_tag_intelligence.py`
   - `test_observability.py`
   - `test_outlier_review.py`
2. Enhanced test scenarios:
   - Multiple malformed inputs with semantic validation
   - Full end-to-end CLI run with metrics
   - Storage performance and atomicity tests
   - Tag intelligence and drift detection
   - Observability and monitoring accuracy
   - Metadata completeness and quality
3. Add `run_tests.sh` script for automation
4. Performance benchmarks:
   - Ingestion latency targets (<100ms)
   - Storage throughput testing
   - Semantic search performance
   - Memory usage profiling

---

## STAGE 10 — CONFIGURATION AND DEPLOYMENT

### Objective

Ensure reproducibility, configurability, and production readiness.

### Actions

1. Create `requirements.txt`:
   - Python >= 3.11
   - `rich`, `uuid`, `argparse`, `pyyaml`
   - `sqlite3`, `numpy`, `scikit-learn`
   - `sentence-transformers` (for embeddings)
   - `prometheus-client` (for metrics)
2. Add comprehensive YAML config files under `config/`:
   - `grammar_rules.yaml`: validation rule toggles
   - `semantic_validation.yaml`: semantic thresholds
   - `tag_intelligence.yaml`: tag management rules
   - `storage_config.yaml`: database and vector store settings
   - `observability.yaml`: monitoring configuration
3. Environment configuration:
   - `.env.example` for system-level parameters
   - Database connection settings
   - Vector store configuration
   - Monitoring endpoints
4. Prepare comprehensive README with:
   - Full CLI usage and examples
   - Architecture description
   - Configuration guide
   - Performance tuning tips
   - Troubleshooting guide

---

## FINAL OUTPUT

By the end of this enhanced build guide, the system will be able to:

- Capture structured notes from the CLI with semantic intelligence
- Parse and validate both structural and semantic coherence
- Store entries in hybrid SQLite/vector store with atomic operations
- Provide intelligent tag suggestions and drift detection
- Review and revise outliers with semantic assistance
- Monitor system health and performance comprehensively
- Prepare all notes for semantic routing by SNR with quality metrics
- Provide complete traceability, fault tolerance, and downstream compatibility

This enhanced plan reflects production-grade ML engineering practices, comprehensive epistemic awareness, and infrastructure-level modularity suitable for high-volume semantic pipelines.

---

Author: Alejandro Garay\
Project: QuickCapture (Enhanced Symbolic Ingestion Layer)\
Context: SNR Ecosystem / Semantic Note Infrastructure / Cognitive Architecture Systems

