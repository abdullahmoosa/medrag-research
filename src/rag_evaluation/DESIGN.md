# Medical RAG Evaluation System - Design Document

## Overview

This document describes the design decisions and architecture of the Medical RAG Evaluation System, a production-grade framework for evaluating retrieval-augmented generation on medical question answering tasks.

## System Architecture

### High-Level Design

The system uses a **strategy pattern** to support two evaluation modes:
- **No-RAG Mode**: LLM-only baseline (no retrieval)
- **RAG Mode**: Full retrieval pipeline

Both modes share the same evaluation code path, differing only in context provision.

### Module Structure

```
src/rag_evaluation/
├── __init__.py              # Package exports
├── chunk_schema.py          # Immutable data models
├── config.py                # Configuration dataclasses
├── index_loader.py          # Index loading (BM25, FAISS, chunks)
├── embedding_client.py      # Pluggable embedding backends
├── query_reformulation.py   # LLM query reformulation
├── hyde.py                  # HyDE hypothetical passages
├── retrieval.py             # Multi-stage retrieval pipeline
├── fusion.py                # RRF and fusion strategies
├── evaluation.py            # Evaluation orchestrator (strategy pattern)
├── metrics.py               # Metrics calculation
├── cli.py                   # Command-line interface
├── examples.py              # Usage examples
├── tests.py                 # Unit tests
├── README.md                # Full documentation
└── QUICKSTART.md            # Quick start guide
```

## Design Decisions

### 1. Strategy Pattern for Evaluation Modes

**Decision**: Use strategy pattern with `ContextProvider` abstract base class.

**Rationale**:
- Clean separation between No-RAG and RAG modes
- Shared evaluation logic (LLM scoring, metrics)
- Easy to add new modes (e.g., RAG-only, IR-only)

**Implementation**:
```python
class ContextProvider(ABC):
    @abstractmethod
    def get_context(self, question: str, example: Dict) -> Optional[Context]:
        pass

class NoRAGContextProvider(ContextProvider):
    def get_context(self, question: str, example: Dict) -> Optional[Context]:
        return None

class RAGContextProvider(ContextProvider):
    def get_context(self, question: str, example: Dict) -> Optional[Context]:
        # Perform retrieval
        return context
```

### 2. Multi-Stage Retrieval

**Decision**: Separate coarse (section-level) and fine (chunk-level) retrieval.

**Rationale**:
- Reduces search space for fine-grained retrieval
- Allows different strategies per stage
- Improves performance on large corpora
- Matches medical textbook structure (sections → chunks)

**Implementation**:
```python
class RetrievalPipeline:
    def retrieve(self, queries: List[QueryVariant], question: str):
        # Stage 1: Coarse section-level retrieval (optional)
        section_filter = self._coarse_retrieval(queries)

        # Stage 2: Fine chunk-level retrieval
        results = self._fine_retrieval(queries, section_filter)

        # Stage 3: Reranking (optional)
        if self.config.reranker.enabled:
            results = self._rerank(results, question)

        return results
```

### 3. Query Variant System

**Decision**: Tagged query variants with independent retrieval contributions.

**Rationale**:
- Enables multi-query retrieval
- Tracks which queries contributed to results
- Allows per-query analysis and debugging
- Supports fusion across query types

**Implementation**:
```python
@dataclass(frozen=True)
class QueryVariant:
    query_id: str
    text: str
    kind: QueryKind  # BASE | REFORM | HYDE | OPTION | EXPANSION
    example_id: str
    metadata: Dict[str, Any]

# Each query variant is retrieved independently
# Results are fused using RRF
```

### 4. Immutable Data Models

**Decision**: Use frozen dataclasses for all data models.

**Rationale**:
- Prevents accidental modification
- Enables safe sharing between components
- Clear data flow
- Easy serialization

**Examples**:
```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    section_metadata: SectionMetadata
    # ...
```

### 5. Configuration-Driven Design

**Decision**: All parameters via dataclasses, no hardcoded values.

**Rationale**:
- Easy experimentation
- Reproducible results
- No code changes for different configs
- CLI and programmatic interfaces

**Implementation**:
```python
@dataclass
class EvaluationConfig:
    retrieval: RetrievalConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    # ...

# CLI
python -m src.rag_evaluation.cli --top-k 12 --use-reformulation

# Programmatic
config = EvaluationConfig()
config.retrieval.fine.top_k = 12
```

### 6. Pluggable Backends

**Decision**: Abstract base classes for embedders and retrievers.

**Rationale**:
- Easy to add new models (MedEmbed, custom models)
- Swap backends without code changes
- Testable components

**Implementation**:
```python
class EmbeddingBackend(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass

class SentenceTransformerBackend(EmbeddingBackend):
    # ...

class OllamaBackend(EmbeddingBackend):
    # ...
```

### 7. Type Safety

**Decision**: Strong typing throughout with type hints.

**Rationale**:
- Catch errors at development time
- Better IDE support
- Self-documenting code
- Easier refactoring

**Approach**:
- All functions have type hints
- Dataclasses for structured data
- Enums for fixed choices
- TYPE_CHECKING for forward references

### 8. Async I/O

**Decision**: Use asyncio for LLM calls and query generation.

**Rationale**:
- Efficient concurrent processing
- Non-blocking I/O for Ollama
- Better throughput

**Implementation**:
```python
class AsyncLLMClient:
    async def generate_batch(self, prompts: List[str]) -> List[str]:
        tasks = [self.generate(p) for p in prompts]
        return await asyncio.gather(*tasks)
```

### 9. Circular Import Prevention

**Decision**: Use TYPE_CHECKING and string annotations.

**Rationale**:
- Allows type hints without runtime imports
- Prevents circular dependencies
- Cleaner module structure

**Implementation**:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chunk_schema import Content_type

def filter_by_content_type(
    prioritize_types: List["Content_type"],  # String annotation
) -> None:
    pass
```

## Key Features

### 1. Evaluation Modes

**No-RAG (Baseline)**
- LLM receives question + options only
- Establishes upper bound for LLM capability
- Measures value of retrieval

**RAG**
- Full multi-stage retrieval
- Query variants (reformulation, HyDE, option-aware)
- Fusion and reranking
- Context injection

### 2. Query Variants

| Variant | Description | Use Case |
|---------|-------------|----------|
| Base | Original question | Baseline retrieval |
| Reform | LLM-rewritten query | Medical terminology optimization |
| HyDE | Hypothetical passage | Semantic bridge |
| Option | Per-option queries | Option-aware retrieval |
| Expansion | Dataset expansions | Provided variations |

### 3. Retrieval Stages

**Stage 1: Coarse (Optional)**
- Section-level retrieval
- Reduces search space
- Configurable top-k sections

**Stage 2: Fine**
- Chunk-level retrieval
- Dense, BM25, or Hybrid
- RRF fusion

**Stage 3: Reranking (Optional)**
- Cross-encoder reranker
- Top-K candidates
- Configurable gating

### 4. Metadata-Aware Filtering

- Filter by content_type (anatomy, physiology, etc.)
- Filter by medical_system (cardiovascular, etc.)
- Deduplication by doc_id or section_id
- Prioritization of specific types

### 5. Metrics

**Overall**
- Accuracy
- With/without context accuracy

**Per-Subject**
- Accuracy by medical subject
- Count per subject

**Per-Choice-Type**
- Single vs multiple choice

**Confusion Matrix**
- Gold vs predicted
- By answer option

**Coverage**
- Mean/min/max/median passages
- With/without context counts

## Extension Points

### Adding a New Retriever

```python
class CustomRetriever(Retriever):
    def retrieve(self, queries, top_k, filter_ids=None):
        # Your retrieval logic
        return results

# Register in RetrievalPipeline.__init__
if mode == "custom":
    self.retriever = CustomRetriever(...)
```

### Adding a New Query Variant

```python
# In QueryGenerator.generate_queries
if self.config.use_custom_query:
    queries.append(QueryVariant(
        query_id=f"{example_id}_custom",
        text=custom_query,
        kind=QueryKind.CUSTOM,  # Add to enum
        example_id=example_id,
    ))
```

### Adding a New Fusion Strategy

```python
class CustomFusion:
    def fuse(self, rank_lists, config):
        # Your fusion logic
        return fused_results

# Use in RetrievalPipeline
self.fusion = CustomFusion()
```

## Performance Considerations

### GPU Utilization
- Embeddings on GPU (configurable)
- Reranker FP16 mode
- Batch processing

### I/O Optimization
- Async LLM calls
- Batched embeddings
- Efficient index loading

### Memory Management
- Configurable batch sizes
- LRU cache for tokenization
- Streaming results to disk

## Testing Strategy

### Unit Tests
- Config validation
- Schema serialization
- Fusion algorithms
- Query sanitization

### Integration Tests
- Index loading
- End-to-end retrieval
- Evaluation pipeline

### Experiment Tests
- Ablation studies
- Mode comparisons
- Parameter sweeps

## Future Enhancements

### Planned Features
1. Distributed evaluation (multiple GPUs)
2. Result caching and incremental updates
3. Web UI for experiment tracking
4. Automatic hyperparameter tuning
5. More reranker options (MonoT5, etc.)
6. Query difficulty estimation
7. Error analysis tools

### Research Directions
1. Learned fusion weights
2. Dynamic top-k selection
3. Query routing (when to use which variant)
4. Multi-hop reasoning support
5. Confidence calibration

## Conclusion

This system provides a solid foundation for medical RAG research with:
- Clean modular architecture
- Flexible configuration
- Multiple evaluation modes
- Extensible design
- Production-ready code quality

The emphasis on type safety, immutability, and clear separation of concerns makes it suitable for serious research and production use.
