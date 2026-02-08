"""
Configuration system for RAG evaluation.

All configuration is done via dataclasses with type hints and validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Set, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .chunk_schema import Content_type


class Mode(str, Enum):
    """Evaluation mode."""

    NO_RAG = "no_rag"          # LLM-only baseline
    RAG = "rag"                # Retrieval + LLM


class RetrievalMode(str, Enum):
    """Retrieval strategy."""

    DENSE = "dense"            # FAISS only
    BM25 = "bm25"              # BM25 only
    HYBRID = "hybrid"          # Dense + BM25 fusion


class RerankerType(str, Enum):
    """Reranker model types."""

    CROSS_ENCODER = "cross_encoder"
    NONE = "none"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    # Model selection
    model_name: str = "BAAI/bge-large-en-v1.5"
    backend: str = "sentence_transformers"  # sentence_transformers | ollama | medembed
    device: str = "cuda"  # cuda | cpu | auto

    # Ollama-specific (if backend == "ollama")
    ollama_base_url: Optional[str] = None

    # Encoding settings
    batch_size: int = 128
    normalize: bool = True

    # Ollama-specific (if backend == "ollama")
    ollama_base_url: Optional[str] = None

    # Instruction for BGE models
    use_instruction: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages:"

    def validate(self) -> None:
        """Validate configuration."""
        valid_backends = {"sentence_transformers", "ollama", "medembed"}
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. Choose from {valid_backends}")

        valid_devices = {"cuda", "cpu", "auto"}
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}. Choose from {valid_devices}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class QueryConfig:
    """Query generation and reformulation configuration."""

    # Base query
    use_base_query: bool = True

    # LLM-based reformulation
    use_reformulation: bool = False
    reformulation_model: str = "thewindmom/llama3-med42-8b:latest"
    reformulation_temperature: float = 0.0
    reformulation_max_tokens: int = 120
    reformulation_prompt_template: Optional[str] = None

    # HyDE
    use_hyde: bool = False
    hyde_mode: str = "question"  # question | option | both
    hyde_model: str = "thewindmom/llama3-med42-8b:latest"
    hyde_temperature: float = 0.2
    hyde_max_tokens: int = 120

    # Option-aware queries
    use_option_aware: bool = False

    # Dataset query expansions
    use_expansions: bool = False

    def validate(self) -> None:
        """Validate configuration."""
        if not any([self.use_base_query, self.use_reformulation, self.use_hyde, self.use_option_aware]):
            raise ValueError("At least one query variant must be enabled")

        valid_hyde_modes = {"question", "option", "both"}
        if self.hyde_mode not in valid_hyde_modes:
            raise ValueError(f"Invalid hyde_mode: {self.hyde_mode}. Choose from {valid_hyde_modes}")


@dataclass
class FusionConfig:
    """Fusion strategy configuration."""

    # RRF parameters
    use_rrf: bool = True
    rrf_k: int = 60

    # Scoring weights
    alpha_rrf: float = 0.6      # Weight for RRF score
    beta_overlap: float = 0.4   # Weight for lexical overlap
    bm25_weight: float = 0.0    # Weight for raw BM25 score (if available)
    dense_weight: float = 0.0   # Weight for raw dense score (if available)

    # Filtering
    min_final_threshold: float = 0.35
    min_lexical_overlap: float = 0.01

    # Deduplication
    dedupe_by_doc: bool = True
    dedupe_by_section: bool = False

    def validate(self) -> None:
        """Validate configuration."""
        total_weight = self.alpha_rrf + self.beta_overlap + self.bm25_weight + self.dense_weight
        if total_weight == 0:
            raise ValueError("At least one fusion weight must be non-zero")

        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {self.rrf_k}")


@dataclass
class RerankerConfig:
    """Reranker configuration."""

    # Model
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-large"
    model_type: RerankerType = RerankerType.CROSS_ENCODER

    # Reranking settings
    top_k: int = 150             # Rerank top-K candidates
    batch_size: int = 128
    max_length: int = 384
    use_fp16: bool = False

    # Gating
    min_score_threshold: Optional[float] = None  # Drop all if top score < threshold

    def validate(self) -> None:
        """Validate configuration."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class CoarseRetrievalConfig:
    """Coarse (section-level) retrieval configuration."""

    enabled: bool = False
    top_k_sections: int = 20

    # Retrieval mode for coarse stage
    mode: RetrievalMode = RetrievalMode.HYBRID

    # Top-k for each retriever
    dense_k: int = 100
    bm25_k: int = 200

    def validate(self) -> None:
        """Validate configuration."""
        if self.top_k_sections <= 0:
            raise ValueError(f"top_k_sections must be positive, got {self.top_k_sections}")


@dataclass
class FineRetrievalConfig:
    """Fine (chunk-level) retrieval configuration."""

    # Top-k for final results
    top_k: int = 12

    # Retrieval mode
    mode: RetrievalMode = RetrievalMode.HYBRID

    # Top-k for each retriever (before fusion)
    dense_k: int = 80
    bm25_k: int = 400

    # Max passages after fusion/reranking
    max_passages: int = 6

    # Evidence packing
    max_evidence_tokens: int = 1200

    def validate(self) -> None:
        """Validate configuration."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

        if self.max_passages > self.top_k:
            raise ValueError(f"max_passages ({self.max_passages}) cannot exceed top_k ({self.top_k})")


@dataclass
class RetrievalConfig:
    """Complete retrieval configuration."""

    # Evaluation mode
    mode: Mode = Mode.RAG

    # Stages
    coarse: CoarseRetrievalConfig = field(default_factory=CoarseRetrievalConfig)
    fine: FineRetrievalConfig = field(default_factory=FineRetrievalConfig)

    # Query generation
    queries: QueryConfig = field(default_factory=QueryConfig)

    # Fusion
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Reranking
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    # Metadata filtering
    content_type_filters: Set["Content_type"] = field(default_factory=set)
    prioritize_content_types: List["Content_type"] = field(default_factory=list)

    def validate(self) -> None:
        """Validate configuration."""
        self.coarse.validate()
        self.fine.validate()
        self.queries.validate()
        self.fusion.validate()
        self.reranker.validate()

        if self.mode == Mode.NO_RAG:
            # No retrieval settings needed
            return

        if self.coarse.enabled and self.coarse.top_k_sections < self.fine.top_k:
            raise ValueError(
                f"coarse.top_k_sections ({self.coarse.top_k_sections}) should be >= fine.top_k ({self.fine.top_k})"
            )


@dataclass
class LLMConfig:
    """LLM configuration for answer generation."""

    model_name: str = "thewindmom/llama3-med42-8b:latest"
    base_url: str = "http://172.25.208.1:11434"
    max_tokens: int = 2
    temperature: float = 0.0
    top_p: float = 1.0
    num_workers: int = 8

    # Chain-of-thought
    use_cot: bool = False
    cot_max_tokens: int = 512

    # Reasoning models (e.g., gpt-oss, deepseek-r1)
    reasoning_model: bool = False
    reasoning_max_tokens: int = 2048  # Allow more tokens for reasoning

    def validate(self) -> None:
        """Validate configuration."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")


@dataclass
class EvaluationConfig:
    """
    Complete evaluation configuration.

    This is the main configuration object passed to the evaluation orchestrator.
    """

    # Paths
    index_dir: Path = field(default_factory=lambda: Path("/home/ser/medrag/processed_corpus"))
    eval_data_path: Path = field(default_factory=lambda: Path("/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl"))
    output_dir: Path = field(default_factory=lambda: Path("/home/ser/medrag/evaluation_results"))

    # Retrieval
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Embeddings
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # LLM
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Tokenization (for evidence packing)
    tokenizer_name: str = "deepseek-ai/DeepSeek-R1"

    # Processing
    batch_size: int = 32
    retrieval_batch_size: int = 256
    limit: Optional[int] = None  # Limit number of examples (for testing)

    # Output
    save_predictions: bool = True
    save_retrieved_contents: bool = True
    stream_mode: bool = False  # Score each batch immediately

    # Logging
    log_level: str = "INFO"
    verbose: bool = False

    def validate(self) -> None:
        """Validate complete configuration."""
        self.retrieval.validate()
        self.embedding.validate()
        self.llm.validate()

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.retrieval_batch_size <= 0:
            raise ValueError(f"retrieval_batch_size must be positive, got {self.retrieval_batch_size}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        """Create configuration from dictionary (useful for loading from YAML/JSON)."""
        # This is a simplified version - in production, you'd want more robust parsing
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (useful for saving to YAML/JSON)."""
        from dataclasses import asdict

        return asdict(self)
