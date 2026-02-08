"""
Data models for chunks, queries, and retrieval results.

All data structures use strong typing and are immutable where possible.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any, Union
from datetime import datetime


class Content_type(str, Enum):
    """Medical content type classifications."""

    ANATOMY = "anatomy"
    PHYSIOLOGY = "physiology"
    PATHOLOGY = "pathology"
    PHARMACOLOGY = "pharmacology"
    MICROBIOLOGY = "microbiology"
    IMMUNOLOGY = "immunology"
    GENETICS = "genetics"
    EPIDEMIOLOGY = "epidemiology"
    CLINICAL = "clinical"
    DIAGNOSTICS = "diagnostics"
    THERAPEUTICS = "therapeutics"
    PROCEDURES = "procedures"
    MECHANISM = "mechanism"
    DEFINITION = "definition"
    OTHER = "other"


class MedicalSystem(str, Enum):
    """Medical system classifications."""

    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    NERVOUS = "nervous"
    GASTROINTESTINAL = "gastrointestinal"
    RENAL = "renal"
    ENDOCRINE = "endocrine"
    HEMATOLOGIC = "hematologic"
    MUSCULOSKELETAL = "musculoskeletal"
    INTEGUMENTARY = "integumentary"
    REPRODUCTIVE = "reproductive"
    IMMUNE = "immune"
    LYMPHATIC = "lymphatic"
    OTHER = "other"


class QueryKind(str, Enum):
    """Query variant types for multi-query retrieval."""

    BASE = "base"                    # Original question
    REFORM = "reform"                # LLM-reformulated query
    HYDE = "hyde"                    # HyDE hypothetical passage
    OPTION = "option"                # Option-aware query
    EXPANSION = "expansion"          # Dataset query expansion


@dataclass(frozen=True)
class SectionMetadata:
    """
    Metadata for a section (chapter/subsection) in a textbook.

    Attributes:
        textbook: Name of the textbook
        chapter: Chapter title/number
        section: Section title
        content_type: Primary medical content type
        medical_system: Medical system classification
        keywords: Extracted keywords
    """

    textbook: str
    chapter: str
    section: str
    content_type: Optional[Content_type] = None
    medical_system: Optional[MedicalSystem] = None
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "textbook": self.textbook,
            "chapter": self.chapter,
            "section": self.section,
            "content_type": self.content_type.value if self.content_type else None,
            "medical_system": self.medical_system.value if self.medical_system else None,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionMetadata":
        """Create from dictionary."""
        return cls(
            textbook=data["textbook"],
            chapter=data["chapter"],
            section=data["section"],
            content_type=Content_type(data["content_type"]) if data.get("content_type") else None,
            medical_system=MedicalSystem(data["medical_system"]) if data.get("medical_system") else None,
            keywords=data.get("keywords", []),
        )


@dataclass(frozen=True)
class Chunk:
    """
    A semantic chunk from a medical textbook.

    Attributes:
        chunk_id: Unique identifier for this chunk
        text: The chunk text content
        token_count: Number of tokens in the chunk
        section_metadata: Section-level metadata
        doc_id: Document identifier (textbook name)
        section_id: Section identifier (textbook:chapter:section)
    """

    chunk_id: str
    text: str
    token_count: int
    section_metadata: SectionMetadata
    doc_id: str
    section_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "section_metadata": self.section_metadata.to_dict(),
            "doc_id": self.doc_id,
            "section_id": self.section_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            token_count=data["token_count"],
            section_metadata=SectionMetadata.from_dict(data["section_metadata"]),
            doc_id=data["doc_id"],
            section_id=data["section_id"],
        )


@dataclass(frozen=True)
class QueryVariant:
    """
    A query variant for multi-query retrieval.

    Attributes:
        query_id: Unique identifier for this variant
        text: The query text
        kind: Type of query variant
        example_id: ID of the example this query belongs to
        metadata: Additional metadata
    """

    query_id: str
    text: str
    kind: QueryKind
    example_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "text": self.text,
            "kind": self.kind.value,
            "example_id": self.example_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryVariant":
        """Create from dictionary."""
        return cls(
            query_id=data["query_id"],
            text=data["text"],
            kind=QueryKind(data["kind"]),
            example_id=data["example_id"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class RetrievalResult:
    """
    A single retrieval result with scoring information.

    Attributes:
        chunk: The retrieved chunk
        score: Combined retrieval score
        rank: Rank position (0-indexed)
        scores: Individual component scores
        metadata: Additional retrieval metadata
    """

    chunk: Chunk
    score: float
    rank: int
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "text": self.chunk.text,
            "score": self.score,
            "rank": self.rank,
            "scores": self.scores,
            "metadata": self.metadata,
            "doc_id": self.chunk.doc_id,
            "section_id": self.chunk.section_id,
        }


@dataclass
class Context:
    """
    Assembled context for LLM prompting.

    Attributes:
        passages: Retrieved passages
        total_tokens: Total token count
        metadata: Assembly metadata
    """

    passages: List[RetrievalResult]
    total_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passages": [p.to_dict() for p in self.passages],
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }


@dataclass
class Prediction:
    """
    A single prediction result.

    Attributes:
        example_id: Example identifier
        question: The question text
        gold_answer: Gold standard answer
        predicted_answer: Model's prediction
        is_correct: Whether prediction matches gold
        context: Context used (if any)
        raw_output: Raw model output
        metadata: Additional metadata
    """

    example_id: str
    question: str
    gold_answer: Optional[str]
    predicted_answer: str
    is_correct: Optional[bool]
    context: Optional[Context]
    raw_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "example_id": self.example_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "context": self.context.to_dict() if self.context else None,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
