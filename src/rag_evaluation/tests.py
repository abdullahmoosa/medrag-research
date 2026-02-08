#!/usr/bin/env python3
"""
Unit tests for the Medical RAG Evaluation System.

Run with: python -m pytest src/rag_evaluation/tests.py -v
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from src.rag_evaluation.config import (
    EvaluationConfig,
    Mode,
    RetrievalMode,
    EmbeddingConfig,
    QueryConfig,
    FusionConfig,
)
from src.rag_evaluation.chunk_schema import (
    Chunk,
    SectionMetadata,
    QueryVariant,
    QueryKind,
    Content_type,
    RetrievalResult,
    Context,
    Prediction,
)
from src.rag_evaluation.index_loader import IndexManager
from src.rag_evaluation.query_reformulation import QueryReformulator, sanitize_query_expansions
from src.rag_evaluation.hyde import HyDEGenerator
from src.rag_evaluation.fusion import FusionStrategy


class TestConfig:
    """Test configuration classes."""

    def test_embedding_config_validation(self):
        """Test EmbeddingConfig validation."""
        # Valid config
        config = EmbeddingConfig()
        config.validate()

        # Invalid backend
        config.backend = "invalid"
        with pytest.raises(ValueError):
            config.validate()

    def test_query_config_validation(self):
        """Test QueryConfig validation."""
        # At least one query variant must be enabled
        config = QueryConfig(
            use_base_query=False,
            use_reformulation=False,
            use_hyde=False,
            use_option_aware=False,
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_fusion_config_validation(self):
        """Test FusionConfig validation."""
        # At least one weight must be non-zero
        config = FusionConfig(
            alpha_rrf=0.0,
            beta_overlap=0.0,
            bm25_weight=0.0,
            dense_weight=0.0,
        )
        with pytest.raises(ValueError):
            config.validate()


class TestChunkSchema:
    """Test chunk schema classes."""

    def test_section_metadata(self):
        """Test SectionMetadata creation and serialization."""
        meta = SectionMetadata(
            textbook="Harrison",
            chapter="Cardiology",
            section="Myocardial Infarction",
            content_type=Content_type.PATHOLOGY,
            medical_system="cardiovascular",
            keywords=["heart", "attack", "coronary"],
        )

        assert meta.textbook == "Harrison"
        assert meta.content_type == Content_type.PATHOLOGY

        # Test serialization
        d = meta.to_dict()
        assert d["textbook"] == "Harrison"
        assert d["content_type"] == "pathology"

        # Test deserialization
        meta2 = SectionMetadata.from_dict(d)
        assert meta2.textbook == meta.textbook
        assert meta2.content_type == meta.content_type

    def test_chunk(self):
        """Test Chunk creation and serialization."""
        section_meta = SectionMetadata(
            textbook="Harrison",
            chapter="Cardiology",
            section="MI",
        )

        chunk = Chunk(
            chunk_id="Harrison:Cardiology:MI:0",
            text="Myocardial infarction is...",
            token_count=42,
            section_metadata=section_meta,
            doc_id="Harrison",
            section_id="Harrison:Cardiology:MI",
        )

        assert chunk.chunk_id == "Harrison:Cardiology:MI:0"
        assert chunk.token_count == 42

    def test_query_variant(self):
        """Test QueryVariant creation."""
        query = QueryVariant(
            query_id="test_1",
            text="What is MI?",
            kind=QueryKind.BASE,
            example_id="example_1",
        )

        assert query.kind == QueryKind.BASE

        # Test serialization
        d = query.to_dict()
        assert d["kind"] == "base"

    def test_retrieval_result(self):
        """Test RetrievalResult creation."""
        section_meta = SectionMetadata(
            textbook="Harrison",
            chapter="Cardiology",
            section="MI",
        )

        chunk = Chunk(
            chunk_id="test_1",
            text="Test",
            token_count=10,
            section_metadata=section_meta,
            doc_id="Harrison",
            section_id="Harrison:Cardiology:MI",
        )

        result = RetrievalResult(
            chunk=chunk,
            score=0.89,
            rank=0,
            scores={"rrf": 0.85, "overlap": 0.12},
        )

        assert result.score == 0.89
        assert result.rank == 0


class TestQueryReformulation:
    """Test query reformulation."""

    def test_sanitize_query_expansions(self):
        """Test query expansion sanitization."""
        expansions = [
            "myocardial infarction",  # Good
            "answer is A",  # Bad - contains "answer"
            "true/false",  # Bad - contains "true/false"
            "a",  # Bad - too short
            "coronary artery disease",  # Good
        ]

        sanitized = sanitize_query_expansions(expansions)

        assert len(sanitized) == 2
        assert "myocardial infarction" in sanitized
        assert "coronary artery disease" in sanitized

    def test_heuristic_fallback(self):
        """Test heuristic fallback query generation."""
        reformulator = QueryReformulator(
            config=QueryConfig(),
            base_url="http://localhost:11434",
        )

        question = "A 45-year-old male presents with chest pain and shortness of breath."

        fallback = reformulator.heuristic_fallback(question)

        # Should extract meaningful keywords
        assert len(fallback) > 0
        assert "year-old" not in fallback
        assert "presents" not in fallback


class TestFusion:
    """Test fusion strategies."""

    def test_minmax_normalize(self):
        """Test min-max normalization."""
        fusion = FusionStrategy(FusionConfig())

        values = [0.1, 0.5, 0.9]
        normalized = fusion._minmax_normalize(values)

        assert normalized[0] == 0.0
        assert normalized[1] == 0.5
        assert normalized[2] == 1.0

    def test_rrf_fusion(self):
        """Test RRF fusion."""
        fusion = FusionStrategy(FusionConfig())

        # Two retriever results
        list1 = [(0, 0.9), (1, 0.8), (2, 0.7)]
        list2 = [(1, 0.9), (2, 0.8), (3, 0.7)]

        fused = fusion.fuse_rrf(
            rank_lists=[list1, list2],
            rrf_k=60,
            top_k=5,
        )

        # Result should be list of (idx, score) tuples
        assert len(fused) > 0
        assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in fused)

    def test_lexical_overlap(self):
        """Test lexical overlap calculation."""
        fusion = FusionStrategy(FusionConfig())

        query = "myocardial infarction treatment"
        text = "myocardial infarction is treated with thrombolytics"

        overlap = fusion._lexical_overlap(query, text)

        # Should have some overlap
        assert overlap > 0

        # No overlap
        no_overlap = fusion._lexical_overlap("diabetes", "myocardial infarction")
        assert no_overlap == 0.0


class TestHyDE:
    """Test HyDE generation."""

    def test_is_negation_question(self):
        """Test negation question detection."""
        config = QueryConfig()
        generator = HyDEGenerator(config, "http://localhost:11434")

        # Negation questions
        assert generator.is_negation_question("All of the following EXCEPT:")
        assert generator.is_negation_question("Which is NOT true?")

        # Non-negation
        assert not generator.is_negation_question("What is the diagnosis?")


class TestIndexManager:
    """Test index manager."""

    @pytest.fixture
    def index_dir(self, tmp_path):
        """Create a temporary index directory."""
        # Create mock index files
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        # Create a mock chunks file
        chunks_file = index_dir / "medtextbooks_v1.jsonl"
        with open(chunks_file, "w") as f:
            f.write('{"textbook": "Harrison", "chapter": "Cardiology", "section": "MI", "text": "Test", "token_count": 10}')

        return index_dir

    def test_load_chunks(self, index_dir):
        """Test loading chunks from file."""
        manager = IndexManager(index_dir, "medtextbooks_v1")

        # Should not raise errors
        manager._load_chunks()

        assert len(manager.chunks) == 1
        assert manager.chunks[0].textbook == "Harrison"


# Integration tests (require actual indexes)

@pytest.mark.integration
class TestIntegration:
    """Integration tests (require actual indexes)."""

    def test_load_real_indexes(self):
        """Test loading real indexes from processed_corpus."""
        index_dir = Path("/home/ser/medrag/processed_corpus")

        if not index_dir.exists():
            pytest.skip("Processed corpus not found")

        manager = IndexManager(index_dir, "medtextbooks_v1")
        manager.load()

        assert manager.num_chunks > 0
        assert manager.bm25_index is not None
        assert manager.faiss_index is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
