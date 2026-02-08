#!/usr/bin/env python3
"""
Example usage of the Medical RAG Evaluation System.

Demonstrates various evaluation scenarios.
"""

import asyncio
import logging
from pathlib import Path

from src.rag_evaluation import (
    EvaluationConfig,
    EvaluationOrchestrator,
    Mode,
    RetrievalMode,
)


async def example_no_rag_baseline():
    """
    Example 1: No-RAG baseline evaluation.

    This establishes the LLM-only performance without retrieval.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: No-RAG Baseline Evaluation")
    print("=" * 70)

    config = EvaluationConfig()
    config.retrieval.mode = Mode.NO_RAG
    config.eval_data_path = Path("/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl")
    config.output_dir = Path("./examples/no_rag_baseline")
    config.limit = 100  # For demo purposes

    orchestrator = EvaluationOrchestrator(config)
    metrics = await orchestrator.evaluate()

    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  With context: {metrics['with_context_accuracy']:.4f}")
    print(f"  Without context: {metrics['without_context_accuracy']:.4f}")


async def example_basic_rag():
    """
    Example 2: Basic RAG with hybrid retrieval.

    Simple RAG setup with dense + BM25 fusion.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Basic RAG with Hybrid Retrieval")
    print("=" * 70)

    config = EvaluationConfig()
    config.retrieval.mode = Mode.RAG
    config.retrieval.fine.mode = RetrievalMode.HYBRID
    config.retrieval.fine.top_k = 12
    config.eval_data_path = Path("/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl")
    config.output_dir = Path("./examples/basic_rag")
    config.limit = 100

    orchestrator = EvaluationOrchestrator(config)
    metrics = await orchestrator.evaluate()

    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  With context: {metrics['with_context_accuracy']:.4f}")


async def example_rag_with_reformulation():
    """
    Example 3: RAG with LLM query reformulation.

    Uses LLM to rewrite questions for better retrieval.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: RAG with Query Reformulation")
    print("=" * 70)

    config = EvaluationConfig()
    config.retrieval.mode = Mode.RAG
    config.retrieval.fine.mode = RetrievalMode.HYBRID
    config.retrieval.fine.top_k = 12

    # Enable query reformulation
    config.retrieval.queries.use_reformulation = True
    config.retrieval.queries.reformulation_model = "llama3-med42-8b"
    config.retrieval.queries.reformulation_temperature = 0.0

    config.eval_data_path = Path("/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl")
    config.output_dir = Path("./examples/rag_reformulation")
    config.limit = 100

    orchestrator = EvaluationOrchestrator(config)
    metrics = await orchestrator.evaluate()

    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  With context: {metrics['with_context_accuracy']:.4f}")


async def example_rag_with_hyde():
    """
    Example 4: RAG with HyDE (Hypothetical Document Embeddings).

    Generates hypothetical passages for retrieval.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: RAG with HyDE")
    print("=" * 70)

    config = EvaluationConfig()
    config.retrieval.mode = Mode.RAG
    config.retrieval.fine.mode = RetrievalMode.HYBRID
    config.retrieval.fine.top_k = 12

    # Enable HyDE
    config.retrieval.queries.use_hyde = True
    config.retrieval.queries.hyde_mode = "question"
    config.retrieval.queries.hyde_model = "llama3-med42-8b"
    config.retrieval.queries.hyde_temperature = 0.2

    config.eval_data_path = Path("/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl")
    config.output_dir = Path("./examples/rag_hyde")
    config.limit = 100

    orchestrator = EvaluationOrchestrator(config)
    metrics = await orchestrator.evaluate()

    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  With context: {metrics['with_context_accuracy']:.4f}")


async def example_full_pipeline():
    """
    Example 5: Full pipeline with all features.

    Demonstrates the complete system with:
    - Coarse retrieval (section-level)
    - Fine retrieval (chunk-level)
    - Query reformulation
    - HyDE
    - Option-aware retrieval
    - Reranking
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Full Pipeline")
    print("=" * 70)

    config = EvaluationConfig()
    config.retrieval.mode = Mode.RAG

    # Coarse retrieval
    config.retrieval.coarse.enabled = True
    config.retrieval.coarse.top_k_sections = 20
    config.retrieval.coarse.mode = RetrievalMode.HYBRID

    # Fine retrieval
    config.retrieval.fine.top_k = 12
    config.retrieval.fine.dense_k = 80
    config.retrieval.fine.bm25_k = 400
    config.retrieval.fine.max_passages = 6
    config.retrieval.fine.mode = RetrievalMode.HYBRID

    # Query variants
    config.retrieval.queries.use_base_query = True
    config.retrieval.queries.use_reformulation = True
    config.retrieval.queries.use_hyde = True
    config.retrieval.queries.hyde_mode = "both"
    config.retrieval.queries.use_option_aware = True

    # Fusion
    config.retrieval.fusion.rrf_k = 60
    config.retrieval.fusion.alpha_rrf = 0.6
    config.retrieval.fusion.beta_overlap = 0.4

    # Reranker
    config.retrieval.reranker.enabled = True
    config.retrieval.reranker.model_name = "BAAI/bge-reranker-large"
    config.retrieval.reranker.top_k = 150
    config.retrieval.reranker.use_fp16 = True

    # LLM
    config.llm.use_cot = False

    config.eval_data_path = Path("/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl")
    config.output_dir = Path("./examples/full_pipeline")
    config.limit = 100

    orchestrator = EvaluationOrchestrator(config)
    metrics = await orchestrator.evaluate()

    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  With context: {metrics['with_context_accuracy']:.4f}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MEDICAL RAG EVALUATION SYSTEM - EXAMPLES")
    print("=" * 70)

    # Uncomment the examples you want to run:

    # await example_no_rag_baseline()
    # await example_basic_rag()
    # await example_rag_with_reformulation()
    # await example_rag_with_hyde()
    # await example_full_pipeline()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
