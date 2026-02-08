"""
Command-line interface for RAG evaluation.

Supports running evaluations with various configurations via CLI flags.
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from .config import EvaluationConfig, Mode, RetrievalMode, RerankerType
from .evaluation import EvaluationOrchestrator
from .chunk_schema import Content_type


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Medical RAG Evaluation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    ap.add_argument(
        "--mode",
        type=str,
        choices=["no_rag", "rag"],
        default="rag",
        help="Evaluation mode: no_rag (baseline) or rag (retrieval + LLM)",
    )

    # Paths
    ap.add_argument(
        "--index-dir",
        type=str,
        default="/home/ser/medrag/processed_corpus",
        help="Path to processed corpus with indexes",
    )
    ap.add_argument(
        "--eval-path",
        type=str,
        default="/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl",
        help="Path to evaluation data",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="/home/ser/medrag/evaluation_results",
        help="Output directory for results",
    )
    ap.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Index name for organizing results (e.g., 'index_1', 'medembed', 'bge-large'). Auto-detected from index metadata if not specified.",
    )

    # Embeddings
    ap.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model name",
    )
    ap.add_argument(
        "--embedding-backend",
        type=str,
        choices=["sentence_transformers", "ollama", "medembed"],
        default="sentence_transformers",
        help="Embedding backend",
    )
    ap.add_argument(
        "--embedding-device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Embedding inference device",
    )
    ap.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Embedding batch size",
    )

    # Retrieval mode
    ap.add_argument(
        "--retrieval-mode",
        type=str,
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval mode",
    )

    # Coarse retrieval
    ap.add_argument(
        "--coarse-retrieval",
        action="store_true",
        help="Enable coarse (section-level) retrieval",
    )
    ap.add_argument(
        "--coarse-top-k",
        type=int,
        default=20,
        help="Top-k sections for coarse retrieval",
    )

    # Fine retrieval
    ap.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top-k results for final retrieval",
    )
    ap.add_argument(
        "--dense-k",
        type=int,
        default=80,
        help="Top-k for dense retriever",
    )
    ap.add_argument(
        "--bm25-k",
        type=int,
        default=400,
        help="Top-k for BM25 retriever",
    )
    ap.add_argument(
        "--max-passages",
        type=int,
        default=6,
        help="Maximum passages to use as context",
    )
    ap.add_argument(
        "--max-evidence-tokens",
        type=int,
        default=1200,
        help="Maximum tokens for evidence packing",
    )

    # Query variants
    ap.add_argument(
        "--no-base-query",
        action="store_true",
        help="Disable base query",
    )
    ap.add_argument(
        "--use-reformulation",
        action="store_true",
        help="Enable LLM query reformulation",
    )
    ap.add_argument(
        "--use-hyde",
        action="store_true",
        help="Enable HyDE query generation",
    )
    ap.add_argument(
        "--hyde-mode",
        type=str,
        choices=["question", "option", "both"],
        default="question",
        help="HyDE generation mode",
    )
    ap.add_argument(
        "--use-option-aware",
        action="store_true",
        help="Enable option-aware retrieval",
    )
    ap.add_argument(
        "--use-expansions",
        action="store_true",
        help="Use dataset query expansions",
    )

    # Fusion
    ap.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant",
    )
    ap.add_argument(
        "--alpha-rrf",
        type=float,
        default=0.6,
        help="Weight for RRF score",
    )
    ap.add_argument(
        "--beta-overlap",
        type=float,
        default=0.4,
        help="Weight for lexical overlap",
    )
    ap.add_argument(
        "--min-final-threshold",
        type=float,
        default=0.35,
        help="Minimum final score threshold",
    )
    ap.add_argument(
        "--min-lexical-overlap",
        type=float,
        default=0.01,
        help="Minimum lexical overlap threshold",
    )
    ap.add_argument(
        "--prioritize-content-types",
        type=str,
        nargs="+",
        default=None,
        help="Content types to prioritize in ranking (e.g., mechanism definition pathology)",
    )

    # Reranker
    ap.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranker",
    )
    ap.add_argument(
        "--reranker-model",
        type=str,
        default="BAAI/bge-reranker-large",
        help="Reranker model name",
    )
    ap.add_argument(
        "--reranker-top-k",
        type=int,
        default=150,
        help="Top-k candidates to rerank",
    )
    ap.add_argument(
        "--reranker-fp16",
        action="store_true",
        help="Use FP16 for reranker (CUDA only)",
    )
    ap.add_argument(
        "--min-ce-score",
        type=float,
        help="Minimum cross-encoder score threshold (gating)",
    )

    # LLM
    ap.add_argument(
        "--llm-model",
        type=str,
        default="thewindmom/llama3-med42-8b:latest",
        help="LLM model name (Ollama)",
    )
    ap.add_argument(
        "--llm-base-url",
        type=str,
        default="http://172.25.208.1:11434",
        help="Ollama base URL",
    )
    ap.add_argument(
        "--use-cot",
        action="store_true",
        help="Enable chain-of-thought prompting",
    )
    ap.add_argument(
        "--reasoning-model",
        action="store_true",
        help="Enable reasoning model mode (handles thinking/reasoning fields)",
    )
    ap.add_argument(
        "--llm-num-workers",
        type=int,
        default=8,
        help="Number of LLM worker threads",
    )

    # Processing
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for LLM scoring",
    )
    ap.add_argument(
        "--retrieval-batch-size",
        type=int,
        default=256,
        help="Batch size for retrieval",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples (for testing)",
    )

    # Logging
    ap.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return ap.parse_args()


def build_config(args: argparse.Namespace) -> EvaluationConfig:
    """
    Build evaluation configuration from CLI arguments.

    Args:
        args: Parsed arguments

    Returns:
        EvaluationConfig object
    """
    from .config import (
        RetrievalConfig,
        QueryConfig,
        FusionConfig,
        RerankerConfig,
        CoarseRetrievalConfig,
        FineRetrievalConfig,
        EmbeddingConfig,
        LLMConfig,
    )

    # Map mode string to enum
    mode = Mode.NO_RAG if args.mode == "no_rag" else Mode.RAG
    retrieval_mode = (
        RetrievalMode.DENSE
        if args.retrieval_mode == "dense"
        else RetrievalMode.BM25 if args.retrieval_mode == "bm25" else RetrievalMode.HYBRID
    )

    # Build config
    config = EvaluationConfig()

    # Paths
    config.index_dir = Path(args.index_dir)
    config.eval_data_path = Path(args.eval_path)
    config.output_dir = Path(args.output_dir)

    # Mode
    config.retrieval.mode = mode

    # Retrieval
    config.retrieval.coarse.enabled = args.coarse_retrieval
    config.retrieval.coarse.top_k_sections = args.coarse_top_k
    config.retrieval.coarse.mode = retrieval_mode

    config.retrieval.fine.top_k = args.top_k
    config.retrieval.fine.dense_k = args.dense_k
    config.retrieval.fine.bm25_k = args.bm25_k
    config.retrieval.fine.mode = retrieval_mode
    config.retrieval.fine.max_passages = args.max_passages
    config.retrieval.fine.max_evidence_tokens = args.max_evidence_tokens

    # Queries
    config.retrieval.queries.use_base_query = not args.no_base_query
    config.retrieval.queries.use_reformulation = args.use_reformulation
    config.retrieval.queries.use_hyde = args.use_hyde
    config.retrieval.queries.hyde_mode = args.hyde_mode
    config.retrieval.queries.use_option_aware = args.use_option_aware
    config.retrieval.queries.use_expansions = args.use_expansions

    # Fusion
    config.retrieval.fusion.rrf_k = args.rrf_k
    config.retrieval.fusion.alpha_rrf = args.alpha_rrf
    config.retrieval.fusion.beta_overlap = args.beta_overlap
    config.retrieval.fusion.min_final_threshold = args.min_final_threshold
    config.retrieval.fusion.min_lexical_overlap = args.min_lexical_overlap

    # Metadata filtering
    if args.prioritize_content_types:
        from .chunk_schema import Content_type
        content_type_map = {
            "anatomy": Content_type.ANATOMY,
            "physiology": Content_type.PHYSIOLOGY,
            "pathology": Content_type.PATHOLOGY,
            "pharmacology": Content_type.PHARMACOLOGY,
            "microbiology": Content_type.MICROBIOLOGY,
            "immunology": Content_type.IMMUNOLOGY,
            "genetics": Content_type.GENETICS,
            "epidemiology": Content_type.EPIDEMIOLOGY,
            "clinical": Content_type.CLINICAL,
            "diagnostics": Content_type.DIAGNOSTICS,
            "therapeutics": Content_type.THERAPEUTICS,
            "procedures": Content_type.PROCEDURES,
            "mechanism": Content_type.MECHANISM,
            "definition": Content_type.DEFINITION,
            "other": Content_type.OTHER,
        }

        prioritized = []
        for ct_str in args.prioritize_content_types:
            ct_str_lower = ct_str.lower()
            if ct_str_lower in content_type_map:
                prioritized.append(content_type_map[ct_str_lower])
            else:
                print(f"Warning: Unknown content type '{ct_str}', skipping")

        config.retrieval.prioritize_content_types = prioritized

    # Reranker
    config.retrieval.reranker.enabled = args.use_reranker
    config.retrieval.reranker.model_name = args.reranker_model
    config.retrieval.reranker.top_k = args.reranker_top_k
    config.retrieval.reranker.use_fp16 = args.reranker_fp16
    config.retrieval.reranker.min_score_threshold = args.min_ce_score

    # Embeddings
    config.embedding.model_name = args.embedding_model
    config.embedding.backend = args.embedding_backend
    config.embedding.device = args.embedding_device
    config.embedding.batch_size = args.embedding_batch_size

    # LLM
    config.llm.model_name = args.llm_model
    config.llm.base_url = args.llm_base_url
    config.llm.use_cot = args.use_cot
    config.llm.reasoning_model = args.reasoning_model
    config.llm.num_workers = args.llm_num_workers

    # Processing
    config.batch_size = args.batch_size
    config.retrieval_batch_size = args.retrieval_batch_size
    config.limit = args.limit

    # Logging
    config.log_level = args.log_level
    config.verbose = args.verbose

    return config


def detect_index_name(index_dir: Path) -> str:
    """
    Auto-detect index name from metadata.

    Args:
        index_dir: Path to index directory

    Returns:
        Detected index name (e.g., 'medembed', 'bge-large', or 'index_1' as fallback)
    """
    import json

    # Try to find meta.json
    meta_files = list(index_dir.glob("*_meta.json"))

    if not meta_files:
        return "index_1"  # Fallback to original

    meta_file = meta_files[0]

    try:
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        # Get embedding model from config
        embedding_model = metadata.get('config', {}).get('embedding_model', '')

        if not embedding_model:
            return "index_1"

        # Generate meaningful name from embedding model
        # e.g., "abhinand/MedEmbed-large-v0.1" -> "medembed"
        # e.g., "BAAI/bge-large-en-v1.5" -> "bge-large"
        # e.g., "ncbi/MedCPT-Article-Encoder" -> "medcpt"

        # Extract base model name
        model_parts = embedding_model.split('/')
        model_name = model_parts[-1] if model_parts else embedding_model

        # Convert to lowercase and extract key identifier
        model_lower = model_name.lower()

        # Common patterns
        if 'medembed' in model_lower:
            return 'medembed'
        elif 'medcpt' in model_lower:
            return 'medcpt'
        elif 'bge' in model_lower:
            # Extract version: bge-large-en-v1.5 -> bge-large
            bge_parts = model_name.replace('-', '_').split('_')
            if len(bge_parts) >= 2:
                return f"bge-{bge_parts[1]}"  # bge-large, bge-base, etc.
            return 'bge'
        elif 'biobert' in model_lower or 'pubmedbert' in model_lower:
            return 'biobert'
        else:
            # Use first part of model name as fallback
            return model_name.split('-')[0].split('_')[0]

    except Exception as e:
        logger.warning(f"Failed to detect index name from metadata: {e}")
        return "index_1"


def generate_output_dir(config: EvaluationConfig, index_name: str = None) -> Path:
    """
    Generate organized output directory based on configuration.

    Creates a hierarchical structure:
    - final_output/NO_RAG/zero_shot|cot/{model}/
    - final_output/RAG/{index_name}/{retrieval_mode}/reranker_on|off/{reformulation|no_reformulation}/{model}/

    If a directory already exists, creates a versioned subdirectory (v2, v3, etc.)

    Args:
        config: Evaluation configuration
        index_name: Optional index name (auto-detected if not specified)

    Returns:
        Path to output directory
    """
    base_dir = Path("evaluation_results/final_output")

    # Sanitize model name (e.g., "thewindmom/llama3-med42-8b:latest" -> "llama3_med42_8b")
    model_name = config.llm.model_name
    # Remove user/org prefix
    model_name = model_name.split("/")[-1] if "/" in model_name else model_name
    # Remove tag/version suffix
    model_name = model_name.split(":")[0] if ":" in model_name else model_name
    # Replace hyphens and dots with underscores
    model_name = model_name.replace("-", "_").replace(".", "_")

    if config.retrieval.mode == Mode.NO_RAG:
        # NO_RAG structure: NO_RAG/zero_shot|cot/{model}/
        cot_dir = "cot" if config.llm.use_cot else "zero_shot"
        base_output_dir = base_dir / "NO_RAG" / cot_dir / model_name

    else:
        # RAG structure: RAG/{index_name}/{retrieval_mode}/coarse_k{top_k}|coarse_off/reranker_on|off/{reformulation|no_reformulation}/{model}/{zero_shot|cot}/

        # Auto-detect index name if not specified
        if index_name is None:
            index_name = detect_index_name(config.index_dir)
            logger.info(f"Auto-detected index name: {index_name}")

        # Sanitize index name (replace slashes, etc.)
        index_name = index_name.replace('/', '_').replace('\\', '_')

        # Retrieval mode
        retrieval_mode = config.retrieval.fine.mode.value

        # Coarse retrieval
        if config.retrieval.coarse.enabled:
            coarse_dir = f"coarse_k{config.retrieval.coarse.top_k_sections}"
        else:
            coarse_dir = "coarse_off"

        # Reranker
        reranker_dir = "reranker_on" if config.retrieval.reranker.enabled else "reranker_off"

        # Reformulation
        reformulation_dir = "reformulation" if config.retrieval.queries.use_reformulation else "no_reformulation"

        # Chain-of-thought (leaf level directory)
        cot_dir = "cot" if config.llm.use_cot else "zero_shot"

        base_output_dir = base_dir / "RAG" / index_name / retrieval_mode / coarse_dir / reranker_dir / reformulation_dir / model_name / cot_dir

    # Check if directory already exists, create versioned subdirectory if so
    output_dir = base_output_dir
    version = 1

    while output_dir.exists():
        # Check if this is a versioned directory or the original
        if output_dir == base_output_dir:
            # Original exists, try v2
            version = 2
        else:
            version += 1

        # Create versioned directory name
        output_dir = base_output_dir / f"v{version}"

        # Safety check to prevent infinite loop
        if version > 100:
            raise RuntimeError(f"Too many versioned directories (up to v{version}) for {base_output_dir}")

    return output_dir


async def main_async() -> int:
    """Main async entry point."""
    args = parse_args()
    config = build_config(args)

    # Generate organized output directory
    organized_output_dir = generate_output_dir(config, index_name=args.index_name)

    # Override output_dir with organized structure
    config.output_dir = organized_output_dir

    # Print configuration
    print("\n" + "=" * 70)
    print("MEDICAL RAG EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Mode: {config.retrieval.mode.value}")
    print(f"  Index: {config.index_dir}")
    print(f"  Data: {config.eval_data_path}")
    print(f"  Output: {config.output_dir}")
    print(f"\nRetrieval:")
    print(f"  Mode: {config.retrieval.fine.mode.value}")
    print(f"  Coarse: {config.retrieval.coarse.enabled}")
    print(f"  Top-K: {config.retrieval.fine.top_k}")
    print(f"\nQueries:")
    print(f"  Base: {config.retrieval.queries.use_base_query}")
    print(f"  Reformulation: {config.retrieval.queries.use_reformulation}")
    print(f"  HyDE: {config.retrieval.queries.use_hyde}")
    print(f"  Option-aware: {config.retrieval.queries.use_option_aware}")
    print(f"\nEmbeddings:")
    print(f"  Model: {config.embedding.model_name}")
    print(f"  Backend: {config.embedding.backend}")
    print(f"  Device: {config.embedding.device}")
    print(f"\nLLM:")
    print(f"  Model: {config.llm.model_name}")
    print(f"  CoT: {config.llm.use_cot}")
    print("=" * 70 + "\n")

    # Run evaluation
    orchestrator = EvaluationOrchestrator(config)
    metrics = await orchestrator.evaluate()

    return 0


def main():
    """CLI entry point."""
    import sys

    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
