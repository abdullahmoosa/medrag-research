#!/usr/bin/env python3
"""
Example: How to enable overlap in chunking.

This shows the three configurations you might want:
1. No overlap (default) - current setup
2. With overlap - for better boundary retrieval
3. With overlap + filtering - clean up tiny chunks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rechunk_pipeline import PipelineConfig, ChunkingConfig

# =============================================================================
# OPTION 1: NO OVERLAP (DEFAULT)
# =============================================================================
# Use this for your first run - clean, focused chunks
config_no_overlap = PipelineConfig(
    input_dir="/home/ser/medrag/data/medQA USMLE/textbooks/en",
    output_dir="/home/ser/medrag/processed_corpus_no_overlap",
    corpus_name="medtextbooks_no_overlap",

    chunking=ChunkingConfig(
        min_tokens=150,
        target_tokens=250,
        max_tokens=400,
        # Overlap disabled (default)
        enable_overlap=False,
    ),
)

# =============================================================================
# OPTION 2: WITH OVERLAP
# =============================================================================
# Use this if you notice concepts getting cut off at chunk boundaries
config_with_overlap = PipelineConfig(
    input_dir="/home/ser/medrag/data/medQA USMLE/textbooks/en",
    output_dir="/home/ser/medrag/processed_corpus_with_overlap",
    corpus_name="medtextbooks_with_overlap",

    chunking=ChunkingConfig(
        min_tokens=150,
        target_tokens=250,
        max_tokens=400,
        # Enable 10% overlap
        enable_overlap=True,
        overlap_ratio=0.10,  # 10% overlap
        overlap_strategy="tokens",  # Can also use "sentences"
    ),
)

# =============================================================================
# OPTION 3: WITH OVERLAP + FILTERING
# =============================================================================
# Use this to both add overlap AND clean up tiny fragments
config_overlap_filtered = PipelineConfig(
    input_dir="/home/ser/medrag/data/medQA USMLE/textbooks/en",
    output_dir="/home/ser/medrag/processed_corpus_overlap_filtered",
    corpus_name="medtextbooks_overlap_filtered",

    chunking=ChunkingConfig(
        min_tokens=150,
        target_tokens=250,
        max_tokens=400,
        # Enable overlap
        enable_overlap=True,
        overlap_ratio=0.15,  # 15% overlap
        overlap_strategy="sentences",  # Complete sentences

        # Filter tiny chunks
        filter_min_tokens=100,  # Remove chunks < 100 tokens
    ),
)

# =============================================================================
# HOW TO USE
# =============================================================================

print("="*70)
print("OVERLAP FEATURE - CONFIGURATION OPTIONS")
print("="*70)

print("""
The overlap feature is now available but DISABLED by default.

To enable it, simply change one parameter in your config:

    chunking=ChunkingConfig(
        min_tokens=150,
        target_tokens=250,
        max_tokens=400,
        enable_overlap=True,  # <-- Just change this!
        overlap_ratio=0.10,  # 10% overlap (adjustable)
    )

RECOMMENDATION:
- First run: enable_overlap=False (current default)
- If retrieval boundary issues: enable_overlap=True
- Also want to clean fragments: add filter_min_tokens=100

To run with overlap:
    python -m src.rechunk_pipeline.run_experiment --config src/rechunk_pipeline/example_config.py

Then edit example_config.py and set:
    chunking.enable_overlap = True
""")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print("""
NO OVERLAP (current default):
  ✓ Cleaner chunks
  ✓ No redundancy
  ✓ Focused retrieval
  ✗ Concepts may split at boundaries
  ✗ Query might match end of chunk 1 AND start of chunk 2

WITH 10% OVERLAP:
  ✓ Better boundary coverage
  ✓ Less likely to miss concepts
  ✓ More context for LLM
  ✗ Redundant content
  ✗ Slightly larger corpus
  ✗ More noisy retrieval

WITH 10% OVERLAP + FILTERING:
  ✓ Best of both worlds
  ✓ Clean fragments removed
  ✓ Boundary issues addressed
  ✗ Most redundant
""")

print("\nRecommendation: Start with NO OVERLAP, add later if needed.")
print("="*70)
