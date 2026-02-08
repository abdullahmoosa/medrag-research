#!/usr/bin/env python3
"""
Example usage of the rechunking pipeline.

This script demonstrates different ways to use the pipeline.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rechunk_pipeline import (
    RechunkingPipeline,
    PipelineConfig,
    SemanticChunker,
    TextCleaner,
)


def example_1_basic_usage():
    """
    Example 1: Basic usage with default config.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)

    # Use default configuration
    config = PipelineConfig()

    # Adjust paths if needed
    config.input_dir = "/home/ser/medrag/data/medQA USMLE/textbooks/en"
    config.output_dir = "/home/ser/medrag/processed_corpus"
    config.corpus_name = "medtextbooks_rechunked"

    # Create and run pipeline
    pipeline = RechunkingPipeline(config)
    stats = pipeline.run()

    print(f"\nProcessed {stats['files_processed']} files")
    print(f"Created {stats['total_chunks']} chunks")


def example_2_custom_config():
    """
    Example 2: Using custom configuration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Configuration")
    print("="*70)

    from src.rechunk_pipeline.config import ChunkingConfig, EmbeddingConfig

    # Create custom config
    config = PipelineConfig(
        input_dir="/home/ser/medrag/data/medQA USMLE/textbooks/en",
        output_dir="/home/ser/medrag/processed_corpus_custom",
        corpus_name="medtextbooks_custom_chunks",

        # Custom chunking parameters
        chunking=ChunkingConfig(
            min_tokens=80,
            target_tokens=150,
            max_tokens=250,
        ),

        # Custom embedding model
        embedding=EmbeddingConfig(
            model="BAAI/bge-large-en-v1.5",
            backend="sentence_transformers",
        ),
    )

    pipeline = RechunkingPipeline(config)
    stats = pipeline.run()


def example_3_load_from_file():
    """
    Example 3: Load configuration from file.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Load Config from File")
    print("="*70)

    from src.rechunk_pipeline.config import get_config

    # Load config from example_config.py
    config_path = Path(__file__).parent / "example_config.py"
    config = get_config(str(config_path))

    pipeline = RechunkingPipeline(config)
    stats = pipeline.run()


def example_4_individual_components():
    """
    Example 4: Using individual components.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Using Individual Components")
    print("="*70)

    from src.rechunk_pipeline.config import CleaningConfig, ChunkingConfig
    from src.rechunk_pipeline.clean_text import TextCleaner
    from src.rechunk_pipeline.split_structure import StructureSplitter
    from src.rechunk_pipeline.semantic_chunker import SemanticChunker
    from src.rechunk_pipeline.metadata import MetadataExtractor

    # Process a single file
    file_path = Path("/home/ser/medrag/data/medQA USMLE/textbooks/en/Anatomy_Gray.txt")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    # Read file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    print(f"Processing: {file_path.name}")
    print(f"Original length: {len(text)} characters")

    # Clean text
    cleaner = TextCleaner(CleaningConfig())
    cleaned = cleaner.clean_text(text)
    print(f"Cleaned length: {len(cleaned)} characters")

    # Split structure
    splitter = StructureSplitter()
    sections = splitter.split_textbook(cleaned, file_path.stem)
    print(f"Found {len(sections)} sections")

    # Chunk
    chunker = SemanticChunker(ChunkingConfig())
    chunks = chunker.chunk_sections(sections, "demo")
    print(f"Created {len(chunks)} chunks")

    # Add metadata
    extractor = MetadataExtractor(
        type('Config', (), {
            'detect_content_type': True,
            'detect_medical_system': True,
            'extract_keywords': True,
            'max_keywords': 10,
        })()
    )
    chunks = extractor.enrich_chunks(chunks)

    # Show sample chunk
    if chunks:
        print("\n" + "-"*70)
        print("SAMPLE CHUNK:")
        print("-"*70)
        chunk = chunks[0]
        print(f"Textbook: {chunk.textbook}")
        print(f"Chapter: {chunk.chapter}")
        print(f"Section: {chunk.section}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Content Type: {chunk.metadata.get('content_type')}")
        print(f"System: {chunk.metadata.get('system')}")
        print(f"\nText preview:\n{chunk.text[:300]}...")


def main():
    """Run examples."""
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="Example usage of rechunking pipeline")
    parser.add_argument(
        '--example',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help='Which example to run (1-4)'
    )

    args = parser.parse_args()

    examples = {
        1: example_1_basic_usage,
        2: example_2_custom_config,
        3: example_3_load_from_file,
        4: example_4_individual_components,
    }

    examples[args.example]()


if __name__ == '__main__':
    main()
