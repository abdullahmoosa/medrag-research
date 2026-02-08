#!/usr/bin/env python3
"""
Experiment runner for medical textbook rechunking pipeline.

Usage:
    python run_experiment.py --config config.py
    python run_experiment.py --input_dir /path/to/textbooks --output_dir ./output
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

from .config import PipelineConfig, get_config
from .clean_text import TextCleaner, clean_text_file
from .split_structure import StructureSplitter, process_textbook_file
from .semantic_chunker import SemanticChunker
from .metadata import MetadataExtractor, create_chunk_metadata, validate_metadata
from .index_builder import IndexBuilder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rechunking_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class RechunkingPipeline:
    """
    Main pipeline for rechunking medical textbooks.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize components
        self.cleaner = TextCleaner(config.cleaning)
        self.splitter = StructureSplitter(config.chunking.preserve_structure)
        self.chunker = SemanticChunker(config.chunking)
        self.metadata_extractor = MetadataExtractor(config.metadata)

        # Index builder (lazy initialization)
        self.index_builder = None

        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'cleaning_stats': [],
        }

    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Returns:
            Pipeline statistics
        """
        logger.info("="*70)
        logger.info("Starting Medical Textbook Rechunking Pipeline")
        logger.info("="*70)

        start_time = time.time()

        # Find textbook files
        textbook_files = self._find_textbooks()
        logger.info(f"Found {len(textbook_files)} textbook files")

        if not textbook_files:
            logger.error("No textbook files found!")
            return self.stats

        # Process each textbook
        all_chunks = []

        for i, file_path in enumerate(textbook_files, 1):
            logger.info(f"\n[{i}/{len(textbook_files)}] Processing: {file_path.name}")

            try:
                chunks = self._process_textbook(file_path)
                all_chunks.extend(chunks)
                self.stats['files_processed'] += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Enrich chunks with metadata
        logger.info(f"\nEnriching {len(all_chunks)} chunks with metadata...")
        all_chunks = self.metadata_extractor.enrich_chunks(all_chunks)

        # Update statistics
        self.stats['total_chunks'] = len(all_chunks)
        self.stats['total_tokens'] = sum(c.token_count for c in all_chunks)

        # Validate chunks
        if self.config.validate_chunks:
            validation = self.chunker.validate_chunks(all_chunks)
            self.stats['validation'] = validation
            logger.info(f"Validation: {validation['valid']} - {validation['num_issues']} issues")

        # Build indexes
        if self.config.index.bm25_enabled or self.config.index.faiss_enabled:
            logger.info("\nBuilding indexes...")
            self.index_builder = IndexBuilder(
                self.config.index,
                self.config.embedding
            )
            index_stats = self.index_builder.build_indexes(all_chunks)
            self.stats.update(index_stats)

        # Save outputs
        self._save_outputs(all_chunks)

        elapsed = time.time() - start_time
        self.stats['processing_time_seconds'] = elapsed
        logger.info(f"\nPipeline completed in {elapsed:.1f} seconds")

        self._log_final_stats()

        return self.stats

    def _find_textbooks(self) -> List[Path]:
        """Find all textbook files in input directory."""
        input_dir = Path(self.config.input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all .txt files
        files = list(input_dir.glob("*.txt"))
        logger.info(f"Found {len(files)} .txt files in {input_dir}")

        return sorted(files)

    def _process_textbook(self, file_path: Path) -> List:
        """
        Process a single textbook file.

        Args:
            file_path: Path to textbook file

        Returns:
            List of chunks
        """
        # Step 1: Clean text
        logger.info("  Cleaning text...")
        cleaned_text, clean_stats = clean_text_file(file_path, self.cleaner)
        self.stats['cleaning_stats'].append(clean_stats)

        # Step 2: Split structure
        logger.info("  Splitting into sections...")
        sections = process_textbook_file(file_path, cleaned_text, self.splitter)
        logger.info(f"    Found {len(sections)} sections")

        # Step 3: Semantic chunking
        logger.info("  Creating semantic chunks...")
        doc_id_prefix = f"doc_{file_path.stem}"
        chunks = self.chunker.chunk_sections(sections, doc_id_prefix)
        logger.info(f"    Created {len(chunks)} chunks")

        return chunks

    def _save_outputs(self, chunks: List):
        """Save all outputs."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save chunks as JSONL
        if 'jsonl' in self.config.output_formats:
            chunks_path = self.config.get_output_path(f"{self.config.corpus_name}.jsonl")
            self._save_chunks_jsonl(chunks, chunks_path)
            logger.info(f"Saved chunks to {chunks_path}")

        # Save metadata
        meta_path = self.config.get_output_path(f"{self.config.corpus_name}_meta.json")
        self._save_metadata(chunks, meta_path)
        logger.info(f"Saved metadata to {meta_path}")

        # Save indexes
        if self.index_builder:
            self.index_builder.save_indexes(
                str(output_dir),
                self.config.corpus_name
            )

        # Save pipeline statistics
        stats_path = self.config.get_output_path(f"{self.config.corpus_name}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")

    def _save_chunks_jsonl(self, chunks: List, output_path: str):
        """Save chunks in JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                chunk_dict = chunk.to_dict()
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')

    def _save_metadata(self, chunks: List, output_path: str):
        """Save corpus metadata."""
        metadata_stats = validate_metadata(chunks)

        metadata = {
            'corpus_name': self.config.corpus_name,
            'num_chunks': len(chunks),
            'total_tokens': sum(c.token_count for c in chunks),
            'textbooks': list(set(c.textbook for c in chunks)),
            'metadata_stats': metadata_stats,
            'config': {
                'embedding_model': self.config.embedding.model,
                'embedding_backend': self.config.embedding.backend,
                'chunk_min_tokens': self.config.chunking.min_tokens,
                'chunk_target_tokens': self.config.chunking.target_tokens,
                'chunk_max_tokens': self.config.chunking.max_tokens,
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def _log_final_stats(self):
        """Log final statistics."""
        logger.info("="*70)
        logger.info("PIPELINE STATISTICS")
        logger.info("="*70)
        logger.info(f"Files processed:    {self.stats['files_processed']}")
        logger.info(f"Total chunks:       {self.stats['total_chunks']}")
        logger.info(f"Total tokens:       {self.stats['total_tokens']:,}")
        logger.info(f"Processing time:    {self.stats.get('processing_time_seconds', 0):.1f}s")

        if self.stats.get('total_chunks'):
            avg_tokens = self.stats['total_tokens'] / self.stats['total_chunks']
            logger.info(f"Avg chunk size:     {avg_tokens:.0f} tokens")

        logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run medical textbook rechunking pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (uses default if not specified)'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Override input directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--corpus_name',
        type=str,
        default=None,
        help='Override corpus name'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default=None,
        help='Override embedding model'
    )

    args = parser.parse_args()

    # Load config
    config = get_config(args.config)

    # Apply overrides
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.corpus_name:
        config.corpus_name = args.corpus_name
    if args.embedding_model:
        config.embedding.model = args.embedding_model

    # Run pipeline
    pipeline = RechunkingPipeline(config)
    stats = pipeline.run()

    # Print summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Output directory: {config.output_dir}")
    print(f"Corpus name: {config.corpus_name}")
    print(f"Total chunks: {stats['total_chunks']}")
    print("="*70)


if __name__ == '__main__':
    main()
