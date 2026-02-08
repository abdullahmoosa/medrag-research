#!/usr/bin/env python3
"""
Test script for the rechunking pipeline.

Verifies that all components work correctly with sample data.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")

    try:
        from src.rechunk_pipeline import (
            PipelineConfig,
            TextCleaner,
            StructureSplitter,
            SemanticChunker,
            MetadataExtractor,
            IndexBuilder,
            RechunkingPipeline,
        )
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration system."""
    logger.info("\nTesting configuration...")

    try:
        from src.rechunk_pipeline import PipelineConfig

        config = PipelineConfig()
        assert config.input_dir
        assert config.output_dir
        assert config.chunking.min_tokens > 0
        assert config.embedding.model

        logger.info("✓ Configuration system works")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_cleaning():
    """Test text cleaning."""
    logger.info("\nTesting text cleaning...")

    try:
        from src.rechunk_pipeline import TextCleaner, CleaningConfig

        # Sample text with artifacts
        sample = """
        Fig. 1.1 This is a figure caption.

        This is actual content about anatomy.

        42

        More content here.

        References
        1. Some reference
        """

        cleaner = TextCleaner(CleaningConfig(remove_figures=True, remove_references=True))
        cleaned = cleaner.clean_text(sample)

        # Check that figure reference is removed
        assert "Fig. 1.1" not in cleaned or "figure caption" not in cleaned.lower()

        # Check that page number is removed
        assert "42" not in cleaned or cleaned.strip() == ""

        # Check that references are removed
        assert "References" not in cleaned
        assert "Some reference" not in cleaned

        logger.info("✓ Text cleaning works")
        return True
    except Exception as e:
        logger.error(f"✗ Text cleaning test failed: {e}")
        return False


def test_splitting():
    """Test structural splitting."""
    logger.info("\nTesting structural splitting...")

    try:
        from src.rechunk_pipeline import StructureSplitter

        # Sample text with structure
        sample = """
Chapter 1: Introduction

This is the introduction.

Chapter 2: Methods

This is the methods section.
        """

        splitter = StructureSplitter()
        sections = splitter.split_textbook(sample, "TestBook")

        assert len(sections) >= 1, f"Expected at least 1 section, got {len(sections)}"

        logger.info(f"✓ Structural splitting works ({len(sections)} sections found)")
        return True
    except Exception as e:
        logger.error(f"✗ Structural splitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunking():
    """Test semantic chunking."""
    logger.info("\nTesting semantic chunking...")

    try:
        from src.rechunk_pipeline import SemanticChunker, ChunkingConfig, TextSection

        # Create sample sections
        section = TextSection(
            textbook="TestBook",
            chapter="Chapter 1",
            section="Section 1.1",
            text="This is a test paragraph. " * 50,  # ~400 words
        )

        chunker = SemanticChunker(ChunkingConfig(
            min_tokens=50,
            target_tokens=100,
            max_tokens=150,
        ))

        chunks = chunker.chunk_sections([section], "test")

        assert len(chunks) > 0, "No chunks created"
        assert all(c.token_count > 0 for c in chunks), "Empty chunks found"

        logger.info(f"✓ Semantic chunking works ({len(chunks)} chunks created)")
        return True
    except Exception as e:
        logger.error(f"✗ Semantic chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata():
    """Test metadata extraction."""
    logger.info("\nTesting metadata extraction...")

    try:
        from src.rechunk_pipeline import MetadataExtractor, SemanticChunker, ChunkingConfig, TextSection

        # Create sample chunk with longer text
        section = TextSection(
            textbook="TestBook",
            chapter="Cardiovascular System",
            section="Heart Structure",
            text=("The heart is a muscular organ that pumps blood throughout the body. " +
                   "It is located in the thoracic cavity and is about the size of a fist. " +
                   "The heart consists of four chambers: two atria and two ventricles. " +
                   "The right atrium receives deoxygenated blood from the body through the veins. " +
                   "The right ventricle pumps this blood to the lungs for oxygenation. " +
                   "The left atrium receives oxygenated blood from the lungs. " +
                   "The left ventricle pumps the oxygenated blood to the rest of the body. " +
                   "This continuous circulation is essential for delivering oxygen and nutrients to tissues " +
                   "and removing waste products like carbon dioxide. ") * 3,
        )

        chunker = SemanticChunker(ChunkingConfig(
            min_tokens=50,
            target_tokens=100,
            max_tokens=200,
        ))
        chunks = chunker.chunk_sections([section], "test")

        # Extract metadata
        extractor = MetadataExtractor(
            type('Config', (), {
                'detect_content_type': True,
                'detect_medical_system': True,
                'extract_keywords': True,
                'max_keywords': 10,
            })()
        )

        enriched_chunks = extractor.enrich_chunks(chunks)

        assert len(enriched_chunks) > 0, "No chunks after enrichment"
        assert enriched_chunks[0].metadata, "No metadata extracted"

        logger.info(f"✓ Metadata extraction works")
        logger.info(f"  Content type: {enriched_chunks[0].metadata.get('content_type')}")
        logger.info(f"  System: {enriched_chunks[0].metadata.get('system')}")
        return True
    except Exception as e:
        logger.error(f"✗ Metadata extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_token_counting():
    """Test token counting."""
    logger.info("\nTesting token counting...")

    try:
        from src.rechunk_pipeline.semantic_chunker import TokenCounter

        counter = TokenCounter()

        # Test with known text
        text = "This is a test sentence."  # ~7 tokens
        count = counter.count_tokens(text)

        assert count > 0, "Token count is zero"
        assert 5 <= count <= 15, f"Token count out of expected range: {count}"

        logger.info(f"✓ Token counting works ({count} tokens for test sentence)")
        return True
    except Exception as e:
        logger.error(f"✗ Token counting test failed: {e}")
        return False


def test_with_real_file():
    """Test with a real textbook file (if available)."""
    logger.info("\nTesting with real textbook file...")

    textbook_dir = Path("/home/ser/medrag/data/medQA USMLE/textbooks/en")

    if not textbook_dir.exists():
        logger.warning("Textbook directory not found, skipping real file test")
        return True

    # Find first textbook
    textbook_files = list(textbook_dir.glob("*.txt"))
    if not textbook_files:
        logger.warning("No textbook files found, skipping real file test")
        return True

    file_path = textbook_files[0]
    logger.info(f"Testing with: {file_path.name}")

    try:
        from src.rechunk_pipeline import (
            TextCleaner, StructureSplitter, SemanticChunker,
            MetadataExtractor, CleaningConfig, ChunkingConfig
        )

        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Clean
        cleaner = TextCleaner(CleaningConfig())
        cleaned = cleaner.clean_text(text)
        logger.info(f"  Cleaned: {len(cleaned)} chars")

        # Split
        splitter = StructureSplitter()
        sections = splitter.split_textbook(cleaned, file_path.stem)
        logger.info(f"  Sections: {len(sections)}")

        # Chunk
        chunker = SemanticChunker(ChunkingConfig(
            min_tokens=120,
            target_tokens=220,
            max_tokens=350,
        ))
        chunks = chunker.chunk_sections(sections, file_path.stem)
        logger.info(f"  Chunks: {len(chunks)}")

        # Metadata
        extractor = MetadataExtractor(
            type('Config', (), {
                'detect_content_type': True,
                'detect_medical_system': True,
                'extract_keywords': True,
                'max_keywords': 10,
            })()
        )
        chunks = extractor.enrich_chunks(chunks)

        # Validate
        validation = chunker.validate_chunks(chunks)
        logger.info(f"  Validation: {validation['valid']}")
        logger.info(f"  Issues: {validation['num_issues']}")

        logger.info("✓ Real file processing successful")
        return True

    except Exception as e:
        logger.error(f"✗ Real file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("RECHUNKING PIPELINE TEST SUITE")
    print("="*70)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Text Cleaning", test_cleaning),
        ("Structural Splitting", test_splitting),
        ("Semantic Chunking", test_chunking),
        ("Metadata Extraction", test_metadata),
        ("Token Counting", test_token_counting),
        ("Real File Processing", test_with_real_file),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
