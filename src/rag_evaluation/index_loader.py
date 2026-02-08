"""
Index loading and management.

Handles loading BM25, FAISS indexes, and chunk metadata from the processed corpus.
"""

import os
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import faiss

from .chunk_schema import Chunk, SectionMetadata
from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages loading and accessing indexes from the processed corpus.

    Provides unified access to:
    - Chunk data (from JSONL)
    - BM25 index (from pickle)
    - FAISS index (from .index file)
    - Embeddings (from .npy file)
    - Metadata (from JSON)
    """

    def __init__(self, index_dir: Path, corpus_name: str = None):
        """
        Initialize index manager.

        Args:
            index_dir: Directory containing the processed corpus
            corpus_name: Name of the corpus (auto-detected if not specified)
        """
        self.index_dir = Path(index_dir)

        # Auto-detect corpus name if not specified
        if corpus_name is None:
            corpus_name = self._detect_corpus_name()
            logger.info(f"Auto-detected corpus name: {corpus_name}")

        self.corpus_name = corpus_name

        # Index storage
        self._chunks: List[Chunk] = []
        self._chunk_by_id: Dict[str, Chunk] = {}
        self._chunks_by_section: Dict[str, List[Chunk]] = defaultdict(list)
        self._chunks_by_doc: Dict[str, List[Chunk]] = defaultdict(list)

        self._bm25_index: Optional[Any] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._embeddings: Optional[np.ndarray] = None

        self._metadata: Dict[str, Any] = {}
        self._stats: Dict[str, Any] = {}

        self._loaded = False

    def _detect_corpus_name(self) -> str:
        """
        Auto-detect corpus name from index directory.

        Looks for *_meta.json files and extracts the corpus name from the filename.

        Returns:
            Detected corpus name or 'medtextbooks_v1' as fallback
        """
        import glob

        # Try to find meta.json files
        meta_files = list(self.index_dir.glob("*_meta.json"))

        if meta_files:
            # Extract corpus name from first meta file
            # e.g., "medtextbooks_v1_medembed_meta.json" -> "medtextbooks_v1_medembed"
            meta_file = meta_files[0]
            corpus_name = meta_file.stem.replace("_meta", "")
            logger.info(f"Detected corpus name from meta file: {corpus_name}")
            return corpus_name

        # Fallback: look for any .jsonl file
        jsonl_files = list(self.index_dir.glob("*.jsonl"))
        if jsonl_files:
            # e.g., "medtextbooks_v1_medembed.jsonl" -> "medtextbooks_v1_medembed"
            corpus_name = jsonl_files[0].stem
            logger.info(f"Detected corpus name from jsonl file: {corpus_name}")
            return corpus_name

        # Fallback to default
        logger.warning("Could not auto-detect corpus name, using default 'medtextbooks_v1'")
        return "medtextbooks_v1"

    def load(self) -> None:
        """Load all indexes from disk."""
        if self._loaded:
            logger.warning("Indexes already loaded, skipping")
            return

        logger.info(f"Loading indexes from {self.index_dir}...")

        # Load chunks
        self._load_chunks()

        # Load BM25
        self._load_bm25()

        # Load FAISS
        self._load_faiss()

        # Load embeddings
        self._load_embeddings()

        # Load metadata and stats
        self._load_metadata()

        self._loaded = True
        logger.info(
            f"✓ Loaded {len(self._chunks)} chunks, "
            f"{len(self._chunks_by_section)} sections, "
            f"{len(self._chunks_by_doc)} documents"
        )

    def _load_chunks(self) -> None:
        """Load chunks from JSONL file."""
        chunk_file = self.index_dir / f"{self.corpus_name}.jsonl"

        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

        logger.info(f"Loading chunks from {chunk_file}...")
        chunks = []

        with open(chunk_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    chunk = self._parse_chunk(data)
                    chunks.append(chunk)

                    if line_num % 10000 == 0:
                        logger.debug(f"  Loaded {line_num} chunks...")
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        self._chunks = chunks

        # Build indexes
        for chunk in chunks:
            self._chunk_by_id[chunk.chunk_id] = chunk
            self._chunks_by_section[chunk.section_id].append(chunk)
            self._chunks_by_doc[chunk.doc_id].append(chunk)

        logger.info(f"  ✓ Loaded {len(chunks)} chunks")

    def _parse_chunk(self, data: Dict[str, Any]) -> Chunk:
        """Parse chunk data from JSON."""
        # Extract metadata (may be nested or at top level)
        metadata = data.get("metadata", {})

        # Extract section metadata (try metadata field first, then top-level)
        section_meta = SectionMetadata(
            textbook=data.get("textbook", metadata.get("textbook", "")),
            chapter=data.get("chapter", metadata.get("chapter", "")),
            section=data.get("section", metadata.get("section", "")),
            content_type=data.get("content_type", metadata.get("content_type")),
            medical_system=data.get("medical_system", metadata.get("system")),
            keywords=data.get("keywords", metadata.get("keywords", [])),
        )

        # Create chunk
        chunk_id = data.get("chunk_id") or data.get("id", "")
        if not chunk_id:
            # Generate ID from position
            chunk_id = f"{data.get('textbook', 'unknown')}:{data.get('chapter', 'unknown')}:{data.get('chunk_idx', 0)}"

        doc_id = data.get("textbook", "")
        section_id = f"{doc_id}:{data.get('chapter', '')}:{data.get('section', '')}"

        return Chunk(
            chunk_id=chunk_id,
            text=data.get("text", ""),
            token_count=data.get("token_count", 0),
            section_metadata=section_meta,
            doc_id=doc_id,
            section_id=section_id,
        )

    def _load_bm25(self) -> None:
        """Load BM25 index from pickle file."""
        bm25_file = self.index_dir / f"{self.corpus_name}_bm25.pkl"

        if not bm25_file.exists():
            logger.warning(f"BM25 index not found: {bm25_file}")
            return

        logger.info(f"Loading BM25 index from {bm25_file}...")
        with open(bm25_file, "rb") as f:
            data = pickle.load(f)

        # Handle different formats
        if isinstance(data, dict):
            # Check if this is a rank_bm25 format with 'bm25' key
            if 'bm25' in data and hasattr(data['bm25'], 'get_scores'):
                # This is the format used by build_medcorp_index.py
                self._bm25_index = data['bm25']
                logger.info(f"  ✓ Loaded BM25Okapi index")
            else:
                # Legacy format - inefficient wrapper (NOT RECOMMENDED)
                logger.warning("  Detected legacy BM25 index format (dict). This will be VERY SLOW.")
                logger.warning("  Recommendation: Rebuild index with rank_bm25.BM25Okapi for performance.")

                class LegacyBM25Index:
                    """Wrapper for legacy BM25 index saved as dict (INEFFICIENT!)."""

                    def __init__(self, data_dict):
                        from rank_bm25 import BM25Okapi
                        # Try to rebuild a proper BM25 index if we have corpus
                        if 'corpus' in data_dict and data_dict['corpus']:
                            corpus = data_dict['corpus']
                            # Tokenize corpus
                            tokenized_corpus = [doc.lower().split() for doc in corpus]
                            self.bm25 = BM25Okapi(tokenized_corpus)
                            logger.info(f"  ✓ Rebuilt BM25Okapi from corpus ({len(corpus)} docs)")
                        else:
                            # Fall back to slow implementation
                            self.idf = data_dict.get('idf', {})
                            self.doc_lens = data_dict.get('doc_lens', [])
                            self.corpus = data_dict.get('corpus', [])
                            self.k1 = data_dict.get('k1', 1.5)
                            self.b = data_dict.get('b', 0.75)
                            self.bm25 = None

                    def get_scores(self, query_tokens):
                        """Calculate BM25 scores for a query."""
                        if self.bm25 is not None:
                            return self.bm25.get_scores(query_tokens)

                        # Slow fallback
                        import numpy as np
                        scores = np.zeros(len(self.doc_lens))

                        N = len(self.corpus)
                        avg_doc_len = sum(self.doc_lens) / N if N > 0 else 0

                        for token in query_tokens:
                            if token not in self.idf:
                                continue

                            # Calculate score for each document
                            for i, doc_len in enumerate(self.doc_lens):
                                doc = self.corpus[i]
                                doc_tokens = doc.lower().split()

                                # Count term frequency in document
                                tf = doc_tokens.count(token)

                                if tf > 0:
                                    # BM25 formula
                                    numerator = tf * (self.k1 + 1)
                                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_doc_len))
                                    scores[i] += self.idf.get(token, 0.0) * (numerator / denominator)

                        return scores

                self._bm25_index = LegacyBM25Index(data)
                logger.info(f"  ✓ Loaded legacy BM25 index wrapper")
        else:
            # Direct BM25Okapi object
            self._bm25_index = data
            logger.info(f"  ✓ Loaded BM25 index with {len(self._bm25_index.idf)} unique terms")

    def _load_faiss(self) -> None:
        """Load FAISS index from file."""
        faiss_file = self.index_dir / f"{self.corpus_name}_faiss.index"

        if not faiss_file.exists():
            logger.warning(f"FAISS index not found: {faiss_file}")
            return

        logger.info(f"Loading FAISS index from {faiss_file}...")
        self._faiss_index = faiss.read_index(str(faiss_file))

        logger.info(f"  ✓ Loaded FAISS index with {self._faiss_index.ntotal} vectors")

    def _load_embeddings(self) -> None:
        """Load embeddings from numpy file."""
        emb_file = self.index_dir / f"{self.corpus_name}_faiss_embeddings.npy"

        if not emb_file.exists():
            logger.warning(f"Embeddings file not found: {emb_file}")
            return

        logger.info(f"Loading embeddings from {emb_file}...")
        self._embeddings = np.load(emb_file)

        logger.info(f"  ✓ Loaded embeddings: {self._embeddings.shape}")

    def _load_metadata(self) -> None:
        """Load corpus metadata and statistics."""
        meta_file = self.index_dir / f"{self.corpus_name}_meta.json"
        stats_file = self.index_dir / f"{self.corpus_name}_stats.json"

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(f"  ✓ Loaded metadata from {meta_file}")

        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                self._stats = json.load(f)
            logger.info(f"  ✓ Loaded statistics from {stats_file}")

    # Accessor methods

    @property
    def chunks(self) -> List[Chunk]:
        """Get all chunks."""
        return self._chunks

    @property
    def num_chunks(self) -> int:
        """Get total number of chunks."""
        return len(self._chunks)

    @property
    def bm25_index(self) -> Optional[Any]:
        """Get BM25 index."""
        return self._bm25_index

    @property
    def faiss_index(self) -> Optional[faiss.Index]:
        """Get FAISS index."""
        return self._faiss_index

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Get embeddings array."""
        return self._embeddings

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get corpus metadata."""
        return self._metadata

    @property
    def stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        return self._stats

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a specific chunk by ID."""
        return self._chunk_by_id.get(chunk_id)

    def get_chunks_by_section(self, section_id: str) -> List[Chunk]:
        """Get all chunks in a section."""
        return self._chunks_by_section.get(section_id, [])

    def get_chunks_by_doc(self, doc_id: str) -> List[Chunk]:
        """Get all chunks in a document."""
        return self._chunks_by_doc.get(doc_id, [])

    def get_section_ids(self) -> List[str]:
        """Get all unique section IDs."""
        return list(self._chunks_by_section.keys())

    def get_doc_ids(self) -> List[str]:
        """Get all unique document IDs."""
        return list(self._chunks_by_doc.keys())

    def get_sections_summary(self) -> Dict[str, int]:
        """Get summary of chunks per section."""
        return {sid: len(chunks) for sid, chunks in self._chunks_by_section.items()}

    def get_docs_summary(self) -> Dict[str, int]:
        """Get summary of chunks per document."""
        return {doc_id: len(chunks) for doc_id, chunks in self._chunks_by_doc.items()}
