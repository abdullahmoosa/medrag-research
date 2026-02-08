"""
Semantic chunking module for medical textbooks.

Creates mechanism-consistent chunks that respect section boundaries
and maintain semantic coherence.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from .config import ChunkingConfig
from .split_structure import TextSection


logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text."""
    chunk_id: str
    doc_id: str
    textbook: str
    chapter: str
    section: str
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'textbook': self.textbook,
            'chapter': self.chapter,
            'section': self.section,
            'text': self.text,
            'token_count': self.token_count,
            'metadata': self.metadata,
        }


class TokenCounter:
    """Count tokens in text using tiktoken or approximate method."""

    def __init__(self, model: str = "cl100k_base"):
        """
        Initialize token counter.

        Args:
            model: Tokenizer model name
        """
        if HAS_TIKTOKEN:
            try:
                self.encoding = tiktoken.get_encoding(model)
                self.use_tiktoken = True
            except Exception:
                self.use_tiktoken = False
                logger.warning("tiktoken available but encoding failed, using approximation")
        else:
            self.use_tiktoken = False
            logger.warning("tiktoken not available, using token approximation")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        if self.use_tiktoken:
            return len(self.encoding.encode(text))
        else:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4


class SemanticChunker:
    """
    Create semantically coherent chunks from structured sections.

    Principles:
    - Chunk size: 120-350 tokens (configurable)
    - Never cross section boundaries (unless allowed)
    - Maintain semantic coherence
    - Avoid mixing unrelated systems
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize semantic chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.token_counter = TokenCounter()

        # Transition words that indicate continuation
        self.continuation_markers = [
            'furthermore', 'moreover', 'additionally', 'also',
            'similarly', 'likewise', 'in addition', 'additionally',
            'for example', 'for instance', 'specifically',
            'in particular', 'notably', 'importantly',
            'consequently', 'therefore', 'thus', 'hence',
            'however', 'nevertheless', 'nonetheless',
            'in contrast', 'conversely', 'alternatively',
        ]

    def chunk_sections(
        self,
        sections: List[TextSection],
        doc_id_prefix: str = "doc"
    ) -> List[SemanticChunk]:
        """
        Chunk structural sections into semantic chunks.

        Args:
            sections: List of TextSection objects
            doc_id_prefix: Prefix for document IDs

        Returns:
            List of SemanticChunk objects
        """
        all_chunks = []
        chunk_counter = 0

        for section in sections:
            # Chunk each section independently (unless cross-section allowed)
            chunks = self._chunk_section(section, doc_id_prefix, chunk_counter)
            all_chunks.extend(chunks)
            chunk_counter += len(chunks)

        # Apply overlap if enabled
        if self.config.enable_overlap:
            logger.info(f"Applying {self.config.overlap_ratio:.0%} overlap between chunks...")
            all_chunks = self._apply_overlap(all_chunks)

        # Filter tiny chunks if enabled
        if self.config.filter_min_tokens > 0:
            before = len(all_chunks)
            all_chunks = [c for c in all_chunks if c.token_count >= self.config.filter_min_tokens]
            filtered = before - len(all_chunks)
            if filtered > 0:
                logger.info(f"Filtered {filtered} chunks below {self.config.filter_min_tokens} tokens")

        # Log statistics
        self._log_chunking_stats(all_chunks)

        return all_chunks

    def _chunk_section(
        self,
        section: TextSection,
        doc_id_prefix: str,
        start_chunk_id: int
    ) -> List[SemanticChunk]:
        """
        Chunk a single section.

        Args:
            section: TextSection to chunk
            doc_id_prefix: Document ID prefix
            start_chunk_id: Starting chunk ID number

        Returns:
            List of SemanticChunk objects
        """
        chunks = []

        # Split section into paragraphs
        paragraphs = self._split_into_paragraphs(section.text)

        if not paragraphs:
            return chunks

        # Group paragraphs into chunks
        current_chunk_text = []
        current_tokens = 0
        chunk_id = start_chunk_id

        for para in paragraphs:
            para_tokens = self.token_counter.count_tokens(para)

            # If paragraph alone exceeds max, split it
            if para_tokens > self.config.max_tokens:
                # Save current chunk
                if current_chunk_text:
                    chunk = self._create_chunk(
                        section,
                        '\n\n'.join(current_chunk_text),
                        current_tokens,
                        chunk_id,
                        doc_id_prefix
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk_text = []
                    current_tokens = 0

                # Split long paragraph
                sub_chunks = self._split_long_paragraph(
                    section, para, chunk_id, doc_id_prefix
                )
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
                continue

            # Check if adding this paragraph would exceed max
            if current_tokens + para_tokens > self.config.max_tokens:
                # Save current chunk if it meets minimum
                if current_tokens >= self.config.min_tokens:
                    chunk = self._create_chunk(
                        section,
                        '\n\n'.join(current_chunk_text),
                        current_tokens,
                        chunk_id,
                        doc_id_prefix
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk_text = []
                    current_tokens = 0

            # Add paragraph to current chunk
            current_chunk_text.append(para)
            current_tokens += para_tokens

            # Check if we've reached target size
            if current_tokens >= self.config.target_tokens:
                chunk = self._create_chunk(
                    section,
                    '\n\n'.join(current_chunk_text),
                    current_tokens,
                    chunk_id,
                    doc_id_prefix
                )
                chunks.append(chunk)
                chunk_id += 1
                current_chunk_text = []
                current_tokens = 0

        # Don't forget the last chunk
        if current_chunk_text and current_tokens >= self.config.min_tokens:
            chunk = self._create_chunk(
                section,
                '\n\n'.join(current_chunk_text),
                current_tokens,
                chunk_id,
                doc_id_prefix
            )
            chunks.append(chunk)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Section text

        Returns:
            List of paragraphs
        """
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)

        # Clean and filter
        cleaned = []
        for para in paragraphs:
            para = para.strip().replace('\n', ' ')
            # Remove empty or very short paragraphs
            if len(para.split()) >= 3:
                cleaned.append(para)

        return cleaned

    def _split_long_paragraph(
        self,
        section: TextSection,
        paragraph: str,
        start_chunk_id: int,
        doc_id_prefix: str
    ) -> List[SemanticChunk]:
        """
        Split a long paragraph into multiple chunks.

        Args:
            section: Source section
            paragraph: Paragraph text
            start_chunk_id: Starting chunk ID
            doc_id_prefix: Document ID prefix

        Returns:
            List of SemanticChunk objects
        """
        chunks = []

        # Try to split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        current_chunk = []
        current_tokens = 0
        chunk_id = start_chunk_id

        for sent in sentences:
            sent_tokens = self.token_counter.count_tokens(sent)

            if current_tokens + sent_tokens > self.config.max_tokens:
                if current_chunk:
                    chunk = self._create_chunk(
                        section,
                        ' '.join(current_chunk),
                        current_tokens,
                        chunk_id,
                        doc_id_prefix
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(sent)
            current_tokens += sent_tokens

        if current_chunk:
            chunk = self._create_chunk(
                section,
                ' '.join(current_chunk),
                current_tokens,
                chunk_id,
                doc_id_prefix
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        section: TextSection,
        text: str,
        token_count: int,
        chunk_id: int,
        doc_id_prefix: str
    ) -> SemanticChunk:
        """
        Create a SemanticChunk object.

        Args:
            section: Source section
            text: Chunk text
            token_count: Number of tokens
            chunk_id: Chunk ID number
            doc_id_prefix: Document ID prefix

        Returns:
            SemanticChunk object
        """
        doc_id = f"{doc_id_prefix}_{section.textbook}_{chunk_id:06d}"

        return SemanticChunk(
            chunk_id=f"{doc_id}",
            doc_id=doc_id,
            textbook=section.textbook,
            chapter=section.chapter,
            section=section.section,
            text=text,
            token_count=token_count,
        )

    def _apply_overlap(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """
        Apply overlap between consecutive chunks within the same section.

        Only applies overlap between chunks from the same section to avoid
        mixing unrelated content.

        Args:
            chunks: List of chunks to add overlap to

        Returns:
            List of chunks with overlap applied
        """
        if not chunks or len(chunks) < 2:
            return chunks

        overlapped_chunks = []
        chunks_by_section = {}

        # Group chunks by section
        for chunk in chunks:
            section_key = (chunk.textbook, chunk.chapter, chunk.section)
            if section_key not in chunks_by_section:
                chunks_by_section[section_key] = []
            chunks_by_section[section_key].append(chunk)

        # Apply overlap within each section
        for section_chunks in chunks_by_section.values():
            for i, chunk in enumerate(section_chunks):
                # Get previous chunk's text for overlap
                if i > 0 and self.config.overlap_strategy == "tokens":
                    prev_chunk = section_chunks[i - 1]

                    # Calculate overlap size in tokens
                    overlap_tokens = int(prev_chunk.token_count * self.config.overlap_ratio)

                    if overlap_tokens > 0:
                        # Get overlap text from end of previous chunk
                        overlap_text = self._get_overlap_text(
                            prev_chunk.text,
                            overlap_tokens,
                            from_end=True
                        )

                        # Prepend overlap to current chunk
                        new_text = overlap_text + "\n\n" + chunk.text
                        new_tokens = self.token_counter.count_tokens(new_text)

                        # Create new chunk with overlap
                        overlapped_chunk = SemanticChunk(
                            chunk_id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            textbook=chunk.textbook,
                            chapter=chunk.chapter,
                            section=chunk.section,
                            text=new_text,
                            token_count=new_tokens,
                            metadata=chunk.metadata.copy() if chunk.metadata else {},
                        )
                        overlapped_chunks.append(overlapped_chunk)
                        continue

                # No overlap or first chunk - keep as-is
                overlapped_chunks.append(chunk)

        logger.info(f"Applied overlap to {len(overlapped_chunks)} chunks")
        return overlapped_chunks

    def _get_overlap_text(
        self,
        text: str,
        target_tokens: int,
        from_end: bool = True
    ) -> str:
        """
        Extract overlap text from chunk.

        Args:
            text: Source text
            target_tokens: Number of tokens to extract
            from_end: If True, extract from end; else from start

        Returns:
            Extracted text
        """
        if self.config.overlap_strategy == "tokens":
            # Split by tokens (words) and take from end/start
            words = text.split()

            if from_end:
                # Take last N words
                overlap_words = words[-target_tokens * 3:] if len(words) > target_tokens * 3 else words
            else:
                # Take first N words
                overlap_words = words[:target_tokens * 3] if len(words) > target_tokens * 3 else words

            return ' '.join(overlap_words)

        elif self.config.overlap_strategy == "sentences":
            # Split by sentences and take complete sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)

            if from_end:
                # Take last sentences until we reach target tokens
                overlap_sentences = []
                total_tokens = 0
                for sent in reversed(sentences):
                    sent_tokens = self.token_counter.count_tokens(sent)
                    if total_tokens + sent_tokens > target_tokens:
                        break
                    overlap_sentences.insert(0, sent)
                    total_tokens += sent_tokens
            else:
                # Take first sentences
                overlap_sentences = []
                total_tokens = 0
                for sent in sentences:
                    sent_tokens = self.token_counter.count_tokens(sent)
                    if total_tokens + sent_tokens > target_tokens:
                        break
                    overlap_sentences.append(sent)
                    total_tokens += sent_tokens

            return ' '.join(overlap_sentences)

        return text

    def _log_chunking_stats(self, chunks: List[SemanticChunk]):
        """Log chunking statistics."""
        if not chunks:
            logger.warning("No chunks created!")
            return

        token_counts = [c.token_count for c in chunks]

        stats = {
            'total_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'chunks_in_range': sum(
                1 for c in chunks
                if self.config.min_tokens <= c.token_count <= self.config.max_tokens
            ),
        }

        logger.info(f"Chunking statistics:")
        logger.info(f"  Total chunks: {stats['total_chunks']}")
        logger.info(f"  Avg tokens: {stats['avg_tokens']:.0f}")
        logger.info(f"  Min tokens: {stats['min_tokens']}")
        logger.info(f"  Max tokens: {stats['max_tokens']}")
        logger.info(f"  Chunks in target range: {stats['chunks_in_range']}/{stats['total_chunks']} "
                   f"({100*stats['chunks_in_range']/stats['total_chunks']:.1f}%)")

    def validate_chunks(
        self,
        chunks: List[SemanticChunk]
    ) -> Dict[str, Any]:
        """
        Validate chunks meet requirements.

        Args:
            chunks: List of chunks to validate

        Returns:
            Validation results
        """
        issues = []

        # Check token limits
        for chunk in chunks:
            if chunk.token_count > self.config.max_tokens:
                issues.append(f"Chunk {chunk.chunk_id} exceeds max tokens: {chunk.token_count}")
            if chunk.token_count < self.config.min_tokens:
                issues.append(f"Chunk {chunk.chunk_id} below min tokens: {chunk.token_count}")

        # Check for section boundary violations
        if not self.config.allow_cross_section:
            # This is already enforced by chunking per section
            pass

        return {
            'valid': len(issues) == 0,
            'total_chunks': len(chunks),
            'issues': issues[:10],  # Limit output
            'num_issues': len(issues),
        }
