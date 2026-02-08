"""
Metadata extraction and attachment module.

Enriches chunks with medical metadata including content type,
system classification, and keywords.
"""

import re
import logging
from typing import List, Dict, Any, Set, Optional
from collections import Counter

from .config import MetadataConfig
from .semantic_chunker import SemanticChunk


logger = logging.getLogger(__name__)


# Medical system keywords
MEDICAL_SYSTEMS = {
    'cardiovascular': [
        'heart', 'cardiac', 'vascular', 'blood vessel', 'artery', 'vein',
        'circulation', 'coronary', 'myocardial', 'perfusion',
    ],
    'nervous': [
        'brain', 'nerve', 'neural', 'spinal cord', 'cerebral', 'neuron',
        'synapse', 'cortex', 'neurological', ' CNS ', 'central nervous system',
    ],
    'respiratory': [
        'lung', 'pulmonary', 'respiration', 'breath', 'airway', 'alveolar',
        'ventilation', 'oxygenation', 'trachea', 'bronch',
    ],
    'gastrointestinal': [
        'stomach', 'intestine', 'bowel', 'digestion', 'gastro', 'intestinal',
        'colon', 'gastric', 'hepatic', 'liver', 'pancreas',
    ],
    'renal': [
        'kidney', 'renal', 'nephron', 'urine', 'urinary', 'glomerular',
        'filtration', 'dialysis',
    ],
    'endocrine': [
        'hormone', 'endocrine', 'thyroid', 'insulin', 'pituitary', 'adrenal',
        'metabolism', 'glucose',
    ],
    'musculoskeletal': [
        'muscle', 'bone', 'skeletal', 'joint', 'cartilage', 'tendon',
        'ligament', 'fracture', 'orthopedic',
    ],
    'immune': [
        'immune', 'lymphocyte', 'antibody', 'antigen', 'inflammation',
        'infection', 'vaccine', 'immunity', 'lymph',
    ],
    'hematologic': [
        'blood', 'hemoglobin', 'erythrocyte', 'leukocyte', 'platelet',
        'anemia', 'coagulation', 'clotting', 'hematology',
    ],
    'reproductive': [
        'reproductive', 'fertility', 'pregnancy', 'uterus', 'ovary',
        'testis', 'prostate', 'gonad', 'embryo', 'fetus',
    ],
}

# Content type keywords
CONTENT_TYPES = {
    'anatomy': [
        'structure', 'anatomy', 'location', 'origin', 'insertion',
        'nerve supply', 'blood supply', 'lymphatic drainage', 'gross',
        'histology', 'tissue', 'organ', 'system',
    ],
    'physiology': [
        'function', 'physiology', 'mechanism', 'process', 'regulation',
        'metabolism', 'secretion', 'absorption', 'excretion', 'transport',
        'homeostasis', 'normal',
    ],
    'pathology': [
        'disease', 'pathology', 'disorder', 'abnormal', 'injury',
        'tumor', 'cancer', 'malignant', 'benign', 'inflammation',
        'infection', 'degeneration', 'lesion',
    ],
    'pharmacology': [
        'drug', 'medication', 'therapy', 'treatment', 'pharmacological',
        'dose', 'administration', 'contraindication', 'side effect',
        'adverse', 'interaction',
    ],
    'clinical': [
        'symptom', 'sign', 'diagnosis', 'clinical', 'presentation',
        'examination', 'patient', 'case', 'management', 'prognosis',
    ],
}


class MetadataExtractor:
    """
    Extract medical metadata from text chunks.
    """

    def __init__(self, config: MetadataConfig):
        """
        Initialize metadata extractor.

        Args:
            config: Metadata configuration
        """
        self.config = config

    def enrich_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """
        Add metadata to all chunks.

        Args:
            chunks: List of chunks to enrich

        Returns:
            Enriched chunks with metadata
        """
        for chunk in chunks:
            metadata = {}

            # Detect content type
            if self.config.detect_content_type:
                metadata['content_type'] = self._detect_content_type(chunk.text)

            # Detect medical system
            if self.config.detect_medical_system:
                metadata['system'] = self._detect_medical_system(chunk.text)

            # Extract keywords
            if self.config.extract_keywords:
                metadata['keywords'] = self._extract_keywords(chunk.text)

            # Add structural metadata
            metadata['textbook'] = chunk.textbook
            metadata['chapter'] = chunk.chapter
            metadata['section'] = chunk.section

            # Update chunk metadata
            chunk.metadata = metadata

        return chunks

    def _detect_content_type(self, text: str) -> str:
        """
        Detect content type (anatomy, physiology, pathology, etc.).

        Args:
            text: Chunk text

        Returns:
            Content type label
        """
        text_lower = text.lower()
        scores = {}

        for content_type, keywords in CONTENT_TYPES.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[content_type] = score

        if not scores:
            return "general"

        # Return the content type with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def _detect_medical_system(self, text: str) -> Optional[str]:
        """
        Detect which medical system the text refers to.

        Args:
            text: Chunk text

        Returns:
            System label or None
        """
        text_lower = text.lower()
        scores = {}

        for system, keywords in MEDICAL_SYSTEMS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[system] = score

        if not scores:
            return None

        # Return the system with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important medical keywords from text.

        Args:
            text: Chunk text

        Returns:
            List of keywords
        """
        # Simple extraction based on medical terms and word frequency
        words = re.findall(r'\b[A-Z][a-z]+\b', text)

        # Filter common words
        stopwords = {
            'The', 'This', 'That', 'These', 'Those', 'A', 'An',
            'And', 'Or', 'But', 'Is', 'Are', 'Was', 'Were',
            'Be', 'Been', 'Being', 'Have', 'Has', 'Had', 'Do',
            'Does', 'Did', 'Will', 'Would', 'Could', 'Should',
            'May', 'Might', 'Must', 'Can', 'Of', 'In', 'On',
            'At', 'To', 'For', 'From', 'With', 'By', 'As',
        }

        filtered = [w for w in words if w not in stopwords and len(w) > 3]

        # Count frequency
        counter = Counter(filtered)

        # Get top keywords
        top_keywords = [kw for kw, _ in counter.most_common(self.config.max_keywords)]

        return top_keywords


def create_chunk_metadata(chunk: SemanticChunk) -> Dict[str, Any]:
    """
    Create standardized metadata dict for a chunk.

    Args:
        chunk: SemanticChunk object

    Returns:
        Metadata dictionary
    """
    return {
        'chunk_id': chunk.chunk_id,
        'doc_id': chunk.doc_id,
        'textbook': chunk.textbook,
        'chapter': chunk.chapter,
        'section': chunk.section,
        'token_count': chunk.token_count,
        'content_type': chunk.metadata.get('content_type', 'unknown'),
        'system': chunk.metadata.get('system'),
        'keywords': chunk.metadata.get('keywords', []),
    }


def validate_metadata(
    chunks: List[SemanticChunk]
) -> Dict[str, Any]:
    """
    Validate metadata completeness and quality.

    Args:
        chunks: List of chunks with metadata

    Returns:
        Validation results
    """
    stats = {
        'total_chunks': len(chunks),
        'chunks_with_content_type': 0,
        'chunks_with_system': 0,
        'chunks_with_keywords': 0,
        'content_type_distribution': Counter(),
        'system_distribution': Counter(),
    }

    for chunk in chunks:
        if chunk.metadata.get('content_type'):
            stats['chunks_with_content_type'] += 1
            stats['content_type_distribution'][chunk.metadata['content_type']] += 1

        if chunk.metadata.get('system'):
            stats['chunks_with_system'] += 1
            stats['system_distribution'][chunk.metadata['system']] += 1

        if chunk.metadata.get('keywords'):
            stats['chunks_with_keywords'] += 1

    return stats
