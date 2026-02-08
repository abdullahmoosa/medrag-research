"""
Structural splitting module for medical textbooks.

Splits textbooks into chapters and sections based on headers and structure.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class TextSection:
    """Represents a section of text with metadata."""
    textbook: str
    chapter: str
    section: str
    text: str
    chapter_level: int = 1
    section_level: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'textbook': self.textbook,
            'chapter': self.chapter,
            'section': self.section,
            'chapter_level': self.chapter_level,
            'section_level': self.section_level,
            'text': self.text,
        }


class StructureSplitter:
    """
    Split textbooks into structural units (chapters, sections).
    """

    def __init__(self, preserve_structure: bool = True):
        """
        Initialize structure splitter.

        Args:
            preserve_structure: Whether to maintain chapter/section hierarchy
        """
        self.preserve_structure = preserve_structure

        # Header patterns (order matters - more specific first)
        self.chapter_patterns = [
            r'^(Chapter\s+[\dIVX]+[\.:]?\s+.+)$',  # Chapter 1: Title
            r'^(CHAPTER\s+[\dIVX]+[\.:]?\s+.+)$',
            r'^([IVX]+[\.\:]\s+.+)$',  # Roman numerals
            r'^(\d+[\.\:]\s+.+)$',  # 1. Title or 1: Title
        ]

        self.section_patterns = [
            r'^([A-Z][A-Z\s]{5,})$',  # ALL CAPS headers
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+.*)$',  # Title Case
            r'^(\d+\.\d+\s+.+)$',  # 1.1 Subsection
            r'^(\d+\.\d+\.\d+\s+.+)$',  # 1.1.1 Sub-subsection
        ]

        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns."""
        self.chapter_regex = re.compile(
            '|'.join(f'({p})' for p in self.chapter_patterns),
            re.MULTILINE
        )
        self.section_regex = re.compile(
            '|'.join(f'({p})' for p in self.section_patterns),
            re.MULTILINE
        )

    def split_textbook(
        self,
        text: str,
        textbook_name: str
    ) -> List[TextSection]:
        """
        Split textbook into chapters and sections.

        Args:
            text: Full textbook text
            textbook_name: Name of the textbook

        Returns:
            List of TextSection objects
        """
        sections = []

        # Split into chapters first
        chapters = self._split_into_chapters(text)

        if not chapters:
            # No chapter structure found, treat entire text as one chapter
            chapters = [(textbook_name, text)]

        for chapter_title, chapter_text in chapters:
            # Split each chapter into sections
            chapter_sections = self._split_into_sections(
                chapter_text,
                textbook_name,
                chapter_title
            )
            sections.extend(chapter_sections)

        logger.info(f"Split {textbook_name} into {len(chapters)} chapters "
                    f"and {len(sections)} sections")

        return sections

    def _split_into_chapters(self, text: str) -> List[tuple[str, str]]:
        """
        Split text into chapters.

        Args:
            text: Full text

        Returns:
            List of (chapter_title, chapter_text) tuples
        """
        chapters = []
        lines = text.split('\n')

        current_chapter = "Introduction"
        current_chapter_text = []

        for line in lines:
            # Check if this line is a chapter header
            is_chapter = False
            for pattern in self.chapter_patterns:
                if re.match(pattern, line.strip()):
                    # Save previous chapter
                    if current_chapter_text:
                        chapter_text = '\n'.join(current_chapter_text).strip()
                        if chapter_text:
                            chapters.append((current_chapter, chapter_text))

                    # Start new chapter
                    current_chapter = line.strip()
                    current_chapter_text = []
                    is_chapter = True
                    break

            if not is_chapter:
                current_chapter_text.append(line)

        # Don't forget the last chapter
        if current_chapter_text:
            chapter_text = '\n'.join(current_chapter_text).strip()
            if chapter_text:
                chapters.append((current_chapter, chapter_text))

        return chapters

    def _split_into_sections(
        self,
        chapter_text: str,
        textbook_name: str,
        chapter_title: str
    ) -> List[TextSection]:
        """
        Split chapter into sections.

        Args:
            chapter_text: Chapter text
            textbook_name: Textbook name
            chapter_title: Chapter title

        Returns:
            List of TextSection objects
        """
        sections = []
        lines = chapter_text.split('\n')

        current_section = "Introduction"
        current_section_text = []
        current_section_level = 2

        for line in lines:
            # Check if this line is a section header
            is_section, section_level = self._is_section_header(line)

            if is_section:
                # Save previous section
                if current_section_text:
                    section_text = '\n'.join(current_section_text).strip()
                    if section_text:
                        section = TextSection(
                            textbook=textbook_name,
                            chapter=chapter_title,
                            section=current_section,
                            text=section_text,
                            section_level=current_section_level
                        )
                        sections.append(section)

                # Start new section
                current_section = line.strip()
                current_section_text = []
                current_section_level = section_level
            else:
                current_section_text.append(line)

        # Don't forget the last section
        if current_section_text:
            section_text = '\n'.join(current_section_text).strip()
            if section_text:
                section = TextSection(
                    textbook=textbook_name,
                    chapter=chapter_title,
                    section=current_section,
                    text=section_text,
                    section_level=current_section_level
                )
                sections.append(section)

        return sections

    def _is_section_header(self, line: str) -> tuple[bool, int]:
        """
        Check if line is a section header.

        Args:
            line: Line to check

        Returns:
            Tuple of (is_header, level)
        """
        stripped = line.strip()

        # Empty line or too long
        if not stripped or len(stripped) > 100:
            return False, 0

        # Check against section patterns
        for i, pattern in enumerate(self.section_patterns):
            if re.match(pattern, stripped):
                return True, i + 2  # Level 2 or higher

        # Check if it's ALL CAPS (common header style)
        if stripped.isupper() and len(stripped.split()) >= 2 and len(stripped) < 80:
            return True, 2

        # Check if it's Title Case with reasonable length
        if (stripped.istitle() and
            len(stripped.split()) >= 3 and
            len(stripped) < 80 and
            not any(char.isdigit() for char in stripped)):
            # Make sure it's not just a regular sentence
            if not stripped.endswith(('.', ',', ';', ':')):
                return True, 3

        return False, 0

    def split_paragraphs(self, section: TextSection) -> List[str]:
        """
        Split section text into paragraphs.

        Args:
            section: TextSection object

        Returns:
            List of paragraph strings
        """
        paragraphs = []

        # Split by double newlines
        raw_paragraphs = section.text.split('\n\n')

        for para in raw_paragraphs:
            # Clean up the paragraph
            cleaned = para.strip().replace('\n', ' ')

            # Remove empty paragraphs
            if not cleaned or len(cleaned.split()) < 3:
                continue

            # Remove artifacts
            if cleaned.isdigit() or re.match(r'^[\W\d]+$', cleaned):
                continue

            paragraphs.append(cleaned)

        return paragraphs


def process_textbook_file(
    file_path: Path,
    text: str,
    splitter: StructureSplitter
) -> List[TextSection]:
    """
    Process a textbook file into structural sections.

    Args:
        file_path: Path to textbook file
        text: Cleaned text content
        splitter: StructureSplitter instance

    Returns:
        List of TextSection objects
    """
    # Extract textbook name from filename
    textbook_name = file_path.stem

    # Split into sections
    sections = splitter.split_textbook(text, textbook_name)

    logger.info(f"Processed {textbook_name}: {len(sections)} sections")

    return sections
