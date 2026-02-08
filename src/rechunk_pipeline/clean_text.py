"""
Text cleaning module for medical textbooks.

Removes figures, page numbers, references, and other artifacts from textbook content.
"""

import re
import logging
from typing import List, Dict, Any
from pathlib import Path

from .config import CleaningConfig


logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Clean raw textbook text by removing figures, page numbers, references, etc.
    """

    def __init__(self, config: CleaningConfig):
        """
        Initialize text cleaner.

        Args:
            config: Cleaning configuration
        """
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.figure_regex = re.compile(
            '|'.join(self.config.figure_patterns),
            re.IGNORECASE | re.MULTILINE
        )

        self.page_regex = re.compile(
            '|'.join(self.config.page_patterns),
            re.MULTILINE
        )

        self.reference_regex = re.compile(
            '|'.join(self.config.reference_patterns),
            re.IGNORECASE
        )

        # Additional patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # Table continuation markers
        self.table_continuation_pattern = re.compile(
            r'^\s*[…\-]+\s*$',
            re.MULTILINE
        )

        # Copyright notices
        self.copyright_pattern = re.compile(
            r'©.*$\n?|Copyright.*$\n?',
            re.MULTILINE
        )

    def clean_text(self, text: str) -> str:
        """
        Clean text according to configuration.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        cleaned = text

        # Remove figures
        if self.config.remove_figures:
            cleaned = self._remove_figures(cleaned)

        # Remove page numbers
        if self.config.remove_page_numbers:
            cleaned = self._remove_page_numbers(cleaned)

        # Remove references
        if self.config.remove_references:
            cleaned = self._remove_references(cleaned)

        # Remove table continuations
        if self.config.remove_table_continuations:
            cleaned = self._remove_table_continuations(cleaned)

        # Remove URLs and emails
        cleaned = self._remove_urls_and_emails(cleaned)

        # Remove copyright notices
        cleaned = self._remove_copyright(cleaned)

        # Clean up whitespace
        cleaned = self._clean_whitespace(cleaned)

        return cleaned

    def _remove_figures(self, text: str) -> str:
        """Remove figure references and captions."""
        # Remove figure reference lines
        text = self.figure_regex.sub('', text)

        # Remove figure captions (typically on their own line or starting sentences)
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip lines that are clearly figure captions
            line_lower = line.lower().strip()
            if any(phrase in line_lower for phrase in [
                'figure ', 'fig. ', 'shows ', 'illustrates ', 'demonstrates '
            ]) and any(word in line_lower for word in [
                'figure', 'fig', 'panel', 'image', 'drawing'
            ]):
                # Keep it if it has substantial content beyond the figure reference
                if len(line.split()) > 15:  # Has more content
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _remove_page_numbers(self, text: str) -> str:
        """Remove page numbers and page headers."""
        # Remove standalone page numbers
        text = self.page_regex.sub('', text)

        # Remove lines that are just numbers
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip lines that are just numbers or very short numeric content
            if stripped and (stripped.isdigit() or (
                len(stripped) < 10 and stripped.replace(' ', '').replace('-', '').isdigit()
            )):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _remove_references(self, text: str) -> str:
        """
        Remove reference sections from text.

        This detects the start of a references section and removes everything after it.
        """
        lines = text.split('\n')
        cleaned_lines = []
        in_references = False

        for line in lines:
            if self.reference_regex.match(line.strip()):
                in_references = True
                continue

            if not in_references:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _remove_table_continuations(self, text: str) -> str:
        """Remove table continuation markers."""
        return self.table_continuation_pattern.sub('', text)

    def _remove_urls_and_emails(self, text: str) -> str:
        """Remove URLs and email addresses."""
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        return text

    def _remove_copyright(self, text: str) -> str:
        """Remove copyright notices."""
        return self.copyright_pattern.sub('', text)

    def _clean_whitespace(self, text: str) -> str:
        """Clean up excessive whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]

        return '\n'.join(lines).strip()

    def clean_paragraph(self, paragraph: str) -> str:
        """
        Clean a single paragraph.

        Args:
            paragraph: Paragraph text

        Returns:
            Cleaned paragraph
        """
        if not paragraph or not paragraph.strip():
            return ""

        cleaned = self.clean_text(paragraph)

        # Remove paragraphs that are too short after cleaning (likely artifacts)
        if len(cleaned.split()) < 3:
            return ""

        return cleaned

    def get_statistics(self, original: str, cleaned: str) -> Dict[str, Any]:
        """
        Get cleaning statistics.

        Args:
            original: Original text
            cleaned: Cleaned text

        Returns:
            Dictionary with statistics
        """
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'original_words': len(original.split()),
            'cleaned_words': len(cleaned.split()),
            'chars_removed': len(original) - len(cleaned),
            'words_removed': len(original.split()) - len(cleaned.split()),
            'reduction_ratio': 1 - (len(cleaned) / len(original)) if len(original) > 0 else 0,
        }


def clean_text_file(
    file_path: Path,
    cleaner: TextCleaner
) -> tuple[str, Dict[str, Any]]:
    """
    Clean a textbook file.

    Args:
        file_path: Path to textbook file
        cleaner: TextCleaner instance

    Returns:
        Tuple of (cleaned_text, statistics)
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        original_text = f.read()

    cleaned_text = cleaner.clean_text(original_text)
    stats = cleaner.get_statistics(original_text, cleaned_text)

    logger.info(f"Cleaned {file_path.name}: "
                f"{stats['words_removed']} words removed "
                f"({stats['reduction_ratio']:.1%} reduction)")

    return cleaned_text, stats
