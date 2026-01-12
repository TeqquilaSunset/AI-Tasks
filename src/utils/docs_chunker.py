# -*- coding: utf-8 -*-
"""Documentation chunking utility for project files.

Supports chunking of markdown, text, and code files with
appropriate handling of code blocks and markdown structure.
"""

import os
import re
from typing import List, Optional
from pathlib import Path


class DocumentationChunker:
    """
    Class for splitting documentation and code files into chunks.

    Handles markdown files, text files, and code files with
    appropriate section detection and code block preservation.

    Attributes:
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of words to overlap between chunks
        preserve_code_blocks: Whether to keep code blocks intact
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        overlap: int = 50,
        preserve_code_blocks: bool = True,
    ) -> None:
        """
        Initialize the documentation chunker.

        Args:
            chunk_size: Maximum size of each chunk (default: 1024)
            overlap: Number of words to overlap between chunks (default: 50)
            preserve_code_blocks: Keep code blocks intact (default: True)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_code_blocks = preserve_code_blocks

    def read_file(self, file_path: str) -> str:
        """
        Read content from a file.

        Args:
            file_path: Path to the file

        Returns:
            File content as string
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_markdown_sections(self, text: str) -> List[tuple[str, str]]:
        """
        Extract sections from markdown text.

        Args:
            text: Markdown text

        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []
        current_section = "Introduction"
        current_content = []

        # Split into lines
        lines = text.split("\n")

        for line in lines:
            # Check for markdown headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Save previous section
                if current_content:
                    sections.append((current_section, "\n".join(current_content)))
                # Start new section
                current_section = header_match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # Add last section
        if current_content:
            sections.append((current_section, "\n".join(current_content)))

        return sections

    def _extract_code_blocks(self, text: str) -> List[str]:
        """
        Extract code blocks from markdown text.

        Args:
            text: Markdown text

        Returns:
            List of code blocks with their language tags
        """
        code_blocks = []
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        for lang, code in matches:
            code_blocks.append(f"```{lang}\n{code}\n```")

        return code_blocks

    def chunk_text(self, text: str, file_path: Optional[str] = None) -> List[str]:
        """
        Split text into chunks with overlap.

        For markdown files, attempts to preserve section boundaries.
        For code files, attempts to preserve function/class boundaries.

        Args:
            text: Input text to chunk
            file_path: Optional file path for context-aware chunking

        Returns:
            List of text chunks with metadata
        """
        # Determine file type
        file_ext = ""
        if file_path:
            file_ext = Path(file_path).suffix.lower()

        # Special handling for markdown files
        if file_ext in [".md", ".markdown"]:
            return self._chunk_markdown(text, file_path)

        # Special handling for code files
        if file_ext in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"]:
            return self._chunk_code(text, file_path)

        # Default chunking for plain text
        return self._chunk_plain_text(text)

    def _chunk_markdown(self, text: str, file_path: Optional[str]) -> List[str]:
        """Chunk markdown text preserving sections."""
        # Extract sections
        sections = self._extract_markdown_sections(text)

        chunks = []
        current_chunk = ""
        source_file = os.path.basename(file_path) if file_path else "unknown"

        for section_title, section_content in sections:
            section_text = f"# {section_title}\n\n{section_content}"

            # Check if section fits in current chunk
            if len(current_chunk) + len(section_text) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start new chunk (with overlap if possible)
                if self.overlap > 0 and chunks:
                    words = chunks[-1].split()
                    overlap_text = (
                        " ".join(words[-self.overlap:]) if len(words) > self.overlap else chunks[-1]
                    )
                    current_chunk = overlap_text + "\n\n" + section_text
                else:
                    current_chunk = section_text
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section_text
                else:
                    current_chunk = section_text

        # Add last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_code(self, text: str, file_path: Optional[str]) -> List[str]:
        """Chunk code text preserving function/class boundaries."""
        chunks = []
        lines = text.split("\n")

        current_chunk = []
        current_size = 0
        source_file = os.path.basename(file_path) if file_path else "unknown"

        # Pattern to detect function/class definitions
        definition_pattern = re.compile(
            r"^\s*(def|class|function|interface|type)\s+\w+"
        )

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # Check if this is a definition and current chunk is large
            if definition_pattern.match(line) and current_size > self.chunk_size // 2:
                # Save current chunk
                if current_chunk:
                    chunks.append("\n".join(current_chunk))

                # Start new chunk
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

                # Check if we've exceeded chunk size
                if current_size >= self.chunk_size:
                    chunks.append("\n".join(current_chunk))

                    # Start new chunk with overlap
                    if self.overlap > 0:
                        overlap_lines = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                        current_chunk = overlap_lines
                        current_size = sum(len(l) + 1 for l in current_chunk)
                    else:
                        current_chunk = []
                        current_size = 0

        # Add remaining lines
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return [c for c in chunks if c.strip()]

    def _chunk_plain_text(self, text: str) -> List[str]:
        """Chunk plain text with overlap."""
        sentences = re.split(r"[.!?]+\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                if self.overlap > 0:
                    words = current_chunk.split()
                    overlap_text = (
                        " ".join(words[-self.overlap:])
                        if len(words) > self.overlap
                        else current_chunk
                    )
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


def scan_project_docs(
    project_root: str,
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> List[str]:
    """
    Scan project for documentation files.

    Args:
        project_root: Root directory of the project
        extensions: File extensions to include (default: .md, .txt, .py, .js, .ts, .json, .yaml, .yml)
        exclude_dirs: Directories to exclude (default: __pycache__, .git, node_modules, venv)

    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = [".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".rst"]

    if exclude_dirs is None:
        exclude_dirs = ["__pycache__", ".git", "node_modules", "venv", ".venv", "saves", ".pytest_cache"]

    docs_files = []
    project_path = Path(project_root)

    for ext in extensions:
        for file_path in project_path.rglob(f"*{ext}"):
            # Check if file should be excluded
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue

            docs_files.append(str(file_path))

    return sorted(docs_files)
