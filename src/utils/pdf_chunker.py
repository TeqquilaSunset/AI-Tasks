# -*- coding: utf-8 -*-
"""PDF text chunking utility."""

import re
from typing import List
import PyPDF2


class PDFChunker:
    """
    Class for splitting PDF files into text chunks.

    Attributes:
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of words to overlap between chunks
    """

    def __init__(self, chunk_size: int = 1024, overlap: int = 50) -> None:
        """
        Initialize the PDF chunker.

        Args:
            chunk_size: Maximum size of each chunk (default: 1024)
            overlap: Number of words to overlap between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def read_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Remove extra whitespace
            sentence = sentence.strip()

            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap if configured
                if self.overlap > 0:
                    words = current_chunk.split()
                    overlap_text = (
                        ' '.join(words[-self.overlap:])
                        if len(words) > self.overlap
                        else current_chunk
                    )
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Add the last chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
