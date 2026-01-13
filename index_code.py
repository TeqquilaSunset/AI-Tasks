#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index project code for RAG-based PR review
"""
from __future__ import annotations

import ast
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
)
from src.utils.logging_config import setup_logging
from src.utils.qdrant_client import QdrantIndexer

log = setup_logging("code-indexer")


def extract_functions_classes(
    file_path: Path, content: str
) -> List[Tuple[str, str, int, int]]:
    """Extract functions and classes from Python code.

    Args:
        file_path: Path to the file
        content: File content

    Returns:
        List of (name, type, start_line, end_line) tuples
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    elements = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            element_type = (
                "class"
                if isinstance(node, ast.ClassDef)
                else "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            )
            elements.append(
                (node.name, element_type, node.lineno, node.end_lineno or node.lineno)
            )

    return elements


def chunk_code_by_functions(
    file_path: Path, content: str
) -> List[Dict[str, any]]:
    """Chunk code by functions and classes.

    Args:
        file_path: Path to the file
        content: File content

    Returns:
        List of chunks with metadata
    """
    lines = content.split("\n")
    elements = extract_functions_classes(file_path, content)

    if not elements:
        # No elements found, return whole file as one chunk
        return [
            {
                "content": content,
                "file": str(file_path),
                "start_line": 1,
                "end_line": len(lines),
                "element_type": "file",
                "element_name": file_path.name,
            }
        ]

    chunks = []

    for name, element_type, start_line, end_line in elements:
        # Extract the code for this element
        element_lines = lines[start_line - 1 : end_line]
        element_code = "\n".join(element_lines)

        # Get docstring if available
        docstring = ""
        try:
            tree = ast.parse("\n".join(element_lines))
            doc_node = tree.body[0] if tree.body else None
            if doc_node and doc_node.body and isinstance(doc_node.body[0], ast.Expr):
                docstring = ast.get_docstring(doc_node) or ""
        except Exception:
            pass

        # Create chunk with docstring and code
        chunk_content = f"# {element_type}: {name}\n"
        if docstring:
            chunk_content += f'"""\n{docstring}\n"""\n\n'
        chunk_content += element_code

        chunks.append(
            {
                "content": chunk_content,
                "file": str(file_path),
                "start_line": start_line,
                "end_line": end_line,
                "element_type": element_type,
                "element_name": name,
                "docstring": docstring,
            }
        )

    return chunks


def index_python_files(
    root_dir: Path,
    indexer: QdrantIndexer,
    embedding_model: str,
    ollama_host: str,
) -> int:
    """Index all Python files in the project.

    Args:
        root_dir: Root directory of the project
        indexer: QdrantIndexer instance
        embedding_model: Name of the embedding model
        ollama_host: Ollama API host

    Returns:
        Number of indexed chunks
    """
    indexed_count = 0
    python_files = list(root_dir.rglob("*.py"))

    # Exclude common directories
    exclude_dirs = {
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        ".pytest_cache",
        "build",
        "dist",
        ".eggs",
        "*.egg-info",
        "node_modules",
    }

    log.info(f"Found {len(python_files)} Python files")

    for file_path in python_files:
        # Skip excluded directories
        if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
            continue

        # Skip virtual environment
        if ".venv" in str(file_path) or "venv" in str(file_path):
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            chunks = chunk_code_by_functions(file_path, content)

            for chunk in chunks:
                # Generate embedding
                embedding = indexer.generate_embedding(chunk["content"])

                # Create metadata
                metadata = {
                    "file": str(chunk["file"]),
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "element_type": chunk["element_type"],
                    "element_name": chunk["element_name"],
                    "docstring": chunk["docstring"][:200] if chunk["docstring"] else "",
                    "indexed_at": datetime.now().isoformat(),
                }

                # Insert into Qdrant
                point_id = f"{file_path}:{chunk['element_name']}:{chunk['start_line']}"
                indexer.upsert(point_id, embedding, chunk["content"], metadata)

                indexed_count += 1

            log.info(
                f"Indexed {len(chunks)} chunks from {file_path.relative_to(root_dir)}"
            )

        except Exception as e:
            log.error(f"Error indexing {file_path}: {e}")
            continue

    return indexed_count


def main() -> None:
    """Main entry point."""
    log.info("Starting code indexing...")

    # Initialize indexer
    indexer = QdrantIndexer(
        host=DEFAULT_QDRANT_HOST,
        port=DEFAULT_QDRANT_PORT,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    # Create collection if not exists
    indexer.create_collection()

    # Get project root
    project_root = Path.cwd()

    # Index Python files
    indexed_count = index_python_files(
        project_root, indexer, DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_HOST
    )

    log.info(f"Code indexing completed: {indexed_count} chunks indexed")


if __name__ == "__main__":
    main()
