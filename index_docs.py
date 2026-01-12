#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Documentation indexing pipeline for RAG.

Indexes project documentation files (README, API docs, code files)
into Qdrant for use with the /help command.
"""

import argparse
import asyncio
import os
import sys
import time
from typing import List

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_OLLAMA_HOST,
)
from src.utils import (
    OllamaEmbeddingGenerator,
    QdrantIndexer,
    setup_logging,
)
from src.utils.docs_chunker import (
    DocumentationChunker,
    scan_project_docs,
)

log = setup_logging("docs-indexer")


async def index_documentation(
    project_root: str = ".",
    collection_name: str = "project_docs",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    file_extensions: List[str] = None,
) -> bool:
    """
    Index project documentation into Qdrant.

    Args:
        project_root: Root directory of the project
        collection_name: Qdrant collection name
        embedding_model: Ollama embedding model
        ollama_host: Ollama API URL
        max_concurrent: Maximum concurrent requests
        chunk_size: Text chunk size
        file_extensions: File extensions to index

    Returns:
        True if successful, False otherwise
    """
    if file_extensions is None:
        file_extensions = [".md", ".txt", ".py", ".yml", ".yaml", ".json"]

    log.info(f"Starting documentation indexing for: {project_root}")

    # Scan for documentation files
    log.info("Scanning for documentation files...")
    docs_files = scan_project_docs(project_root, extensions=file_extensions)
    log.info(f"Found {len(docs_files)} documentation files")

    if not docs_files:
        log.warning("No documentation files found!")
        return False

    # Initialize chunker and embedder
    chunker = DocumentationChunker(chunk_size=chunk_size, overlap=50)
    embedder = OllamaEmbeddingGenerator(
        model_name=embedding_model,
        ollama_host=ollama_host,
        max_concurrent=max_concurrent,
    )

    # Initialize Qdrant indexer
    indexer = QdrantIndexer(collection_name=collection_name)

    # Collect all chunks with metadata
    all_chunks = []
    all_metadata = []

    for file_path in docs_files:
        try:
            log.info(f"Processing: {file_path}")
            content = chunker.read_file(file_path)
            chunks = chunker.chunk_text(content, file_path)

            rel_path = os.path.relpath(file_path, project_root)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": rel_path,
                    "chunk_index": i,
                    "file_type": os.path.splitext(rel_path)[1],
                })

            log.info(f"  -> {len(chunks)} chunks from {rel_path}")

        except Exception as e:
            log.error(f"Error processing {file_path}: {e}")
            continue

    if not all_chunks:
        log.error("No chunks generated!")
        return False

    log.info(f"Total chunks to index: {len(all_chunks)}")

    # Generate embeddings (async for speed)
    log.info("Generating embeddings...")
    start_time = time.time()

    try:
        embeddings = await embedder.generate_batch_embeddings_async(all_chunks)
        elapsed_time = time.time() - start_time
        log.info(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f} seconds")
    except Exception as e:
        log.error(f"Error generating embeddings: {e}")
        return False

    # Create collection and store in Qdrant
    log.info(f"Creating/updating collection: {collection_name}")
    vector_size = len(embeddings[0])
    if not indexer.create_collection(vector_size=vector_size):
        log.error("Failed to create collection in Qdrant")
        return False

    log.info("Storing embeddings in Qdrant...")
    # Use a dummy PDF path (will be handled via metadata)
    success = indexer.store_embeddings(
        all_chunks,
        embeddings,
        "project_docs",  # Dummy path, actual paths in metadata
        metadata_list=all_metadata,
    )

    if success:
        log.info("Documentation indexing completed successfully!")
        log.info(f"Collection: {collection_name}")
        log.info(f"Documents indexed: {len(docs_files)}")
        log.info(f"Total chunks: {len(all_chunks)}")
    else:
        log.error("Error storing embeddings in Qdrant")

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index project documentation for RAG-powered /help command"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="project_docs",
        help="Qdrant collection name (default: project_docs)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size (default: 1024)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Ollama embedding model",
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama API URL",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum concurrent requests",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".md", ".txt", ".py", ".yml", ".yaml", ".json"],
        help="File extensions to index",
    )

    args = parser.parse_args()

    result = asyncio.run(index_documentation(
        project_root=args.project_root,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
        max_concurrent=args.max_concurrent,
        chunk_size=args.chunk_size,
        file_extensions=args.extensions,
    ))

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
