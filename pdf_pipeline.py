#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF pipeline for splitting PDF files into chunks, generating embeddings
using Ollama, and storing indices in Qdrant.

Refactored to use shared utility modules.
"""

import argparse
import os
import sys
from typing import List

import requests

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
)
from src.utils import (
    OllamaEmbeddingGenerator,
    PDFChunker,
    QdrantIndexer,
    setup_logging,
)

log = setup_logging("pdf-pipeline-ollama")


def main(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = 50,
    collection_name: str = "pdf_chunks",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> bool:
    """
    Main pipeline function.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        collection_name: Qdrant collection name
        embedding_model: Ollama embedding model name
        ollama_host: Ollama API URL

    Returns:
        True if successful, False otherwise
    """
    log.info(f"Starting PDF file processing: {pdf_path}")

    # Check Ollama availability
    try:
        response = requests.get(f"{ollama_host}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        available_models = [model["name"] for model in models]

        if embedding_model not in available_models:
            log.warning(
                f"Model {embedding_model} may not be available. Available models: {available_models}"
            )
        else:
            log.info(f"Model {embedding_model} found in Ollama")
    except Exception as e:
        log.error(f"Could not connect to Ollama: {e}")
        return False

    # Step 1: Split PDF into chunks
    log.info("Step 1: Splitting PDF into chunks")
    chunker = PDFChunker(chunk_size=chunk_size, overlap=overlap)
    text = chunker.read_pdf(pdf_path)
    chunks: List[str] = chunker.chunk_text(text)
    log.info(f"Got {len(chunks)} chunks")

    # Step 2: Generate embeddings using Ollama
    log.info("Step 2: Generating embeddings using Ollama")
    embedder = OllamaEmbeddingGenerator(
        model_name=embedding_model, ollama_host=ollama_host
    )

    try:
        embeddings = embedder.generate_batch_embeddings(chunks)
        log.info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        log.error(f"Error generating embeddings: {e}")
        return False

    # Step 3: Save to Qdrant
    log.info("Step 3: Saving to Qdrant")
    indexer = QdrantIndexer(collection_name=collection_name)

    # Determine vector size from first embedding
    if embeddings:
        vector_size = len(embeddings[0])
        if not indexer.create_collection(vector_size=vector_size):
            log.error("Failed to create collection in Qdrant")
            return False
    else:
        log.error("No embeddings generated")
        return False

    success = indexer.store_embeddings(chunks, embeddings, pdf_path)

    if success:
        log.info("Pipeline completed successfully!")
    else:
        log.error("Error writing to Qdrant")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF processing pipeline for Qdrant using Ollama"
    )
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to PDF file")
    parser.add_argument(
        "--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size"
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlap between chunks"
    )
    parser.add_argument(
        "--collection_name", type=str, default="pdf_chunks", help="Qdrant collection name"
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

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        log.error(f"File not found: {args.pdf_path}")
        sys.exit(1)

    success = main(
        args.pdf_path,
        args.chunk_size,
        args.overlap,
        args.collection_name,
        args.embedding_model,
        args.ollama_host,
    )

    sys.exit(0 if success else 1)
