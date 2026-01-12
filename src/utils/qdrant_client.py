# -*- coding: utf-8 -*-
"""Qdrant vector database client."""

import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .logging_config import setup_logging


class QdrantIndexer:
    """
    Client for working with Qdrant vector database.

    Handles collection creation, embedding storage, and similarity search.

    Attributes:
        host: Qdrant host address
        port: Qdrant port
        collection_name: Name of the collection to work with
        client: QdrantClient instance
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "pdf_chunks",
        logger_name: str = "qdrant-indexer",
    ) -> None:
        """
        Initialize the Qdrant indexer.

        Args:
            host: Qdrant host address
            port: Qdrant port
            collection_name: Name of the collection
            logger_name: Name for the logger
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.log = setup_logging(logger_name)

    def create_collection(self, vector_size: int = 384) -> bool:
        """
        Create a collection in Qdrant if it doesn't exist.

        Args:
            vector_size: Size of the embedding vectors

        Returns:
            True if collection exists or was created successfully, False otherwise
        """
        try:
            collections = self.client.get_collections()

            # Check if collection exists
            collection_exists = any(
                c.name == self.collection_name for c in collections.collections
            )

            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                self.log.info(f"Created collection '{self.collection_name}'")
            else:
                self.log.info(f"Collection '{self.collection_name}' already exists")

            return True
        except Exception as e:
            self.log.error(f"Error creating collection: {e}")
            return False

    def store_embeddings(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        pdf_path: str,
        metadata_list: Optional[List[Dict]] = None,
        batch_size: int = 100,
    ) -> bool:
        """
        Store embeddings and chunks in Qdrant.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            pdf_path: Path to the source PDF file
            metadata_list: Optional additional metadata for each chunk
            batch_size: Number of points to upload per batch

        Returns:
            True if storage was successful, False otherwise
        """
        try:
            points = []
            source_document = os.path.basename(pdf_path)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "chunk_id": i,
                    "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "full_text": chunk,
                    "source_document": source_document,
                }

                # Add additional metadata if provided
                if metadata_list and i < len(metadata_list):
                    metadata.update(metadata_list[i])

                point = PointStruct(
                    id=str(uuid.uuid4()), vector=embedding, payload=metadata
                )
                points.append(point)

            # Upload in batches
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                self.log.info(
                    f"Saved {len(batch)} points to collection '{self.collection_name}' "
                    f"from '{source_document}' ({i + batch_size}/{len(points)})"
                )

            return True
        except Exception as e:
            self.log.error(f"Error storing embeddings in Qdrant: {e}")
            return False

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents by embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of similar documents with scores and metadata
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
            )

            similar_docs = []
            for result in results:
                doc_info = {
                    "score": result.score,
                    "payload": result.payload,
                    "text": result.payload.get("text", ""),
                    "full_text": result.payload.get("full_text", ""),
                }
                similar_docs.append(doc_info)

            return similar_docs
        except Exception as e:
            self.log.error(f"Error searching similar documents: {e}")
            return []

    def query_points(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Any]:
        """
        Query points using the query_points API (supports newer Qdrant versions).

        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return

        Returns:
            List of query result points
        """
        try:
            from qdrant_client.http.models import QueryResponse

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
            )

            # Handle different response formats
            if isinstance(results, QueryResponse):
                return results.points
            elif isinstance(results, tuple) and len(results) > 0:
                return results[0]
            elif hasattr(results, "__iter__") and not isinstance(
                results, (str, bytes)
            ):
                return list(results)
            else:
                return []

        except Exception as e:
            self.log.error(f"Error querying points: {e}")
            return []
