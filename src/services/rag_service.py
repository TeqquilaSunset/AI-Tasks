# -*- coding: utf-8 -*-
"""RAG (Retrieval Augmented Generation) service."""

from typing import Any, Dict, List

import requests

from ..config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
    DEFAULT_RELEVANCE_THRESHOLD,
)
from ..utils.logging_config import setup_logging
from ..utils.qdrant_client import QdrantIndexer


class RAGService:
    """
    Service for Retrieval Augmented Generation (RAG).

    Provides document search and context enhancement for LLM queries
    using vector embeddings and similarity search.

    Attributes:
        collection_name: Name of the Qdrant collection
        embedding_model: Name of the Ollama embedding model
        ollama_host: URL of the Ollama API
        qdrant_client: QdrantIndexer instance
        reranker_enabled: Whether reranking is enabled
        relevance_threshold: Minimum relevance score for results
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        qdrant_host: str = DEFAULT_QDRANT_HOST,
        qdrant_port: int = DEFAULT_QDRANT_PORT,
    ) -> None:
        """
        Initialize the RAG service.

        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Name of the Ollama embedding model
            ollama_host: URL of the Ollama API
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.qdrant_indexer = QdrantIndexer(
            host=qdrant_host, port=qdrant_port, collection_name=collection_name
        )
        self.reranker_enabled = False
        self.relevance_threshold = DEFAULT_RELEVANCE_THRESHOLD
        self.log = setup_logging("rag-service")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Ollama.

        Args:
            text: Input text to embed

        Returns:
            List of embedding values

        Raises:
            ValueError: If Ollama doesn't return an embedding
            requests.RequestException: If the request fails
        """
        try:
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                headers={"Content-Type": "application/json"},
                json={"model": self.embedding_model, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            embedding = result.get("embedding")

            if embedding is None:
                raise ValueError("Ollama did not return an embedding")

            return embedding
        except requests.RequestException as e:
            self.log.error(f"Error requesting Ollama: {e}")
            raise
        except Exception as e:
            self.log.error(f"Error generating embedding: {e}")
            raise

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents by query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of similar documents with scores and metadata
        """
        try:
            self.log.info(
                f"RAG: Sending query to vector database - Query: '{query[:50]}...', top_k: {top_k}"
            )

            query_embedding = self.generate_embedding(query)
            self.log.info(
                f"RAG: Generated embedding of length {len(query_embedding) if query_embedding else 0} for query"
            )

            points = self.qdrant_indexer.query_points(query_embedding, limit=top_k)
            self.log.info(f"RAG: Received {len(points)} results from vector database")

            similar_docs = []
            for i, result in enumerate(points):
                score = getattr(result, "score", 0)
                payload = getattr(result, "payload", {})

                doc_info = {
                    "score": score,
                    "payload": payload,
                    "text": payload.get("text", "") if isinstance(payload, dict) else "",
                    "full_text": (
                        payload.get("full_text", "") if isinstance(payload, dict) else ""
                    ),
                    "source_document": (
                        payload.get("source_document", "")
                        if isinstance(payload, dict)
                        else ""
                    ),
                }

                source_info = (
                    f" (Source: {doc_info['source_document']})"
                    if doc_info["source_document"]
                    else ""
                )
                self.log.info(
                    f"RAG: Result #{i + 1} - Relevance: {score:.3f}, Text: '{doc_info['text'][:100]}...'{source_info}"
                )
                similar_docs.append(doc_info)

            # Apply reranker if enabled
            if self.reranker_enabled and similar_docs:
                self.log.info(f"RAG: Applying reranker to {len(similar_docs)} documents")
                similar_docs = self.rerank_documents(query, similar_docs)
                self.log.info(f"RAG: After reranking, {len(similar_docs)} documents remain")

            # Apply relevance threshold filter regardless of reranker
            filtered_docs = [
                doc for doc in similar_docs if doc["score"] >= self.relevance_threshold
            ]
            self.log.info(
                f"RAG: After filtering by threshold {self.relevance_threshold}, "
                f"{len(filtered_docs)} documents remain"
            )

            return filtered_docs
        except Exception as e:
            self.log.error(f"Error searching similar documents: {e}")
            return []

    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank documents using a simplified algorithm.

        Args:
            query: Original query text
            documents: List of documents to rerank

        Returns:
            Reranked and filtered list of documents
        """
        try:
            self.log.info(f"RAG: Running reranker for {len(documents)} documents")

            reranked_docs = []
            for doc in documents:
                original_score = doc["score"]
                text_content = (
                    doc.get("text", "")
                    or doc.get("full_text", "")
                    or str(doc.get("payload", ""))
                )

                # Simple reranking algorithm:
                # 1. Text length (longer relevant texts may be more valuable)
                text_length_score = min(len(text_content) / 1000, 0.3)  # Max +0.3 for length

                # 2. Query word overlap in text
                query_words = set(query.lower().split())
                text_words = set(text_content.lower().split())
                overlap_score = (
                    len(query_words.intersection(text_words)) / max(len(query_words), 1) * 0.4
                )  # Max +0.4 for overlap

                # 3. Combine original score with new metrics
                reranked_score = original_score + text_length_score + overlap_score
                reranked_score = min(reranked_score, 1.0)  # Cap at 1.0

                updated_doc = doc.copy()
                updated_doc["score"] = reranked_score
                reranked_docs.append(updated_doc)

            # Sort by new score
            reranked_docs.sort(key=lambda x: x["score"], reverse=True)

            # Apply threshold filter
            filtered_docs = [
                doc for doc in reranked_docs if doc["score"] >= self.relevance_threshold
            ]

            self.log.info(f"RAG: After reranking and filtering, {len(filtered_docs)} documents remain")

            for i, doc in enumerate(filtered_docs):
                self.log.info(
                    f"RAG: Rerank #{i + 1} - Relevance: {doc['score']:.3f}, Text: '{doc['text'][:100]}...'"
                )

            return filtered_docs

        except Exception as e:
            self.log.error(f"Error reranking documents: {e}")
            # Return original documents filtered by threshold if reranking fails
            return [
                doc for doc in documents if doc["score"] >= self.relevance_threshold
            ]

    def toggle_reranker(self) -> bool:
        """
        Toggle reranker state.

        Returns:
            New reranker state (True if enabled, False if disabled)
        """
        self.reranker_enabled = not self.reranker_enabled
        status = "enabled" if self.reranker_enabled else "disabled"
        self.log.info(f"RAG: Reranker {status}")
        return self.reranker_enabled

    def set_relevance_threshold(self, threshold: float) -> float:
        """
        Set the relevance threshold.

        Args:
            threshold: Threshold value (will be clamped to [0, 1])

        Returns:
            The set threshold value
        """
        self.relevance_threshold = max(0.0, min(1.0, threshold))
        self.log.info(f"RAG: Relevance threshold set to: {self.relevance_threshold}")
        return self.relevance_threshold

    def get_rag_response(self, query: str, top_k: int = 10) -> str:
        """
        Get an enhanced response using RAG.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Enhanced prompt with context or original query if no documents found
        """
        try:
            # Search for relevant chunks
            similar_docs = self.search_similar(query, top_k)

            if not similar_docs:
                self.log.info("No relevant documents found, returning original query")
                return query

            # Combine found documents with the question
            context_parts = []
            for i, doc in enumerate(similar_docs):
                text_content = (
                    doc["text"]
                    or doc["full_text"]
                    or (str(doc["payload"]) if doc["payload"] else "Document with no text content")
                )
                source_info = (
                    f" (Source: {doc['source_document']})" if doc["source_document"] else ""
                )
                context_parts.append(
                    f"Context document #{i + 1}{source_info} (relevance: {doc['score']:.3f}): {text_content}"
                )

            context = "\n\n".join(context_parts)
            rag_prompt = (
                f"Based on the following information, answer the question:\n\n"
                f"{context}\n\nQuestion: {query}"
            )

            self.log.info(
                f"RAG: Found {len(similar_docs)} documents, context length: {len(rag_prompt)} characters"
            )
            return rag_prompt
        except Exception as e:
            self.log.error(f"Error generating RAG response: {e}")
            # Return original query on error
            return query
