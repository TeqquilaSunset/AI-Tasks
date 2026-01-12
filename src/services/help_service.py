# -*- coding: utf-8 -*-
"""Help service combining RAG documentation search with git context."""

from typing import Any, Dict, List, Optional

import requests

from ..config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
)
from ..utils import setup_logging
from ..utils.qdrant_client import QdrantIndexer


class HelpService:
    """
    Service for project help using RAG and git context.

    Provides intelligent help about the project by combining:
    - Documentation search via RAG
    - Git repository context
    - Code style guidelines
    - Project structure information

    Attributes:
        docs_collection_name: Qdrant collection for documentation
        embedding_model: Ollama embedding model
        ollama_host: Ollama API URL
        qdrant_indexer: QdrantIndexer instance
    """

    DOCS_COLLECTION = "project_docs"  # Default collection for indexed docs

    def __init__(
        self,
        docs_collection_name: str = DOCS_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        qdrant_host: str = DEFAULT_QDRANT_HOST,
        qdrant_port: int = DEFAULT_QDRANT_PORT,
    ) -> None:
        """
        Initialize the help service.

        Args:
            docs_collection_name: Qdrant collection for documentation
            embedding_model: Ollama embedding model
            ollama_host: Ollama API URL
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
        """
        self.docs_collection_name = docs_collection_name
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.qdrant_indexer = QdrantIndexer(
            host=qdrant_host, port=qdrant_port, collection_name=docs_collection_name
        )
        self.log = setup_logging("help-service")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
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
        except Exception as e:
            self.log.error(f"Error generating embedding: {e}")
            raise

    def search_documentation(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search project documentation for relevant information.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant documentation chunks
        """
        try:
            query_embedding = self.generate_embedding(query)
            points = self.qdrant_indexer.query_points(query_embedding, limit=top_k)

            results = []
            for point in points:
                score = getattr(point, "score", 0)
                payload = getattr(point, "payload", {})

                results.append({
                    "score": score,
                    "text": payload.get("text", "") if isinstance(payload, dict) else "",
                    "full_text": payload.get("full_text", "") if isinstance(payload, dict) else "",
                    "source_file": payload.get("source_file", "") if isinstance(payload, dict) else "",
                })

            self.log.info(f"Found {len(results)} documentation results")
            return results

        except Exception as e:
            self.log.error(f"Error searching documentation: {e}")
            return []

    def get_help_response(
        self,
        query: str,
        git_context: Optional[Dict[str, str]] = None,
        top_k: int = 5,
    ) -> str:
        """
        Get help response combining documentation and git context.

        Args:
            query: User's help question
            git_context: Optional git repository context
            top_k: Number of documentation results

        Returns:
            Enhanced help prompt with context
        """
        # Search documentation
        docs_results = self.search_documentation(query, top_k=top_k)

        # Build context parts
        context_parts = []

        # Add git context if available
        if git_context:
            git_info = []
            if git_context.get("current_branch"):
                git_info.append(f"Current branch: {git_context['current_branch']}")
            if git_context.get("repo_info"):
                git_info.append(f"Repository: {git_context['repo_info']}")
            if git_context.get("recent_commits"):
                git_info.append(f"Recent activity:\n{git_context['recent_commits']}")

            if git_info:
                context_parts.append("## Git Context\n" + "\n".join(git_info))

        # Add documentation results
        if docs_results:
            docs_context = []
            for i, doc in enumerate(docs_results):
                text_content = (
                    doc["text"] or doc["full_text"] or "No content available"
                )
                source_info = f" (from {doc['source_file']})" if doc["source_file"] else ""
                docs_context.append(
                    f"### Doc #{i + 1}{source_info} (relevance: {doc['score']:.3f})\n{text_content}"
                )

            context_parts.append("## Documentation\n" + "\n\n".join(docs_context))

        # Build final prompt
        if context_parts:
            context = "\n\n".join(context_parts)
            help_prompt = (
                f"You are a helpful assistant for this project. "
                f"Based on the following context, answer the user's question.\n\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Provide a helpful answer with code examples if relevant. "
                f"Reference specific files or documentation sections when appropriate."
            )
            self.log.info(f"Generated help response with {len(docs_results)} docs and git context")
        else:
            # No context available - basic response
            help_prompt = (
                f"Question: {query}\n\n"
                f"Note: No indexed documentation or git context available. "
                f"Please run 'python index_docs.py' to index project documentation."
            )
            self.log.warning("No documentation or context available for help")

        return help_prompt

    def get_style_guide_help(self) -> str:
        """Get help about project coding style and conventions."""
        style_query = "code style conventions guidelines patterns best practices"
        results = self.search_documentation(style_query, top_k=3)

        if results:
            style_info = "\n\n".join([
                f"From {r['source_file']}:\n{r['text'][:500]}..."
                for r in results if r['source_file']
            ])
            return f"## Project Style Guidelines\n\n{style_info}"
        else:
            return "## Style Guidelines\n\nNo style documentation found. Run 'python index_docs.py' to index README and other docs."

    def get_api_reference(self, component: str = "") -> str:
        """
        Get API reference help for a specific component.

        Args:
            component: Component name (empty for general API help)

        Returns:
            API reference information
        """
        query = f"API reference {component} functions classes methods".strip()
        results = self.search_documentation(query, top_k=5)

        if results:
            api_info = []
            for r in results:
                if r['source_file']:
                    api_info.append(
                        f"### {r['source_file']}\n{r['text'][:800]}"
                    )
            return f"## API Reference\n\n" + "\n\n".join(api_info)
        else:
            return f"## API Reference\n\nNo API documentation found for '{component}'. Run 'python index_docs.py' to index project files."

    def get_project_structure(self) -> str:
        """Get help about project structure and architecture."""
        query = "project structure architecture modules components organization"
        results = self.search_documentation(query, top_k=3)

        if results:
            structure_info = "\n\n".join([
                r['full_text'][:1000] if r['full_text'] else r['text']
                for r in results
            ])
            return f"## Project Structure\n\n{structure_info}"
        else:
            return "## Project Structure\n\nNo structure documentation found. Check README.md for project overview."
