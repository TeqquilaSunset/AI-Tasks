# -*- coding: utf-8 -*-
"""
FAQ Service for Support System

Provides RAG-based search through FAQ and project documentation.
"""

import asyncio
from typing import List, Dict, Any, Optional
from .rag_service import RAGService
from ..utils import setup_logging
from ..config import DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_TOP_K

log = setup_logging("faq-service")


class FAQService:
    """
    Service for searching FAQ and documentation using RAG.

    Provides:
    - Search through FAQ (FAQ.md)
    - Search through project documentation
    - Context-aware answers
    - Combined search results
    """

    def __init__(
        self,
        collection_name: str = "project_docs",
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Initialize FAQ service.

        Args:
            collection_name: Qdrant collection name
            threshold: Relevance threshold for RAG
            top_k: Number of results to retrieve
        """
        self.collection_name = collection_name
        self.threshold = threshold
        self.top_k = top_k
        self.rag_service = RAGService(
            collection_name=collection_name
        )
        # Override default threshold
        self.rag_service.relevance_threshold = threshold
        log.info(f"FAQ service initialized with collection: {collection_name}")

    async def search_faq(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search FAQ for relevant answers.

        Args:
            query: User question
            top_k: Number of results (override default)

        Returns:
            List of relevant FAQ entries with scores
        """
        try:
            k = top_k or self.top_k
            results = self.rag_service.search_similar(query, top_k=k)

            # Convert RAG results to FAQ format
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "score": r.get("score", 0),
                    "text": r.get("text", ""),
                    "metadata": {
                        "source": r.get("source_document", "")
                    }
                })

            # Filter by threshold
            filtered = [
                r for r in formatted_results
                if r.get("score", 0) >= self.threshold
            ]

            log.info(f"FAQ search for '{query}': {len(filtered)} results")
            return filtered

        except Exception as e:
            log.error(f"Error searching FAQ: {e}")
            return []

    async def search_documentation(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search project documentation.

        Args:
            query: User question
            top_k: Number of results (override default)

        Returns:
            List of relevant documentation entries
        """
        try:
            k = top_k or self.top_k
            results = self.rag_service.search_similar(query, top_k=k)

            # Convert RAG results to FAQ format
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "score": r.get("score", 0),
                    "text": r.get("text", ""),
                    "metadata": {
                        "source": r.get("source_document", "")
                    }
                })

            # Filter by threshold and prioritize certain sources
            filtered = [
                r for r in formatted_results
                if r.get("score", 0) >= self.threshold
            ]

            log.info(f"Documentation search for '{query}': {len(filtered)} results")
            return filtered

        except Exception as e:
            log.error(f"Error searching documentation: {e}")
            return []

    async def get_combined_context(
        self,
        query: str,
        ticket_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get combined context from FAQ and documentation.

        Args:
            query: User question
            ticket_context: Optional ticket context for customization

        Returns:
            Formatted context string for LLM
        """
        try:
            # Search FAQ and documentation
            faq_results = await self.search_faq(query, top_k=3)
            doc_results = await self.search_documentation(query, top_k=3)

            # Combine results
            context_parts = []

            if faq_results:
                context_parts.append("## Релевантные разделы FAQ:\n")
                for i, result in enumerate(faq_results, 1):
                    text = result.get("text", "")
                    source = result.get("metadata", {}).get("source", "unknown")
                    score = result.get("score", 0)
                    context_parts.append(f"{i}. [{source}] (релевантность: {score:.2f})\n{text}\n")

            if doc_results:
                context_parts.append("\n## Дополнительная документация:\n")
                for i, result in enumerate(doc_results, 1):
                    text = result.get("text", "")
                    source = result.get("metadata", {}).get("source", "unknown")
                    score = result.get("score", 0)
                    context_parts.append(f"{i}. [{source}] (релевантность: {score:.2f})\n{text}\n")

            # Add ticket context if available
            if ticket_context:
                context_parts.insert(0, f"""## Контекст тикета:
Тикет: #{ticket_context.get('ticket_id')}
Тема: {ticket_context.get('subject')}
Пользователь: {ticket_context.get('user', {}).get('name')}
Статус: {ticket_context.get('status')}
Приоритет: {ticket_context.get('priority')}

История сообщений:
{ticket_context.get('message_history', '')}

""")

            if not context_parts:
                return "Релевантная информация не найдена в документации."

            return "\n".join(context_parts)

        except Exception as e:
            log.error(f"Error getting combined context: {e}")
            return "Произошла ошибка при поиске документации."

    def categorize_question(self, query: str) -> str:
        """
        Categorize user question based on keywords.

        Args:
            query: User question

        Returns:
            Category name
        """
        query_lower = query.lower()

        categories = {
            "authentication": ["авторизац", "вход", "логин", "пароль", "auth", "login"],
            "configuration": ["настройк", "конфигурац", "перемен", ".env", "config"],
            "usage": ["как использовать", "как работает", "команд", "how to use"],
            "technical": ["ошибк", "не работает", "проблема", "error", "bug", "issue"],
            "pdf": ["pdf", "документ", "индексаци", "файл"],
            "rag": ["rag", "поиск", "векторн", "embedding"],
            "mcp": ["mcp", "инструмент", "tool"],
            "performance": ["медленн", "ускорить", "оптимиз", "производительн", "performance"]
        }

        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category

        return "general"

    async def get_answer_suggestion(
        self,
        query: str,
        ticket_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get answer suggestion based on FAQ and documentation.

        Args:
            query: User question
            ticket_context: Optional ticket context

        Returns:
            Dictionary with suggestion and sources
        """
        try:
            # Categorize question
            category = self.categorize_question(query)

            # Get context
            context = await self.get_combined_context(query, ticket_context)

            # Get direct FAQ matches
            faq_results = await self.search_faq(query, top_k=1)

            direct_answer = None
            if faq_results and faq_results[0].get("score", 0) > 0.8:
                # High confidence direct match
                direct_answer = faq_results[0].get("text", "")

            return {
                "category": category,
                "context": context,
                "direct_answer": direct_answer,
                "sources_found": len(faq_results) > 0,
                "confidence": faq_results[0].get("score", 0) if faq_results else 0
            }

        except Exception as e:
            log.error(f"Error getting answer suggestion: {e}")
            return {
                "category": "general",
                "context": "Произошла ошибка при поиске ответа.",
                "direct_answer": None,
                "sources_found": False,
                "confidence": 0
            }
