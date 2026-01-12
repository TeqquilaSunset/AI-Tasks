# -*- coding: utf-8 -*-
"""Ollama embedding generation client."""

import asyncio
from typing import List

import aiohttp
import requests


class OllamaEmbeddingGenerator:
    """
    Client for generating embeddings using Ollama.

    Supports both synchronous and asynchronous embedding generation.

    Attributes:
        model_name: Name of the Ollama model to use
        ollama_host: URL of the Ollama API
        max_concurrent: Maximum number of concurrent requests (async only)
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:latest",
        ollama_host: str = "http://localhost:11434",
        max_concurrent: int = 10,
    ) -> None:
        """
        Initialize the Ollama embedding generator.

        Args:
            model_name: Name of the Ollama embedding model
            ollama_host: URL of the Ollama API
            max_concurrent: Maximum concurrent requests for async operations
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.max_concurrent = max_concurrent

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (synchronous).

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
                json={"model": self.model_name, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            embedding = result.get("embedding")

            if embedding is None:
                raise ValueError("Ollama did not return an embedding")

            return embedding
        except requests.RequestException as e:
            raise  # Re-raise for caller to handle
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}") from e

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (synchronous).

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    async def generate_embedding_async(
        self, session: aiohttp.ClientSession, text: str
    ) -> List[float]:
        """
        Generate embedding for a single text (asynchronous).

        Args:
            session: aiohttp ClientSession
            text: Input text to embed

        Returns:
            List of embedding values

        Raises:
            Exception: If the request fails
        """
        try:
            async with session.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Ollama returned status {response.status}: {await response.text()}"
                    )

                result = await response.json()
                embedding = result.get("embedding")

                if embedding is None:
                    raise ValueError("Ollama did not return an embedding")

                return embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}") from e

    async def generate_batch_embeddings_async(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (asynchronous).

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_with_semaphore(session: aiohttp.ClientSession, text: str):
            async with semaphore:
                return await self.generate_embedding_async(session, text)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            tasks = [fetch_with_semaphore(session, text) for text in texts]
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            processed_embeddings = []
            for i, emb in enumerate(embeddings):
                if isinstance(emb, Exception):
                    raise emb
                processed_embeddings.append(emb)

            return processed_embeddings
