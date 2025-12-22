#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пайплайн разбивки PDF файла на чанки и генерацию эмбедингов с использованием Ollama
и реализация сохранения индексов в qdrant (запущен на http://localhost:6333/)
"""

import logging
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import numpy as np
import requests
import asyncio
import aiohttp
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, CollectionStatus
import uuid
import argparse
import re


# --------------------  ЛОГИРОВАНИЕ  --------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("pdf-pipeline-ollama")


class PDFChunker:
    """Класс для разбиения PDF файла на чанки"""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def read_pdf(self, pdf_path: str) -> str:
        """Чтение текста из PDF файла"""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Разбиение текста на чанки с перекрытием"""
        sentences = re.split(r'[.!?]+\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Удаляем лишние пробелы и добавляем разделитель
            sentence = sentence.strip()

            # Проверяем, превышает ли текущий чанк размер, если добавим новое предложение
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Начинаем новый чанк, возможно с перекрытием
                if self.overlap > 0:
                    # Получаем последние несколько слов из текущего чанка для перекрытия
                    words = current_chunk.split()
                    overlap_text = ' '.join(words[-self.overlap:]) if len(words) > self.overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Добавляем последний чанк, если он не пустой
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


class OllamaEmbeddingGenerator:
    """Класс для генерации эмбеддингов через Ollama"""

    def __init__(self, model_name: str = "qwen3-embedding:latest", ollama_host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host

    def generate_embedding(self, text: str) -> List[float]:
        """Генерация эмбеддинга через Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            embedding = result.get("embedding")

            if embedding is None:
                raise ValueError("Ollama не вернул эмбеддинг")

            return embedding
        except requests.exceptions.RequestException as e:
            log.error(f"Ошибка при запросе к Ollama: {e}")
            raise
        except Exception as e:
            log.error(f"Ошибка при генерации эмбеддинга: {e}")
            raise

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для списка текстов"""
        embeddings = []
        for i, text in enumerate(texts):
            log.info(f"Генерация эмбеддинга {i+1}/{len(texts)}")
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings


class QdrantIndexer:
    """Класс для работы с Qdrant индексом"""

    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "pdf_chunks"):
        self.host = host
        self.port = port
        self.collection_name = collection_name

        # Подключение к Qdrant (без протокола в host)
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, vector_size: int = 384) -> bool:
        """Создание коллекции в Qdrant"""
        try:
            collections = self.client.get_collections()

            # Проверяем, существует ли коллекция
            collection_exists = False
            for collection in collections.collections:
                if collection.name == self.collection_name:
                    collection_exists = True
                    break

            if not collection_exists:
                # Создаем новую коллекцию
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                log.info(f"Создана коллекция '{self.collection_name}'")
            else:
                log.info(f"Коллекция '{self.collection_name}' уже существует")

            return True
        except Exception as e:
            log.error(f"Ошибка при создании коллекции: {e}")
            return False

    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], metadata_list: Optional[List[Dict]] = None) -> bool:
        """Сохранение эмбеддингов и чанков в Qdrant"""
        try:
            points = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Подготовка метаданных
                metadata = {
                    "chunk_id": i,
                    "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,  # Обрезаем текст для экономии места
                    "full_text": chunk  # Полный текст в метаданных
                }

                # Добавляем дополнительные метаданные, если они предоставлены
                if metadata_list and i < len(metadata_list):
                    metadata.update(metadata_list[i])

                # Создание точки для Qdrant
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=metadata
                )

                points.append(point)

            # Загрузка точек в Qdrant
            self.client.upsert(collection_name=self.collection_name, points=points)
            log.info(f"Сохранено {len(points)} точек в коллекцию '{self.collection_name}'")

            return True
        except Exception as e:
            log.error(f"Ошибка при сохранении эмбеддингов в Qdrant: {e}")
            return False

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск похожих документов по эмбеддингу"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )

            similar_docs = []
            for result in results:
                doc_info = {
                    "score": result.score,
                    "payload": result.payload,
                    "text": result.payload.get("text", ""),
                    "full_text": result.payload.get("full_text", "")
                }
                similar_docs.append(doc_info)

            return similar_docs
        except Exception as e:
            log.error(f"Ошибка при поиске похожих документов: {e}")
            return []


def main(pdf_path: str, chunk_size: int = 512, overlap: int = 50,
         collection_name: str = "pdf_chunks",
         embedding_model: str = "qwen3-embedding:latest",
         ollama_host: str = "http://localhost:11434"):
    """Основная функция пайплайна"""
    log.info(f"Начало обработки PDF файла: {pdf_path}")

    # Проверка доступности Ollama
    try:
        response = requests.get(f"{ollama_host}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        available_models = [model["name"] for model in models]

        if embedding_model not in available_models:
            log.warning(f"Модель {embedding_model} может не быть доступна. Доступные модели: {available_models}")
        else:
            log.info(f"Модель {embedding_model} найдена в Ollama")
    except Exception as e:
        log.error(f"Не удалось подключиться к Ollama: {e}")
        return False

    # Шаг 1: Разбиение PDF на чанки
    log.info("Шаг 1: Разбиение PDF на чанки")
    chunker = PDFChunker(chunk_size=chunk_size, overlap=overlap)
    text = chunker.read_pdf(pdf_path)
    chunks = chunker.chunk_text(text)
    log.info(f"Получено {len(chunks)} чанков")

    # Шаг 2: Генерация эмбеддингов с использованием Ollama
    log.info("Шаг 2: Генерация эмбеддингов с использованием Ollama")
    embedder = OllamaEmbeddingGenerator(model_name=embedding_model, ollama_host=ollama_host)

    try:
        embeddings = embedder.generate_batch_embeddings(chunks)
        log.info(f"Сгенерировано {len(embeddings)} эмбеддингов")
    except Exception as e:
        log.error(f"Ошибка при генерации эмбеддингов: {e}")
        return False

    # Шаг 3: Сохранение в Qdrant
    log.info("Шаг 3: Сохранение в Qdrant")
    indexer = QdrantIndexer(collection_name=collection_name)

    # Определяем размер вектора из первого эмбеддинга
    if embeddings:
        vector_size = len(embeddings[0])
        if not indexer.create_collection(vector_size=vector_size):
            log.error("Не удалось создать коллекцию в Qdrant")
            return False
    else:
        log.error("Нет сгенерированных эмбеддингов")
        return False

    success = indexer.store_embeddings(chunks, embeddings)

    if success:
        log.info("Пайплайн успешно завершен!")
    else:
        log.error("Ошибка при записи в Qdrant")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пайплайн обработки PDF для Qdrant с использованием Ollama")
    parser.add_argument("--pdf_path", type=str, required=True, help="Путь к PDF файлу")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Размер чанка (по умолчанию 512)")
    parser.add_argument("--overlap", type=int, default=50, help="Перекрытие между чанками (по умолчанию 50)")
    parser.add_argument("--collection_name", type=str, default="pdf_chunks", help="Название коллекции в Qdrant")
    parser.add_argument("--embedding_model", type=str, default="qwen3-embedding:latest", help="Модель эмбеддингов в Ollama")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="URL для Ollama API")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        log.error(f"Файл не найден: {args.pdf_path}")
        sys.exit(1)

    main(args.pdf_path, args.chunk_size, args.overlap, args.collection_name, args.embedding_model, args.ollama_host)