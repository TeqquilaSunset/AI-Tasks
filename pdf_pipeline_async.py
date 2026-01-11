#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пайплайн разбивки PDF файла на чанки и генерацию эмбедингов с использованием Ollama
и реализация сохранения индексов в qdrant (запущен на http://localhost:6333/)
С асинхронной обработкой для ускорения за счет видеокарты
"""

import logging
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import numpy as np
import asyncio
import aiohttp
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, CollectionStatus
import uuid
import argparse
import re
import time


# --------------------  ЛОГИРОВАНИЕ  --------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("pdf-pipeline-ollama-async")


class PDFChunker:
    """Класс для разбиения PDF файла на чанки"""

    def __init__(self, chunk_size: int = 1024, overlap: int = 50):
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
    """Класс для генерации эмбеддингов через Ollama с асинхронной обработкой"""

    def __init__(self, model_name: str = "qwen3-embedding:latest", ollama_host: str = "http://localhost:11434", max_concurrent: int = 10):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.max_concurrent = max_concurrent  # Количество одновременных запросов для оптимизации GPU

    async def generate_embedding_async(self, session, text: str) -> List[float]:
        """Асинхронная генерация эмбеддинга через Ollama API"""
        try:
            async with session.post(
                f"{self.ollama_host}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama вернул статус {response.status}: {await response.text()}")

                result = await response.json()
                embedding = result.get("embedding")

                if embedding is None:
                    raise ValueError("Ollama не вернул эмбеддинг")

                return embedding
        except Exception as e:
            log.error(f"Ошибка при генерации эмбеддинга: {e}")
            raise

    async def generate_batch_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Асинхронная генерация эмбеддингов для списка текстов с ограничением параллелизма"""
        semaphore = asyncio.Semaphore(self.max_concurrent)  # Ограничивает количество одновременных запросов

        async def fetch_embedding_with_semaphore(session, text):
            async with semaphore:
                return await self.generate_embedding_async(session, text)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            tasks = [fetch_embedding_with_semaphore(session, text) for text in texts]
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            # Проверяем, были ли исключения
            processed_embeddings = []
            for i, emb in enumerate(embeddings):
                if isinstance(emb, Exception):
                    log.error(f"Ошибка при генерации эмбеддинга для текста {i}: {emb}")
                    raise emb
                processed_embeddings.append(emb)

            return processed_embeddings


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

    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], pdf_path: str, metadata_list: Optional[List[Dict]] = None) -> bool:
        """Сохранение эмбеддингов и чанков в Qdrant"""
        try:
            points = []

            # Получаем имя файла из пути
            source_document = os.path.basename(pdf_path)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Подготовка метаданных
                metadata = {
                    "chunk_id": i,
                    "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,  # Обрезаем текст для экономии места
                    "full_text": chunk,  # Полный текст в метаданных
                    "source_document": source_document  # Добавляем имя исходного документа
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

            # Загрузка точек в Qdrant пакетно для улучшения производительности
            batch_size = 100  # Количество точек за один запрос
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                log.info(f"Сохранено {len(batch)} точек в коллекцию '{self.collection_name}' из документа '{source_document}' ({i+batch_size}/{len(points)})")

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


async def main(pdf_path: str, chunk_size: int = 1024, overlap: int = 50,
               collection_name: str = "pdf_chunks",
               embedding_model: str = "qwen3-embedding:latest",
               ollama_host: str = "http://localhost:11434",
               max_concurrent: int = 10):
    """Основная функция пайплайна с асинхронной обработкой"""
    log.info(f"Начало обработки PDF файла: {pdf_path}")

    # Проверка доступности Ollama
    import requests
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

    # Шаг 2: Генерация эмбеддингов с использованием Ollama (асинхронно)
    log.info("Шаг 2: Генерация эмбеддингов с использованием Ollama (асинхронно)")
    embedder = OllamaEmbeddingGenerator(model_name=embedding_model, ollama_host=ollama_host, max_concurrent=max_concurrent)

    start_time = time.time()
    try:
        embeddings = await embedder.generate_batch_embeddings_async(chunks)
        elapsed_time = time.time() - start_time
        log.info(f"Сгенерировано {len(embeddings)} эмбеддингов за {elapsed_time:.2f} секунд")
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

    success = indexer.store_embeddings(chunks, embeddings, pdf_path)

    if success:
        log.info("Пайплайн успешно завершен!")
    else:
        log.error("Ошибка при записи в Qdrant")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пайплайн обработки PDF для Qdrant с использованием Ollama (асинхронная версия)")
    parser.add_argument("--pdf_path", type=str, required=True, help="Путь к PDF файлу")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Размер чанка (по умолчанию 1024)")
    parser.add_argument("--overlap", type=int, default=50, help="Перекрытие между чанками (по умолчанию 50)")
    parser.add_argument("--collection_name", type=str, default="pdf_chunks", help="Название коллекции в Qdrant")
    parser.add_argument("--embedding_model", type=str, default="qwen3-embedding:latest", help="Модель эмбеддингов в Ollama")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="URL для Ollama API")
    parser.add_argument("--max_concurrent", type=int, default=10, help="Максимальное количество одновременных запросов к Ollama")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        log.error(f"Файл не найден: {args.pdf_path}")
        sys.exit(1)

    # Запускаем асинхронную функцию
    result = asyncio.run(main(args.pdf_path, args.chunk_size, args.overlap,
                              args.collection_name, args.embedding_model,
                              args.ollama_host, args.max_concurrent))

    if not result:
        sys.exit(1)