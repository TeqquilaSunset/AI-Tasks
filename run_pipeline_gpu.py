#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска пайплайна PDF с оптимизациями для GPU (NVIDIA RTX 4070)
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    # Добавляем текущую директорию в путь Python
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(description="Запуск пайплайна PDF с оптимизациями для GPU")
    parser.add_argument("--pdf_path", type=str, required=True, help="Путь к PDF файлу для обработки")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Размер чанка (по умолчанию 2048)")
    parser.add_argument("--overlap", type=int, default=50, help="Перекрытие между чанками (по умолчанию 50)")
    parser.add_argument("--collection_name", type=str, default="pdf_chunks", help="Название коллекции в Qdrant")
    parser.add_argument("--embedding_model", type=str, default="qwen3-embedding:latest", help="Модель эмбеддингов в Ollama")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="URL для Ollama API")
    parser.add_argument("--max_concurrent", type=int, default=15, help="Максимальное количество одновременных запросов (оптимизировано для RTX 4070)")
    parser.add_argument("--use_async", action="store_true", help="Использовать асинхронную версию пайплайна (рекомендуется для GPU)")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Ошибка: Файл не найден: {args.pdf_path}")
        sys.exit(1)

    # Выводим информацию о запуске
    print(f"Запуск пайплайна для файла: {args.pdf_path}")
    print(f"Параметры:")
    print(f"  - Размер чанка: {args.chunk_size}")
    print(f"  - Перекрытие: {args.overlap}")
    print(f"  - Коллекция Qdrant: {args.collection_name}")
    print(f"  - Модель: {args.embedding_model}")
    print(f"  - Макс. одновременных запросов: {args.max_concurrent}")
    print(f"  - Использовать асинхронную версию: {args.use_async}")
    print()

    if args.use_async:
        # Запускаем асинхронную версию
        import asyncio
        from pdf_pipeline_async import main as async_main

        print("Запуск асинхронного пайплайна...")
        result = asyncio.run(async_main(
            args.pdf_path,
            args.chunk_size,
            args.overlap,
            args.collection_name,
            args.embedding_model,
            args.ollama_host,
            args.max_concurrent
        ))
    else:
        # Запускаем синхронную версию
        from pdf_pipeline import main as sync_main

        print("Запуск синхронного пайплайна...")
        result = sync_main(
            args.pdf_path,
            args.chunk_size,
            args.overlap,
            args.collection_name,
            args.embedding_model,
            args.ollama_host
        )

    if result:
        print("\n✓ Пайплайн успешно завершен!")
    else:
        print("\n✗ Пайплайн завершился с ошибкой!")
        sys.exit(1)

if __name__ == "__main__":
    main()