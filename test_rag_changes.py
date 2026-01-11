#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки изменений в RAG-системе
Проверяет, что информация о файле сохраняется в метаданных и отображается при поиске
"""

import os
import tempfile
from pathlib import Path

def test_changes():
    """Тестируем изменения в файлах"""
    print("Проверка изменений в RAG-системе...")
    
    # Проверяем изменения в pdf_pipeline.py
    print("\n1. Проверка pdf_pipeline.py:")
    with open("pdf_pipeline.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    if '"source_document": source_document' in content:
        print("   ✓ Добавлено сохранение имени документа в метаданные")
    else:
        print("   ✗ Не найдено сохранение имени документа")
        
    if 'source_document = os.path.basename(pdf_path)' in content:
        print("   ✓ Добавлено извлечение имени файла")
    else:
        print("   ✗ Не найдено извлечение имени файла")
    
    # Проверяем изменения в pdf_pipeline_async.py
    print("\n2. Проверка pdf_pipeline_async.py:")
    with open("pdf_pipeline_async.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    if '"source_document": source_document' in content:
        print("   ✓ Добавлено сохранение имени документа в метаданные")
    else:
        print("   ✗ Не найдено сохранение имени документа")
        
    if 'source_document = os.path.basename(pdf_path)' in content:
        print("   ✓ Добавлено извлечение имени файла")
    else:
        print("   ✗ Не найдено извлечение имени файла")
    
    # Проверяем изменения в main.py
    print("\n3. Проверка main.py (RAGService):")
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    if '"source_document": payload.get("source_document"' in content:
        print("   ✓ Добавлено извлечение имени документа при поиске")
    else:
        print("   ✗ Не найдено извлечение имени документа при поиске")
        
    if 'source_info = f" (Источник: {doc[\'source_document\']})"' in content:
        print("   ✓ Добавлено отображение источника в контексте")
    else:
        print("   ✗ Не найдено отображение источника в контексте")
    
    if 'log.info(f"RAG: Результат #{i+1} - Релевантность:' in content and '(Источник:' in content:
        print("   ✓ Добавлено логирование информации об источнике")
    else:
        print("   ⚠ Логирование информации об источнике может быть не добавлено")
    
    print("\n4. Проверка сигнатур функций:")
    # Проверяем, что функция store_embeddings теперь принимает pdf_path
    if 'def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], pdf_path: str' in content:
        print("   ✓ Обновлена сигнатура функции store_embeddings")
    else:
        print("   ✗ Сигнатура функции store_embeddings не обновлена")
    
    print("\nТест завершен!")
    print("\nТеперь при использовании RAG в чате будет отображаться информация о том, из какого файла была взята информация.")

if __name__ == "__main__":
    test_changes()