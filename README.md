# AI Challenge Task 1 - OpenAI Compatible Chat Client

Проект представляет собой консольный чат-клиент с интеграцией MCP (Model Context Protocol), RAG (Retrieval Augmented Generation) и поддержкой обработки PDF документов.

## [English](#english) | [Русский](#русский)

---

<a name="русский"></a>
## Русский

### Основные возможности

- **MCP интеграция**: Полная поддержка протокола Model Context Protocol
  - STDIO-транспорт для локальных MCP серверов
  - HTTP-транспорт для Docker MCP серверов
  - Автоматическое обнаружение инструментов
- **RAG поддержка**: Работа с векторной базой данных Qdrant
  - Поиск по документам
  - Реранкинг результатов
  - Настраиваемый порог релевантности
- **Git интеграция через MCP**:
  - Получение текущей ветки и статуса
  - Просмотр изменений (diff)
  - История коммитов
  - Содержимое файлов
- **/help команда**:
  - Поиск по проектной документации
  - Git-контекстная справка
  - Справочник по API и структуре
  - Руководство по стилю кода
- **PDF обработка**: Пайплайн для индексации документов
  - Синхронная и асинхронная (GPU-оптимизированная) версии
  - Генерация эмбеддингов через Ollama
  - Хранение в Qdrant
- **Управление разговором**:
  - Сохранение/загрузка истории
  - Настройка температуры
  - RAG команды
  - Help команды

### Структура проекта (после рефакторинга)

```
AI-Task-1/
├── src/                          # Модульная архитектура
│   ├── config.py                 # Конфигурация и константы
│   ├── utils/                    # Общие утилиты
│   │   ├── __init__.py
│   │   ├── logging_config.py     # Настройка логирования
│   │   ├── pdf_chunker.py        # Разбиение PDF на чанки
│   │   ├── docs_chunker.py       # Разбиение документации
│   │   ├── ollama_client.py      # Клиент для эмбеддингов
│   │   └── qdrant_client.py      # Клиент векторной БД
│   ├── services/                 # Бизнес-логика
│   │   ├── __init__.py
│   │   ├── rag_service.py        # RAG сервис
│   │   └── help_service.py       # Help сервис (новое!)
│   └── clients/                  # MCP клиенты
│       ├── __init__.py
│       ├── mcp_client.py         # STDIO MCP клиент
│       └── docker_mcp_client.py  # Docker MCP клиент
├── main.py                       # Главный чат-клиент (рефакторинг)
├── mcp_server.py                 # MCP сервер (+ Git инструменты!)
├── index_docs.py                 # Индексация документации (новое!)
├── pdf_pipeline.py               # PDF пайплайн (синхронный)
├── pdf_pipeline_async.py         # PDF пайплайн (асинхронный)
├── run_pipeline_gpu.py           # GPU-оптимизированный запуск
├── regular.py                    # Weather мониторинг агент
├── test_rag_changes.py           # Тесты RAG
├── requirements.txt              # Зависимости
├── docker-compose.yml            # Docker конфигурация
├── .env                          # Переменные окружения
├── README.md                     # Этот файл
├── Pipeline_README.md            # Документация PDF пайплайна
└── GPU_SETUP.md                  # GPU настройки
```

### Установка

1. **Клонирование репозитория**:
   ```bash
   git clone <repo-url>
   cd AI-Task-1
   ```

2. **Установка зависимостей**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Настройка окружения** (.env):
   ```env
   # OpenAI/LLM
   OPENAI_API_KEY=your_key_here
   OPENAI_BASE_URL=https://api.example.com/v1

   # Weather API (для MCP сервера)
   OPEN_WEATHER=your_openweather_api_key

   # MCP Docker
   DOCKER_MCP_HOST=localhost
   DOCKER_MCP_PORT=9011
   ```

4. **Запуск сервисов**:
   - Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
   - Ollama: `ollama serve` (для RAG и PDF пайплайна)

### Использование

#### Чат-клиент

```bash
python main.py
```

**Команды чата**:
- `quit` / `exit` - выход
- `save <имя>` - сохранить разговор
- `load <имя>` - загрузить разговор
- `temp <0-2>` - установить температуру
- `clear` - очистить историю
- `print` - показать историю

**RAG команды**:
- `/rag` - вкл/выкл RAG режим
- `/rag <вопрос>` - задать вопрос с RAG
- `/rag rerank` - вкл/выкл реранкер
- `/rag threshold <0-1>` - установить порог

**Help команды** (новое!):
- `/help` - общая справка по проекту
- `/help <вопрос>` - поиск по документации проекта
- `/help style` - руководство по стилю кода
- `/help api [компонент]` - справочник по API
- `/help structure` - структура проекта
- `/help git` - текущий git-контекст

#### Индексация документации (новое!)

Перед использованием команды `/help`, необходимо проиндексировать документацию проекта:

```bash
# Индексация всех файлов проекта
python index_docs.py

# С параметрами
python index_docs.py --project_root . --collection_name project_docs --extensions .md .py .txt .yml .json
```

**Параметры индексации**:
- `--project_root` - корневая директория проекта (по умолчанию: текущая)
- `--collection_name` - название коллекции в Qdrant (по умолчанию: project_docs)
- `--chunk_size` - размер чанка текста (по умолчанию: 1024)
- `--extensions` - типы файлов для индексации

#### PDF индексация

```bash
# Синхронная версия
python pdf_pipeline.py --pdf_path document.pdf

# Асинхронная (GPU оптимизированная)
python pdf_pipeline_async.py --pdf_path document.pdf --max_concurrent 10

# GPU скрипт
python run_pipeline_gpu.py --pdf_path document.pdf --use_async --max_concurrent 15
```

**Параметры**:
- `--pdf_path` - путь к PDF (обязательно)
- `--chunk_size` - размер чанка (по умолчанию 1024)
- `--overlap` - перекрытие (по умолчанию 50)
- `--collection_name` - коллекция Qdrant (по умолчанию pdf_chunks)
- `--embedding_model` - модель Ollama (по умолчанию qwen3-embedding:latest)
- `--ollama_host` - URL Ollama (по умолчанию http://localhost:11434)
- `--max_concurrent` - кол-во параллельных запросов (по умолчанию 10)

### MCP инструменты

Доступные инструменты при запуске чата:

**Погодные инструменты**:
- `get_weather` - текущая погода для города
- `get_weather_forecast` - прогноз на несколько дней
- `convert_temperature` - конвертация температуры
- `save_weather_data` - сохранение погодных данных

**Git инструменты** (новое!):
- `git_get_current_branch` - текущая ветка
- `git_get_branches` - список всех веток
- `git_get_status` - статус изменений
- `git_get_diff` - показать изменения (diff)
- `git_get_recent_commits` - последние коммиты
- `git_get_file_content` - содержимое файла
- `git_list_files` - список файлов в репозитории
- `git_get_repo_info` - информация о репозитории

**Другие инструменты**:
- `execute_python_code` - выполнение Python в Docker
- `stub_tool` - заглушка для тестов

### Изменения после рефакторинга

#### Архитектурные улучшения:

1. **Модульная структура**: Код разделен на логические модули
   - `src/utils/` - переиспользуемые утилиты
   - `src/services/` - бизнес-логика
   - `src/clients/` - клиенты внешних сервисов
   - `src/config.py` - централизованная конфигурация

2. **Устранение дублирования**:
   - Единый модуль логирования
   - Общий класс PDFChunker
   - Общий клиент Ollama
   - Общий клиент Qdrant

3. **Улучшенная читаемость**:
   - Добавлены type hints
   - Docstrings на английском
   - Последовательное именование

4. **Конфигурация**:
   - Все константы в `src/config.py`
   - Централизованная настройка параметров

#### Размер файлов:

| Файл | До | После |
|------|-----|-------|
| main.py | ~1113 строк | ~587 строк |
| pdf_pipeline.py | ~312 строк | ~163 строки |
| pdf_pipeline_async.py | ~338 строк | ~181 строка |

---

<a name="english"></a>
## English

### Features

- **MCP Integration**: Full Model Context Protocol support
  - STDIO transport for local MCP servers
  - HTTP transport for Docker MCP servers
  - Automatic tool discovery
- **RAG Support**: Qdrant vector database integration
  - Document search
  - Result reranking
  - Configurable relevance threshold
- **Git Integration via MCP**:
  - Current branch and status
  - View changes (diff)
  - Commit history
  - File content retrieval
- **/help Command**:
  - Search project documentation
  - Git-context aware help
  - API reference and structure info
  - Code style guidelines
- **PDF Processing**: Document indexing pipeline
  - Synchronous and asynchronous (GPU-optimized) versions
  - Ollama embedding generation
  - Qdrant storage
- **Conversation Management**:
  - Save/load history
  - Temperature control
  - RAG commands
  - Help commands

### Project Structure (after refactoring)

```
AI-Task-1/
├── src/                          # Modular architecture
│   ├── config.py                 # Configuration and constants
│   ├── utils/                    # Shared utilities
│   │   ├── __init__.py
│   │   ├── logging_config.py     # Logging setup
│   │   ├── pdf_chunker.py        # PDF chunking
│   │   ├── docs_chunker.py       # Documentation chunking
│   │   ├── ollama_client.py      # Embedding client
│   │   └── qdrant_client.py      # Vector DB client
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   ├── rag_service.py        # RAG service
│   │   └── help_service.py       # Help service (new!)
│   └── clients/                  # MCP clients
│       ├── __init__.py
│       ├── mcp_client.py         # STDIO MCP client
│       └── docker_mcp_client.py  # Docker MCP client
├── main.py                       # Main chat client (refactored)
├── mcp_server.py                 # MCP server (+ Git tools!)
├── index_docs.py                 # Documentation indexer (new!)
├── pdf_pipeline.py               # PDF pipeline (sync)
├── pdf_pipeline_async.py         # PDF pipeline (async)
├── run_pipeline_gpu.py           # GPU-optimized runner
├── regular.py                    # Weather monitoring agent
├── test_rag_changes.py           # RAG tests
├── requirements.txt              # Dependencies
├── docker-compose.yml            # Docker configuration
├── .env                          # Environment variables
├── README.md                     # This file
├── Pipeline_README.md            # PDF pipeline documentation
└── GPU_SETUP.md                  # GPU setup guide
```

### Installation

1. **Clone repository**:
   ```bash
   git clone <repo-url>
   cd AI-Task-1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (.env):
   ```env
   # OpenAI/LLM
   OPENAI_API_KEY=your_key_here
   OPENAI_BASE_URL=https://api.example.com/v1

   # Weather API (for MCP server)
   OPEN_WEATHER=your_openweather_api_key

   # MCP Docker
   DOCKER_MCP_HOST=localhost
   DOCKER_MCP_PORT=9011
   ```

4. **Start services**:
   - Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
   - Ollama: `ollama serve` (for RAG and PDF pipeline)

### Usage

#### Chat Client

```bash
python main.py
```

**Chat Commands**:
- `quit` / `exit` - exit
- `save <name>` - save conversation
- `load <name>` - load conversation
- `temp <0-2>` - set temperature
- `clear` - clear history
- `print` - show history

**RAG Commands**:
- `/rag` - toggle RAG mode
- `/rag <question>` - ask with RAG
- `/rag rerank` - toggle reranker
- `/rag threshold <0-1>` - set threshold

**Help Commands** (new!):
- `/help` - general project help
- `/help <question>` - search project documentation
- `/help style` - code style guidelines
- `/help api [component]` - API reference
- `/help structure` - project structure
- `/help git` - current git context

#### Documentation Indexing (new!)

Before using `/help` command, index project documentation:

```bash
# Index all project files
python index_docs.py

# With parameters
python index_docs.py --project_root . --collection_name project_docs --extensions .md .py .txt .yml .json
```

**Indexing Parameters**:
- `--project_root` - project root directory (default: current)
- `--collection_name` - Qdrant collection name (default: project_docs)
- `--chunk_size` - text chunk size (default: 1024)
- `--extensions` - file types to index

#### PDF Indexing

```bash
# Synchronous version
python pdf_pipeline.py --pdf_path document.pdf

# Asynchronous (GPU optimized)
python pdf_pipeline_async.py --pdf_path document.pdf --max_concurrent 10

# GPU script
python run_pipeline_gpu.py --pdf_path document.pdf --use_async --max_concurrent 15
```

**Parameters**:
- `--pdf_path` - path to PDF (required)
- `--chunk_size` - chunk size (default: 1024)
- `--overlap` - overlap (default: 50)
- `--collection_name` - Qdrant collection (default: pdf_chunks)
- `--embedding_model` - Ollama model (default: qwen3-embedding:latest)
- `--ollama_host` - Ollama URL (default: http://localhost:11434)
- `--max_concurrent` - concurrent requests (default: 10)

### MCP Tools

Available tools when running chat:

**Weather Tools**:
- `get_weather` - current weather for city
- `get_weather_forecast` - multi-day forecast
- `convert_temperature` - temperature conversion
- `save_weather_data` - save weather data

**Git Tools** (new!):
- `git_get_current_branch` - current branch
- `git_get_branches` - list all branches
- `git_get_status` - working directory status
- `git_get_diff` - show changes (diff)
- `git_get_recent_commits` - recent commits
- `git_get_file_content` - file content
- `git_list_files` - repository files
- `git_get_repo_info` - repository information

**Other Tools**:
- `execute_python_code` - execute Python in Docker
- `stub_tool` - stub for testing

### Changes After Refactoring

#### Architectural Improvements:

1. **Modular Structure**: Code divided into logical modules
   - `src/utils/` - reusable utilities
   - `src/services/` - business logic
   - `src/clients/` - external service clients
   - `src/config.py` - centralized configuration

2. **Eliminated Duplication**:
   - Single logging module
   - Shared PDFChunker class
   - Shared Ollama client
   - Shared Qdrant client

3. **Improved Readability**:
   - Added type hints
   - English docstrings
   - Consistent naming

4. **Configuration**:
   - All constants in `src/config.py`
   - Centralized parameter settings

#### File Sizes:

| File | Before | After |
|------|--------|-------|
| main.py | ~1113 lines | ~587 lines |
| pdf_pipeline.py | ~312 lines | ~163 lines |
| pdf_pipeline_async.py | ~338 lines | ~181 lines |
