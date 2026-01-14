# FAQ - Frequently Asked Questions

## [English](#english-questions) | [Русский](#русские-вопросы)

---

<a name="русские-вопросы"></a>
## Русские вопросы

### Установка и настройка

**В: Какие зависимости требуются для работы проекта?**

О: Вам понадобятся:
- Python 3.10+
- Docker (для Qdrant)
- Ollama (для генерации эмбеддингов)
- Все зависимости из `requirements.txt`

Установка:
```bash
pip install -r requirements.txt
```

**В: Как настроить переменные окружения?**

О: Создайте файл `.env` в корне проекта со следующими переменными:

Обязательные:
```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.example.com/v1
OPEN_WEATHER=your_openweather_api_key
DOCKER_MCP_HOST=localhost
DOCKER_MCP_PORT=9011
```

Опциональные:
```
OPENAI_VERIFY_SSL=false
OLLAMA_NUM_PARALLEL=10
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_GPU_MEMORY=10240
```

**В: Как запустить необходимые сервисы (Qdrant, Ollama)?**

O: Для Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Для Ollama:
```bash
ollama serve
```

### Использование чат-клиента

**В: Как начать работу с чат-клиентом?**

O: Просто запустите:
```bash
python main.py
```

Клиент автоматически подключится к MCP серверу и будет готов к работе.

**В: Какие команды доступны в чате?**

О: Базовые команды:
- `quit` / `exit` - выход из программы
- `save <имя>` - сохранить разговор
- `load <имя>` - загрузить разговор
- `temp <0-2>` - установить температуру LLM
- `clear` - очистить историю
- `print` - показать историю

RAG команды:
- `/rag` - вкл/выкл RAG режим
- `/rag <вопрос>` - задать вопрос с RAG
- `/rag rerank` - вкл/выкл реранкинг
- `/rag threshold <0-1>` - установить порог релевантности

Help команды:
- `/help` - общая справка
- `/help <вопрос>` - поиск по документации
- `/help style` - стиль кода
- `/help api [компонент]` - справочник API
- `/help structure` - структура проекта
- `/help git` - текущий git-контекст

**В: Как работает RAG в чате?**

O: RAG (Retrieval Augmented Generation) дополняет ответы LLM контекстом из ваших документов. При включенном RAG режиме:
1. Ваш вопрос преобразуется в эмбеддинг
2. Векторная база (Qdrant) ищет релевантные фрагменты
3. Найденный контекст добавляется к запросу LLM

**В: Как использовать команду /help?**

O: Команда `/help` требует предварительной индексации документации:
```bash
# Шаг 1: Индексация документации
python index_docs.py

# Шаг 2: Использование в чате
python main.py
/help ваш_вопрос
```

### PDF обработка

**В: Как проиндексировать PDF документ?**

O: Три варианта:

Синхронный:
```bash
python pdf_pipeline.py --pdf_path document.pdf
```

Асинхронный (рекомендуется):
```bash
python pdf_pipeline_async.py --pdf_path document.pdf --max_concurrent 10
```

GPU-оптимизированный:
```bash
python run_pipeline_gpu.py --pdf_path document.pdf --use_async --max_concurrent 15
```

**В: Какие параметры доступны при индексации PDF?**

O:
- `--pdf_path` - путь к PDF (обязательно)
- `--chunk_size` - размер чанка (по умолчанию 1024)
- `--overlap` - перекрытие чанков (по умолчанию 50)
- `--collection_name` - название коллекции в Qdrant (pdf_chunks)
- `--embedding_model` - модель Ollama (qwen3-embedding:latest)
- `--ollama_host` - URL Ollama (http://localhost:6333)
- `--max_concurrent` - кол-во параллельных запросов (для асинхронной версии)

**В: Как оптимизировать PDF обработку для RTX 4070?**

O: Следуйте инструкциям в `GPU_SETUP.md`:

1. Установите CUDA Toolkit
2. Настройте переменные окружения Ollama:
```bash
set OLLAMA_NUM_PARALLEL=10
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_GPU_MEMORY=10240
```

3. Используйте асинхронную версию с оптимальными параметрами:
```bash
python pdf_pipeline_async.py --pdf_path doc.pdf --max_concurrent 15 --chunk_size 2048
```

### RAG и индексация

**В: Как проиндексировать документацию проекта?**

O: Для команды `/help` нужно проиндексировать документацию:

```bash
# Базовая индексация
python index_docs.py

# С параметрами
python index_docs.py --project_root . --collection_name project_docs --extensions .md .py .txt .yml .json
```

**В: Как проиндексировать код проекта?**

O: Для локального CI ревьюера:

```bash
python index_code.py
```

Это индексирует Python код по функциям и классам для контекстного поиска.

**В: Какие коллекции используются в Qdrant?**

O:
- `project_docs` - документация проекта (README, md, py файлы)
- `code_chunks` - Python код по функциям/классам
- `pdf_chunks` - индексированные PDF документы (пользовательские)

**В: Как настроить порог релевантности RAG?**

O: В чате:
```
/rag threshold 0.5
```

Или программно в `src/config.py`:
```python
DEFAULT_RELEVANCE_THRESHOLD = 0.5
DEFAULT_TOP_K = 5
```

### MCP инструменты

**В: Что такое MCP и зачем он нужен?**

O: MCP (Model Context Protocol) - протокол для интеграции внешних инструментов с LLM. В проекте используются два MCP сервера:

1. **STDIO сервер** (mcp_server.py) - локальный, предоставляет погодные и git инструменты
2. **Docker сервер** (опционально) - удаленный через HTTP

**В: Какие MCP инструменты доступны?**

О: Погодные инструменты:
- `get_weather(city_name)` - текущая погода
- `get_weather_forecast(city_name, days)` - прогноз
- `convert_temperature(value, from_unit, to_unit)` - конвертация температуры
- `save_weather_data(city_name, data)` - сохранение погодных данных

Git инструменты:
- `git_get_current_branch()` - текущая ветка
- `git_get_branches()` - список веток
- `git_get_status()` - статус изменений
- `git_get_diff()` - показать diff
- `git_get_recent_commits(count)` - последние коммиты
- `git_get_file_content(file_path)` - содержимое файла
- `git_list_files()` - файлы репозитория

### Локальный CI ревьюер

**В: Как работает локальный CI ревьюер?**

O: `local_ci_reviewer.py` анализирует изменения в коде:
1. Получает git diff относительно базовой ветки
2. Ищет релевантную документацию через RAG (collection: project_docs)
3. Ищет релевантный код через RAG (collection: code_chunks)
4. Генерирует структурированное ревью с помощью LLM

**В: Как запустить локальный CI ревьюер?**

O: Подготовка:
```bash
# Индексация документации (обязательно)
python index_docs.py

# Индексация кода (рекомендуется)
python index_code.py
```

Запуск:
```bash
# Базовый запуск
python local_ci_reviewer.py

# С параметрами
python local_ci_reviewer.py --base origin/master --threshold 0.5 --top_k 10 --output my_review.md
```

**В: Какие секции включает сгенерированное ревью?**

O:
- **Обзор** (Summary) - краткий обзор изменений
- **Сильные стороны** (Strengths) - что сделано хорошо
- **Проблемы** (Concerns & Issues) - потенциальные проблемы
- **Рекомендации** (Suggestions) - конкретные улучшения
- **Проверка документации** (Documentation Check) - проверка наличия документации
- **Рекомендации по тестированию** (Testing Considerations) - рекомендации по тестам

### Troubleshooting

**В: Ollama не отвечает при генерации эмбеддингов. Что делать?**

O: Проверьте:
1. Ollama запущен: `ollama list`
2. Модель загружена: `ollama pull qwen3-embedding:latest`
3. URL корректен: http://localhost:11434
4. Проверьте логи Ollama

**В: Qdrant недоступен. Как исправить?**

O: Проверьте:
1. Docker контейнер запущен: `docker ps`
2. Порт доступен: `curl http://localhost:6333/health`
3. Перезапустите Qdrant при необходимости

**В: Команда /help не работает. Почему?**

O: Убедитесь:
1. Документация проиндексирована: `python index_docs.py`
2. Коллекция `project_docs` существует в Qdrant
3. Ollama запущен для генерации эмбеддингов

**В: MCP сервер не подключается. Что проверить?**

O: Для STDIO сервера:
1. Python путь в `mcp_client.py` корректный
2. `mcp_server.py` существует и исполняемый
3. Логи проверяйте в stderr

Для Docker сервера:
1. DOCKER_MCP_HOST и PORT настроены в .env
2. MCP сервер запущен в Docker

**В: Как проверить, что GPU используется для эмбеддингов?**

O: Мониторинг GPU:
```bash
nvidia-smi -l 1
```

Использование GPU память должно расти при генерации эмбеддингов. Также проверьте логи Ollama.

### Производительность

**В: Как ускорить индексацию PDF?**

O:
1. Используйте асинхронную версию
2. Увеличьте `max_concurrent` до 15-20 для RTX 4070
3. Увеличьте `chunk_size` до 2048 или 4096
4. Настройте Ollama для GPU (см. GPU_SETUP.md)

**В: Рекомендации по размеру чанков?**

O:
- Тексты на русском: 1024-2048 символов
- Техническая документация: 2048-4096
- Код: 512-1024 (код индексируется отдельно по функциям)

**В: Какое значение max_concurrent оптимально?**

O: Зависит от видеокарты:
- RTX 4070 (12GB): 10-15
- RTX 4080/4090: 15-20
- Без GPU: 2-5

---

<a name="english-questions"></a>
## English Questions

### Installation and Setup

**Q: What dependencies are required to run the project?**

A: You need:
- Python 3.10+
- Docker (for Qdrant)
- Ollama (for embedding generation)
- All dependencies from `requirements.txt`

Installation:
```bash
pip install -r requirements.txt
```

**Q: How do I configure environment variables?**

A: Create a `.env` file in the project root with these variables:

Required:
```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.example.com/v1
OPEN_WEATHER=your_openweather_api_key
DOCKER_MCP_HOST=localhost
DOCKER_MCP_PORT=9011
```

Optional:
```
OPENAI_VERIFY_SSL=false
OLLAMA_NUM_PARALLEL=10
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_GPU_MEMORY=10240
```

**Q: How do I start required services (Qdrant, Ollama)?**

A: For Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

For Ollama:
```bash
ollama serve
```

### Using the Chat Client

**Q: How do I start using the chat client?**

A: Simply run:
```bash
python main.py
```

The client will automatically connect to the MCP server and be ready for use.

**Q: What commands are available in chat?**

A: Basic commands:
- `quit` / `exit` - exit the program
- `save <name>` - save conversation
- `load <name>` - load conversation
- `temp <0-2>` - set LLM temperature
- `clear` - clear history
- `print` - show history

RAG commands:
- `/rag` - toggle RAG mode
- `/rag <question>` - ask a question with RAG
- `/rag rerank` - toggle reranking
- `/rag threshold <0-1>` - set relevance threshold

Help commands:
- `/help` - general help
- `/help <question>` - search documentation
- `/help style` - code style guidelines
- `/help api [component]` - API reference
- `/help structure` - project structure
- `/help git` - current git context

**Q: How does RAG work in chat?**

A: RAG (Retrieval Augmented Generation) enhances LLM responses with context from your documents. When RAG mode is enabled:
1. Your question is converted to an embedding
2. Vector database (Qdrant) searches for relevant fragments
3. Found context is added to the LLM request

**Q: How do I use the /help command?**

A: The `/help` command requires prior documentation indexing:
```bash
# Step 1: Index documentation
python index_docs.py

# Step 2: Use in chat
python main.py
/help your_question
```

### PDF Processing

**Q: How do I index a PDF document?**

A: Three options:

Synchronous:
```bash
python pdf_pipeline.py --pdf_path document.pdf
```

Asynchronous (recommended):
```bash
python pdf_pipeline_async.py --pdf_path document.pdf --max_concurrent 10
```

GPU-optimized:
```bash
python run_pipeline_gpu.py --pdf_path document.pdf --use_async --max_concurrent 15
```

**Q: What parameters are available for PDF indexing?**

A:
- `--pdf_path` - path to PDF (required)
- `--chunk_size` - chunk size (default 1024)
- `--overlap` - chunk overlap (default 50)
- `--collection_name` - Qdrant collection name (pdf_chunks)
- `--embedding_model` - Ollama model (qwen3-embedding:latest)
- `--ollama_host` - Ollama URL (http://localhost:11434)
- `--max_concurrent` - concurrent requests (for async version)

**Q: How do I optimize PDF processing for RTX 4070?**

A: Follow instructions in `GPU_SETUP.md`:

1. Install CUDA Toolkit
2. Configure Ollama environment variables:
```bash
set OLLAMA_NUM_PARALLEL=10
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_GPU_MEMORY=10240
```

3. Use async version with optimal parameters:
```bash
python pdf_pipeline_async.py --pdf_path doc.pdf --max_concurrent 15 --chunk_size 2048
```

### RAG and Indexing

**Q: How do I index project documentation?**

A: For `/help` command, index documentation:

```bash
# Basic indexing
python index_docs.py

# With parameters
python index_docs.py --project_root . --collection_name project_docs --extensions .md .py .txt .yml .json
```

**Q: How do I index project code?**

A: For local CI reviewer:

```bash
python index_code.py
```

This indexes Python code by functions and classes for contextual search.

**Q: What collections are used in Qdrant?**

A:
- `project_docs` - project documentation (README, md, py files)
- `code_chunks` - Python code by functions/classes
- `pdf_chunks` - indexed PDF documents (user-defined)

**Q: How do I configure RAG relevance threshold?**

A: In chat:
```
/rag threshold 0.5
```

Or programmatically in `src/config.py`:
```python
DEFAULT_RELEVANCE_THRESHOLD = 0.5
DEFAULT_TOP_K = 5
```

### MCP Tools

**Q: What is MCP and why is it needed?**

A: MCP (Model Context Protocol) - protocol for integrating external tools with LLM. The project uses two MCP servers:

1. **STDIO server** (mcp_server.py) - local, provides weather and git tools
2. **Docker server** (optional) - remote via HTTP

**Q: What MCP tools are available?**

A: Weather tools:
- `get_weather(city_name)` - current weather
- `get_weather_forecast(city_name, days)` - forecast
- `convert_temperature(value, from_unit, to_unit)` - temperature conversion
- `save_weather_data(city_name, data)` - save weather data

Git tools:
- `git_get_current_branch()` - current branch
- `git_get_branches()` - list branches
- `git_get_status()` - working tree status
- `git_get_diff()` - show changes
- `git_get_recent_commits(count)` - recent commits
- `git_get_file_content(file_path)` - file content
- `git_list_files()` - repository files

### Local CI Reviewer

**Q: How does the local CI reviewer work?**

A: `local_ci_reviewer.py` analyzes code changes:
1. Gets git diff relative to base branch
2. Searches relevant documentation via RAG (collection: project_docs)
3. Searches relevant code via RAG (collection: code_chunks)
4. Generates structured review with LLM

**Q: How do I run the local CI reviewer?**

A: Preparation:
```bash
# Index documentation (required)
python index_docs.py

# Index code (recommended)
python index_code.py
```

Run:
```bash
# Basic run
python local_ci_reviewer.py

# With parameters
python local_ci_reviewer.py --base origin/master --threshold 0.5 --top_k 10 --output my_review.md
```

**Q: What sections does the generated review include?**

A:
- **Overview** (Summary) - brief overview of changes
- **Strengths** - what was done well
- **Concerns & Issues** - potential problems
- **Suggestions** - specific improvements
- **Documentation Check** - documentation verification
- **Testing Considerations** - testing recommendations

### Troubleshooting

**Q: Ollama doesn't respond during embedding generation. What to do?**

A: Check:
1. Ollama is running: `ollama list`
2. Model is loaded: `ollama pull qwen3-embedding:latest`
3. URL is correct: http://localhost:11434
4. Check Ollama logs

**Q: Qdrant is unavailable. How to fix?**

A: Check:
1. Docker container is running: `docker ps`
2. Port is accessible: `curl http://localhost:6333/health`
3. Restart Qdrant if necessary

**Q: /help command doesn't work. Why?**

A: Ensure:
1. Documentation is indexed: `python index_docs.py`
2. Collection `project_docs` exists in Qdrant
3. Ollama is running for embedding generation

**Q: MCP server won't connect. What to check?**

A: For STDIO server:
1. Python path in `mcp_client.py` is correct
2. `mcp_server.py` exists and is executable
3. Check logs in stderr

For Docker server:
1. DOCKER_MCP_HOST and PORT are configured in .env
2. MCP server is running in Docker

**Q: How do I verify GPU is being used for embeddings?**

A: Monitor GPU:
```bash
nvidia-smi -l 1
```

GPU memory usage should increase during embedding generation. Also check Ollama logs.

### Performance

**Q: How do I speed up PDF indexing?**

A:
1. Use async version
2. Increase `max_concurrent` to 15-20 for RTX 4070
3. Increase `chunk_size` to 2048 or 4096
4. Configure Ollama for GPU (see GPU_SETUP.md)

**Q: What are the recommended chunk sizes?**

A:
- Russian text: 1024-2048 characters
- Technical documentation: 2048-4096
- Code: 512-1024 (code is indexed separately by functions)

**Q: What's the optimal max_concurrent value?**

A: Depends on GPU:
- RTX 4070 (12GB): 10-15
- RTX 4080/4090: 15-20
- No GPU: 2-5
